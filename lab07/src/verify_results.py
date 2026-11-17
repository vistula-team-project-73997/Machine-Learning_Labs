"""Verify complete experiment evidence and freshly reload every final model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from models import weight_parameters, magnitude_masks, compute_counts
from storage import load_csr
from run_lab import LAB, RESULTS, configure, data, evaluate, fingerprint, load_model, plan, sha


def main():
    config=json.loads((LAB/'config.json').read_text());configure(config)
    digest,_=fingerprint(config);results=json.loads((RESULTS/'summary.json').read_text())
    jobs=plan(config);assert len(results)==18 and {r['name'] for r in results}=={j['name'] for j in jobs}
    assert sum(r['epochs'] for r in results)==75
    split=np.load(RESULTS/'split_indices.npz');original_train=split['train'].copy();original_val=split['validation'].copy()
    assert len(np.intersect1d(original_train,original_val))==0
    assert np.array_equal(np.sort(np.r_[original_train,original_val]),np.arange(60000))
    _,validation,testing=data(config)
    split=np.load(RESULTS/'split_indices.npz')
    assert np.array_equal(original_train,split['train']) and np.array_equal(original_val,split['validation'])
    reference=LAB.parent/'lab06/references/lab03_split_indices.npz'
    if reference.exists():
        with np.load(reference) as old:
            assert np.array_equal(original_train,old['train']) and np.array_equal(original_val,old['validation'])
    print('PASS disjoint, complete 55,000/5,000 split and 10,000 test examples')
    teacher=load_model('teacher');teacher_hash=sha(RESULTS/'runs/teacher/model.pt');total_epochs=0
    for r in results:
        folder=RESULTS/'runs'/r['name'];assert r['fingerprint']==digest
        assert sha(folder/'model.pt')==r['checkpoint_sha256']
        local=json.loads((folder/'summary.json').read_text())
        assert all(r[k]==v for k,v in local.items())
        model=load_model(r['name']);epochs=pd.read_csv(folder/'epochs.csv');total_epochs+=len(epochs)
        assert len(epochs)==r['epochs'] and list(epochs.epoch)==list(range(1,r['epochs']+1))
        assert (epochs.samples==55000).all() and (epochs.steps==430).all()
        assert (epochs.masked_nonzero_violations==0).all()
        assert np.isfinite(epochs.select_dtypes('number')).all().all()
        assert sum(p.numel() for p in model.parameters())==r['parameters']
        assert compute_counts(model)==(r['dense_mac'],r['potential_nonzero_mac'])
        if r['method'] in ('global','layerwise'):
            masks=torch.load(folder/'masks.pt',weights_only=True)
            expected=magnitude_masks(teacher,r['target'],r['method'])
            assert all(torch.equal(masks[n],m) for n,m in expected.items())
            assert all(torch.count_nonzero(dict(model.named_parameters())[n]*(1-m))==0 for n,m in masks.items())
            assert r['teacher_checkpoint_sha256']==teacher_hash
            assert r['parameters']==1015370
        if r['method']=='structured':
            assert r['teacher_checkpoint_sha256']==teacher_hash and r['compact_reference_max_error']<2e-4
            assert model.arch[:2]==[16-round(16*r['target']),32-round(32*r['target'])]
        restored=load_csr(folder/'csr.npz')
        with np.load(folder/'dense.npz') as dense:
            for name,tensor in model.state_dict().items():
                assert np.array_equal(restored[name],tensor.numpy()) and np.array_equal(dense[name],tensor.numpy())
        for key,name in [('dense_npz_bytes','dense.npz'),('csr_npz_bytes','csr.npz'),('checkpoint_bytes','model.pt')]:
            assert r[key]==(folder/name).stat().st_size
        assert evaluate(model,testing)['correct']==r['test_correct']
        assert abs(evaluate(model,validation)['accuracy']-r['validation_accuracy'])<1e-12
        assert abs(r['test_accuracy']-r['test_correct']/10000)<1e-12
    assert total_epochs==75
    by={r['name']:r for r in results}
    assert by['teacher']['parameters']==1015370
    assert by['student_plain']['parameters']==by['student_distilled']['parameters']==103018
    assert by['student_plain']['initial_sha256']==by['student_distilled']['initial_sha256']
    assert by['global_0']['checkpoint_sha256']==by['layerwise_0']['checkpoint_sha256']
    print('PASS all 18 experiments / 75 full-data epoch records; every checkpoint re-evaluated')
    print('PASS fixed masks, exact parameter/MAC counts, physically compact structured models')
    print('PASS dense and CSR files restore exactly; measured on-disk sizes agree')
    print('PASS identical student initialization and reproducible zero-pruning controls')
    raw=pd.read_csv(RESULTS/'latency_raw.csv');blocks=pd.read_csv(RESULTS/'latency_blocks.csv')
    assert len(raw)==18*5*(100+20) and len(blocks)==18*5*2
    assert (raw.milliseconds>0).all() and np.isfinite(raw.milliseconds).all()
    for r in results:
        for batch,repeats in zip(config['benchmark']['batches'],config['benchmark']['repeats']):
            rows=raw[(raw.model==r['name'])&(raw.batch==batch)]
            assert len(rows)==5*repeats
            assert all(len(rows[rows.block==i])==repeats for i in range(5))
            assert abs(float(np.median(rows.milliseconds))-r[f'latency_b{batch}_p50_ms'])<1e-10
            assert abs(float(np.percentile(rows.milliseconds,95))-r[f'latency_b{batch}_p95_ms'])<1e-10
        for watts in config['energy_scenario_watts']:
            assert abs(r[f'estimated_inference_mJ_at_{watts}W']-watts*r['latency_b1_p50_ms'])<1e-10
    cache=json.loads((RESULTS/'teacher_cache.json').read_text())
    assert cache['samples']==55000 and cache['teacher_checkpoint_sha256']==teacher_hash
    assert config['kl_direction']=='student||teacher' and config['temperature']==4 and config['alpha']==.5
    print('PASS 10,800 latency observations / 180 randomized model blocks; aggregate quantiles agree')
    print('PASS frozen teacher provenance; T=4, alpha=0.5; energy fields explicitly estimated')
    print('PASS complete Lab 7 evidence verification')


if __name__=='__main__':main()
