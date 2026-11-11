"""Independent saved-evidence checks, MLflow checks and fresh model evaluation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch

from model import CNN
from pipeline import LAB, RESULTS, load_config, load_data, make_loader, evaluate
from tracking import relocate_store


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    config=load_config()
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan=read(RESULTS/'experiment_plan.json')
    names={j['name'] for j in plan}
    summaries={p.parent.name:read(p) for p in (RESULTS/'runs').glob('*/summary.json')}
    assert set(summaries)==names,(len(summaries),len(names))
    relocate_store(RESULTS/'mlruns')
    mlflow.set_tracking_uri((RESULTS/'mlruns').resolve().as_uri())
    client=MlflowClient()
    train,val,test=load_data(config)
    raw=(Path(config['_config_path']).parent/config['dataset']['root']).resolve()/'FashionMNIST/raw'
    expected_hashes=read(RESULTS/'data_sha256.json')
    for filename,digest in expected_hashes.items():
        assert hashlib.sha256((raw/filename).read_bytes()).hexdigest()==digest,filename
    split=np.load(LAB/'references/lab03_split_indices.npz')
    assert np.array_equal(train.indices,split['train'])
    assert np.array_equal(val.indices,split['validation'])
    assert set(train.indices).isdisjoint(val.indices)
    assert len(train)==55000 and len(val)==5000 and len(test)==10000
    rows=0
    for name,summary in sorted(summaries.items()):
        folder=RESULTS/'runs'/name
        state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        settings=summary['settings']
        with (folder/'history.csv').open() as f:
            history=list(csv.DictReader(f))
        assert len(history)==summary['epochs_completed']==settings['epochs']==state['epoch']
        assert [int(h['epoch']) for h in history]==list(range(1,len(history)+1))
        for h in history:
            assert int(h['samples'])==55000
            assert int(h['optimizer_steps'])==math.ceil(math.ceil(55000/settings['batch_size'])/settings['accumulation_steps'])
            assert math.isfinite(float(h['train_loss'])) and 0<=float(h['validation_accuracy'])<=1
        rows+=len(history)
        run=client.get_run(summary['mlflow_run_id'])
        assert run.info.status=='FINISHED'
        assert run.data.tags['mlflow.runName']==name
        assert run.data.params['effective.batch_size']==str(settings['batch_size'])
        assert run.data.params['effective.scheduler']==settings['scheduler']
        metrics=client.get_metric_history(run.info.run_id,'train_loss')
        assert [m.step for m in metrics]==list(range(1,len(history)+1))
        assert all(float(h['train_loss'])==m.value for h,m in zip(history,metrics))
        artifacts=client.list_artifacts(run.info.run_id,'checkpoints')
        assert any(a.path=='checkpoints/checkpoint.pt' for a in artifacts)
        downloaded=client.download_artifacts(run.info.run_id,'evidence/history.csv')
        assert Path(downloaded).read_bytes()==(folder/'history.csv').read_bytes()
        memory=summary['memory']
        assert memory['memory_samples']>1 and memory['native_process_peak_mib']>0
        if summary['final_test'] is not None:
            assert summary['final_test']['samples']==10000
            assert summary['final_test']['accuracy']==summary['final_test']['correct']/10000
        print(f'PASS {name}: {len(history)} epochs, MLflow params/metrics/artifacts and checkpoint',flush=True)
    ref=torch.load(RESULTS/'runs/resume_reference/checkpoint.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(RESULTS/'runs/resume_interrupted/checkpoint.pt',map_location='cpu',weights_only=False)
    for key in ['model','optimizer','scheduler','python_rng','numpy_rng','torch_rng','loader_rng']:
        assert equal(ref[key],resumed[key]),key
    assert read(RESULTS/'resume_comparison.json')['exact_match']
    assert read(RESULTS/'accumulation_comparison.json')['similar_behavior']
    with (RESULTS/'profile_breakdown.csv').open() as f:
        profile=list(csv.DictReader(f))
    assert {p['phase'] for p in profile}=={'data_loading','forward','backward','optimizer_step'}
    assert abs(sum(float(p['share_of_four_phases_pct']) for p in profile)-100)<1e-9
    assert (RESULTS/'profile_trace.json.gz').stat().st_size>1000
    for filename,key,expected in [('worker_study.csv','num_workers',[0,1,2,4,8]),
                                  ('batch_study.csv','batch_size',[32,64,128,256,512])]:
        with (RESULTS/filename).open() as f:
            actual=[int(r[key]) for r in csv.DictReader(f)]
        assert actual==expected
    # A fresh model instance must reproduce the saved batch-128 test result.
    state=torch.load(RESULTS/'runs/batch_128/checkpoint.pt',map_location='cpu',weights_only=False)
    model=CNN()
    model.load_state_dict(state['model'])
    score=evaluate(model,make_loader(test,512,0,False))
    assert score==summaries['batch_128']['final_test'],(score,summaries['batch_128']['final_test'])
    assert summaries['resume_interrupted']['mlflow_run_id']==resumed['mlflow_run_id']
    print(f'PASS all {len(names)} experiment runs / {rows} epoch records; full Lab 3 split preserved.')
    print('PASS actual process restart: model, optimizer, scheduler and all RNG states match exactly.')
    print('PASS accumulation behavior, profile shares, required grids and fresh checkpoint test evaluation.')


if __name__=='__main__':
    main()
