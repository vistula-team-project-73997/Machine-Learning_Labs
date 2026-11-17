"""Full-data controlled compression experiments and repeated CPU timing."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from models import CNN, magnitude_masks, apply_masks, structured_compact, distillation_loss, compute_counts, weight_parameters
from storage import save_storage

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / 'results'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(config):
    files = ['models.py', 'run_lab.py', 'storage.py']
    obj = {'config': config, 'source': {n: sha(LAB/'src'/n) for n in files}}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest(), obj


def configure(config):
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def data(config):
    root = (LAB / config['dataset_root']).resolve()
    full = FashionMNIST(root, train=True, download=config['download'])
    test = FashionMNIST(root, train=False, download=config['download'])
    order = torch.randperm(len(full), generator=torch.Generator().manual_seed(config['seed']))
    train_i, val_i = order[:config['train_size']], order[config['train_size']:]
    assert (len(full), len(test), len(train_i), len(val_i)) == (60000, 10000, 55000, 5000)
    np.savez(RESULTS/'split_indices.npz', train=train_i.numpy(), validation=val_i.numpy())
    def transform(x):
        return (x.unsqueeze(1).float().div(255) - config['mean']) / config['std']
    # Identical pixel scaling to ToTensor + Normalize, cached once in RAM.
    x = transform(full.data); xt = transform(test.data)
    return (x[train_i], full.targets[train_i]), (x[val_i], full.targets[val_i]), (xt, test.targets)


@torch.inference_mode()
def evaluate(model, dataset):
    model.eval(); x, y = dataset; correct = 0; loss = 0.
    for start in range(0, len(y), 256):
        logits = model(x[start:start+256]); labels = y[start:start+256]
        correct += int((logits.argmax(1) == labels).sum())
        loss += float(F.cross_entropy(logits, labels, reduction='sum'))
    return {'accuracy': correct/len(y), 'loss': loss/len(y), 'correct': correct, 'samples': len(y)}


def train(model, training, validation, config, epochs, lr, masks=None, teacher_logits=None):
    generator = torch.Generator().manual_seed(config['seed']+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x, y = training; history = []; handles = []
    if masks:
        for name, p in model.named_parameters():
            if name in masks:
                handles.append(p.register_hook(lambda grad, mask=masks[name]: grad*mask))
        apply_masks(model, masks)
    wall_start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train(); start = time.perf_counter(); permutation = torch.randperm(len(y), generator=generator)
        loss_sum = 0.; ce_sum = 0.; kl_sum = 0.; correct = 0; steps = 0
        for offset in range(0, len(y), config['batch_size']):
            ids = permutation[offset:offset+config['batch_size']]
            optimizer.zero_grad(set_to_none=True); logits = model(x[ids])
            if teacher_logits is not None:
                loss, ce, kl = distillation_loss(logits, teacher_logits[ids], y[ids], config['temperature'], config['alpha'])
            else:
                ce = F.cross_entropy(logits, y[ids]); loss = ce; kl = torch.tensor(0.)
            loss.backward(); optimizer.step()
            if masks:
                apply_masks(model, masks)  # Also suppress optimizer-state regrowth.
            count = len(ids); loss_sum += float(loss.detach())*count
            ce_sum += float(ce.detach())*count; kl_sum += float(kl.detach())*count
            correct += int((logits.detach().argmax(1) == y[ids]).sum()); steps += 1
        seconds = time.perf_counter()-start
        if masks:
            assert all(int(torch.count_nonzero(dict(model.named_parameters())[n].detach()*(1-m))) == 0 for n,m in masks.items())
        val = evaluate(model, validation)
        row = dict(epoch=epoch, samples=len(y), steps=steps, objective=loss_sum/len(y),
                   cross_entropy=ce_sum/len(y), kl=kl_sum/len(y), train_accuracy=correct/len(y),
                   validation_accuracy=val['accuracy'], validation_loss=val['loss'], train_seconds=seconds,
                   masked_nonzero_violations=0, learning_rate=lr)
        history.append(row)
        print(f"  epoch {epoch}/{epochs}: CE={row['cross_entropy']:.4f}, val={val['accuracy']:.2%}, train={seconds:.2f}s", flush=True)
    for handle in handles:
        handle.remove()
    return history, time.perf_counter()-wall_start


def plan(config):
    jobs = [dict(name='teacher', method='baseline', target=0., epochs=config['teacher_epochs'])]
    jobs += [dict(name=f'{method}_{s:g}', method=method, target=s, epochs=config['finetune_epochs'])
             for method in ('global', 'layerwise') for s in config['sparsities']]
    jobs += [dict(name=f'structured_{s:g}', method='structured', target=s, epochs=config['finetune_epochs'])
             for s in config['filter_removal']]
    jobs += [dict(name='student_plain', method='plain', target=0., epochs=config['student_epochs']),
             dict(name='student_distilled', method='distilled', target=0., epochs=config['student_epochs'])]
    return jobs


def load_model(name):
    checkpoint = torch.load(RESULTS/'runs'/name/'model.pt', map_location='cpu', weights_only=True)
    model = CNN(*checkpoint['arch']); model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model


def run_experiment(job, training, validation, testing, config, digest, teacher_logits):
    folder = RESULTS/'runs'/job['name']; folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config['seed']); masks = {}; details = {}
    if job['method'] == 'baseline':
        model = CNN(*config['teacher_arch'])
    elif job['method'] in ('plain', 'distilled'):
        model = CNN(*config['student_arch'])
    else:
        model = load_model('teacher')
        details['teacher_checkpoint_sha256'] = sha(RESULTS/'runs/teacher/model.pt')
        if job['method'] in ('global', 'layerwise'):
            masks = magnitude_masks(model, job['target'], job['method']); apply_masks(model, masks)
            torch.save(masks, folder/'masks.pt')
        else:
            model, reference, keep = structured_compact(model, job['target'])
            with torch.no_grad():
                probe = validation[0][:32]
                diff = float((model(probe)-reference(probe)).abs().max())
            assert diff < 2e-4, diff
            details.update(kept_filters=keep, compact_reference_max_error=diff)
    initial_sha = hashlib.sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    before = evaluate(model, validation)
    lr = config['finetune_learning_rate'] if job['method'] in ('global','layerwise','structured') else config['learning_rate']
    history, wall = train(model, training, validation, config, job['epochs'], lr, masks,
                          teacher_logits if job['method'] == 'distilled' else None)
    val = evaluate(model, validation); test = evaluate(model, testing)
    torch.save({'arch': model.arch, 'state_dict': model.state_dict()}, folder/'model.pt')
    storage = save_storage(model, folder)
    weights = weight_parameters(model); weight_count = sum(p.numel() for p in weights.values())
    zero_weights = sum(int((p == 0).sum()) for p in weights.values())
    params = sum(p.numel() for p in model.parameters()); mac, potential = compute_counts(model)
    layer_rows = []
    for name, p in weights.items():
        layer_rows.append(dict(layer=name, weights=p.numel(), zeros=int((p == 0).sum()),
                               sparsity=float((p == 0).float().mean())))
    write_csv(folder/'layer_sparsity.csv', layer_rows); write_csv(folder/'epochs.csv', history)
    result = dict(**job, arch=model.arch, fingerprint=digest, initial_sha256=initial_sha,
                  parameters=params, nonzero_parameters=sum(int(torch.count_nonzero(p)) for p in model.parameters()),
                  weight_count=weight_count, zero_weights=zero_weights, actual_weight_sparsity=zero_weights/weight_count,
                  pre_validation_accuracy=before['accuracy'], validation_accuracy=val['accuracy'],
                  test_accuracy=test['accuracy'], test_correct=test['correct'], test_samples=test['samples'],
                  train_seconds=sum(r['train_seconds'] for r in history), train_validation_wall_seconds=wall,
                  dense_mac=mac, potential_nonzero_mac=potential, estimated_dense_flops=2*mac,
                  checkpoint_sha256=sha(folder/'model.pt'), **storage, **details)
    write_json(folder/'summary.json', result)
    print(f"COMPLETE {job['name']}: test={test['accuracy']:.2%}, parameters={params:,}", flush=True)
    return result


@torch.inference_mode()
def benchmark(config, summaries, testing):
    b = config['benchmark']; rng = random.Random(config['seed']+100); rows = []; blocks = []
    models = {r['name']: load_model(r['name']) for r in summaries}
    for batch, repeats in zip(b['batches'], b['repeats']):
        x = testing[0][:batch].contiguous()
        for block in range(b['blocks']):
            names = list(models); rng.shuffle(names)
            for position, name in enumerate(names):
                model = models[name]
                for _ in range(b['warmup']): model(x)
                timings = []
                for repeat in range(repeats):
                    start = time.perf_counter_ns(); model(x); elapsed = (time.perf_counter_ns()-start)/1e6
                    timings.append(elapsed)
                    rows.append(dict(model=name, batch=batch, block=block, position=position, repeat=repeat, milliseconds=elapsed))
                blocks.append(dict(model=name, batch=batch, block=block, median_ms=float(np.median(timings))))
            print(f'BENCHMARK batch={batch}, block={block+1}/{b["blocks"]}', flush=True)
    write_csv(RESULTS/'latency_raw.csv', rows); write_csv(RESULTS/'latency_blocks.csv', blocks)
    final = []
    for result in summaries:
        result = dict(result)
        for batch in b['batches']:
            values = [r['milliseconds'] for r in rows if r['model']==result['name'] and r['batch']==batch]
            result[f'latency_b{batch}_p50_ms'] = float(np.median(values))
            result[f'latency_b{batch}_p95_ms'] = float(np.percentile(values,95))
            result[f'throughput_b{batch}_samples_s'] = batch/(np.median(values)/1000)
            result[f'latency_b{batch}_min_block_ms'] = min(r['median_ms'] for r in blocks if r['model']==result['name'] and r['batch']==batch)
            result[f'latency_b{batch}_max_block_ms'] = max(r['median_ms'] for r in blocks if r['model']==result['name'] and r['batch']==batch)
        for watts in config['energy_scenario_watts']:
            # watts * milliseconds = millijoules. Explicit scenario, no measurement.
            result[f'estimated_inference_mJ_at_{watts}W'] = watts*result['latency_b1_p50_ms']
            result[f'estimated_training_Wh_at_{watts}W'] = watts*result['train_validation_wall_seconds']/3600
        final.append(result)
    write_json(RESULTS/'summary.json', final)
    flat = [{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in final]
    keys = list(dict.fromkeys(k for row in flat for k in row))
    write_csv(RESULTS/'comparison.csv', [{k:r.get(k,'') for k in keys} for r in flat])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true', help='Retrain all experiments, refreshing results')
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--only', default='', help='One experiment; does not build the final report')
    args = parser.parse_args()
    config = json.loads((LAB/'config.json').read_text()); configure(config)
    RESULTS.mkdir(exist_ok=True); digest, provenance = fingerprint(config)
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                       processor=platform.processor(), logical_cpus=os.cpu_count(), threads=torch.get_num_threads(),
                       interop_threads=torch.get_num_interop_threads(), cuda=torch.cuda.is_available(),
                       packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','pandas','matplotlib')},
                       energy='Not measured: fixed-power scenarios only', fingerprint=digest, provenance=provenance)
    write_json(RESULTS/'environment.json', environment)
    training, validation, testing = data(config)
    check = subprocess.run([sys.executable,str(LAB/'src/test_compression.py')],capture_output=True,text=True)
    (RESULTS/'unit_verification.txt').write_text(check.stdout+check.stderr,encoding='utf-8')
    print(check.stdout+check.stderr,flush=True); check.check_returncode()
    jobs = plan(config); write_json(RESULTS/'experiment_plan.json',jobs)
    summaries = []; teacher_logits = None
    for index, job in enumerate(jobs):
        if args.only and job['name'] != args.only: continue
        folder = RESULTS/'runs'/job['name']; summary_path = folder/'summary.json'
        if summary_path.exists() and not args.rerun:
            result = json.loads(summary_path.read_text())
            assert result['fingerprint'] == digest, 'Experiment code/config changed; use --rerun to avoid mixed evidence.'
            assert sha(folder/'model.pt') == result['checkpoint_sha256']
            print('SKIP completed:',job['name'],flush=True)
        else:
            assert not args.benchmark_only, f'Missing experiment {job["name"]}'
            if job['method'] == 'distilled':
                teacher = load_model('teacher'); start = time.perf_counter()
                with torch.no_grad():
                    teacher_logits = torch.cat([teacher(training[0][i:i+256]) for i in range(0,len(training[1]),256)])
                teacher_logits = teacher_logits.detach()
                write_json(RESULTS/'teacher_cache.json',dict(seconds=time.perf_counter()-start,samples=len(teacher_logits),
                    bytes=teacher_logits.numel()*teacher_logits.element_size(),teacher_checkpoint_sha256=sha(RESULTS/'runs/teacher/model.pt'),
                    note='Frozen teacher logits for training split only; extra preparation cost, excluded from student epoch times.'))
            print(f'EXPERIMENT {index+1}/{len(jobs)}: {job["name"]}',flush=True)
            result = run_experiment(job,training,validation,testing,config,digest,teacher_logits)
        summaries.append(result)
    if not args.only:
        benchmark(config,summaries,testing)
        from summarize import summarize
        summarize()
        check = subprocess.run([sys.executable,str(LAB/'src/verify_results.py')],capture_output=True,text=True)
        (RESULTS/'verification.txt').write_text(check.stdout+check.stderr,encoding='utf-8')
        print(check.stdout+check.stderr,flush=True); check.check_returncode()


if __name__ == '__main__': main()
