"""Config-driven CPU training, genuine local MLflow, profiling and restart state."""
from pathlib import Path
import argparse
import contextlib
import copy
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import threading
import time

import mlflow
import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import yaml

from model import CNN

LAB = Path(__file__).resolve().parents[1]
RESULTS = LAB / 'results'


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_config(path=None):
    path = Path(path or LAB / 'config.yaml').resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert config['device'] == 'cpu', 'Recorded memory/timing protocol is CPU-only.'
    assert config['model'] == 'CNN'
    assert config['dataset']['train_size'] + config['dataset']['validation_size'] == 60000
    assert config['training']['checkpoint_every'] == 1
    config['_config_path'] = str(path)
    return config


def public_config(config):
    """Internal source locations can change when the lab is copied."""
    return {k:v for k,v in config.items() if not k.startswith('_')}


def configure(config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_data(config):
    d = config['dataset']
    root = (Path(config['_config_path']).parent / d['root']).resolve()
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((d['mean'],), (d['std'],))])
    full = datasets.FashionMNIST(root, train=True, download=d['download'], transform=transform)
    test = datasets.FashionMNIST(root, train=False, download=d['download'], transform=transform)
    train, validation = random_split(full, [d['train_size'], d['validation_size']],
                                    generator=torch.Generator().manual_seed(config['seed']))
    assert len(test) == 10000
    return train, validation, test


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(dataset, batch_size, workers, shuffle, generator=None):
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      shuffle=shuffle, generator=generator, worker_init_fn=seed_worker,
                      persistent_workers=workers > 0, pin_memory=False, drop_last=False)


class MemorySampler:
    """Sample whole process-tree RSS; also capture Windows main-process peak working set."""
    def __init__(self, interval):
        self.interval = interval
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.baseline = self.process.memory_info().rss
        self.peak_main = self.baseline
        self.peak_tree = self.baseline
        self.samples = 0

    def sample(self):
        try:
            main = self.process.memory_info().rss
            total = main
            for child in self.process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak_main = max(self.peak_main, main)
            self.peak_tree = max(self.peak_tree, total)
            self.samples += 1
        except psutil.NoSuchProcess:
            pass

    def loop(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.sample()

    def result(self):
        native_peak = getattr(self.process.memory_info(), 'peak_wset', self.peak_main)
        return dict(baseline_rss_mib=self.baseline / 2**20,
                    peak_main_rss_mib=self.peak_main / 2**20,
                    peak_tree_rss_mib=self.peak_tree / 2**20,
                    native_process_peak_mib=native_peak / 2**20,
                    sampled_increment_mib=(self.peak_main - self.baseline) / 2**20,
                    memory_samples=self.samples, sample_interval_seconds=self.interval)


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss_sum = correct = count = 0
    for x, y in loader:
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        count += len(y)
    return dict(loss=loss_sum / count, accuracy=correct / count, correct=correct, samples=count)


def train_epoch(model, loader, optimizer, accumulation=1, profile=False):
    """Sum loss gradients, divide by actual group size, then step (handles partial tails)."""
    model.train()
    totals = dict(data_loading=0.0, forward=0.0, backward=0.0, optimizer_step=0.0)
    loss_sum = correct = count = group_samples = updates = 0
    iterator = None
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    for i in range(len(loader)):
        t = time.perf_counter()
        with torch.profiler.record_function('phase/data_loading') if profile else contextlib.nullcontext():
            if iterator is None:
                iterator = iter(loader)
            x, y = next(iterator)
        totals['data_loading'] += time.perf_counter() - t
        t = time.perf_counter()
        with torch.profiler.record_function('phase/forward') if profile else contextlib.nullcontext():
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, reduction='sum')
        totals['forward'] += time.perf_counter() - t
        t = time.perf_counter()
        with torch.profiler.record_function('phase/backward') if profile else contextlib.nullcontext():
            loss.backward()
        totals['backward'] += time.perf_counter() - t
        group_samples += len(y)
        if (i + 1) % accumulation == 0 or i + 1 == len(loader):
            t = time.perf_counter()
            with torch.profiler.record_function('phase/optimizer_step') if profile else contextlib.nullcontext():
                for p in model.parameters():
                    p.grad.div_(group_samples)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            totals['optimizer_step'] += time.perf_counter() - t
            group_samples = 0
            updates += 1
        loss_sum += loss.detach().item()
        correct += (logits.detach().argmax(1) == y).sum().item()
        count += len(y)
    seconds = time.perf_counter() - start
    return dict(train_loss=loss_sum / count, train_accuracy=correct / count,
                train_seconds=seconds, throughput=count / seconds, samples=count,
                optimizer_steps=updates, **{k + '_seconds': v for k, v in totals.items()})


def optimizer_scheduler(model, config, settings):
    t = config['training']
    if settings['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=t['adam_lr'], weight_decay=t['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=t['sgd_lr'], momentum=t['momentum'],
                                    weight_decay=t['weight_decay'])
    name = settings['scheduler']
    if name == 'none':
        scheduler = None
    else:
        cls = getattr(torch.optim.lr_scheduler, name)
        scheduler = cls(optimizer, **config['schedulers'][name])
    return optimizer, scheduler


def checkpoint(path, epoch, model, optimizer, scheduler, generator, history, config, settings, run_id):
    value = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 scheduler=scheduler.state_dict() if scheduler else None,
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                 history=history, config=config, settings=settings, mlflow_run_id=run_id)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def restore(path, model, optimizer, scheduler, generator):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if scheduler:
        scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    generator.set_state(state['loader_rng'])
    return state


def flatten(config, prefix=''):
    out = {}
    for key, value in config.items():
        if key.startswith('_'):
            continue
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, name))
        else:
            out[name] = str(value)
    return out


def run_experiment(config, spec):
    configure(config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    folder = RESULTS / 'runs' / spec['name']
    folder.mkdir(parents=True, exist_ok=True)
    settings = copy.deepcopy(config['training'])
    settings.update(spec)
    train, validation, test = load_data(config)
    generator = torch.Generator().manual_seed(config['seed'])
    loader = make_loader(train, settings['batch_size'], settings['num_workers'], True, generator)
    val_loader = make_loader(validation, 512, 0, False)
    model = CNN()
    assert sum(p.numel() for p in model.parameters()) == 20490
    optimizer, scheduler = optimizer_scheduler(model, config, settings)
    history, start_epoch, run_id = [], 0, None
    if spec.get('resume'):
        state = restore(folder / 'checkpoint_epoch3.pt', model, optimizer, scheduler, generator)
        assert public_config(state['config']) == public_config(config), 'Resume config differs from checkpoint.'
        for key in ['batch_size', 'num_workers', 'optimizer', 'scheduler', 'accumulation_steps', 'epochs']:
            assert state['settings'][key] == settings[key], key
        history, start_epoch, run_id = state['history'], state['epoch'], state['mlflow_run_id']
    tracking = (LAB / config['tracking']['directory']).resolve()
    mlflow.set_tracking_uri(tracking.as_uri())
    mlflow.set_experiment(config['tracking']['experiment'])
    prof = None
    with mlflow.start_run(run_id=run_id, run_name=spec['name']) as run:
        run_id = run.info.run_id
        mlflow.set_tags(dict(student_id=config['student']['id'], model='Lab3_CNN',
                             study=spec['study'], assisted_preparation='true'))
        if not spec.get('resume'):
            mlflow.log_params(flatten(config))
            mlflow.log_params({'effective.' + k: str(v) for k, v in settings.items()})
            mlflow.log_artifact(config['_config_path'], 'configuration')
            for source in [Path(__file__), LAB / 'src/model.py']:
                mlflow.log_artifact(str(source), 'source')
        if spec.get('resume'):
            mlflow.set_tag('resumed_from_epoch', start_epoch)
        with MemorySampler(config['memory']['sample_interval_seconds']) as memory:
            for epoch in range(start_epoch + 1, settings['epochs'] + 1):
                lr = optimizer.param_groups[0]['lr']
                if spec.get('profile'):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                record_shapes=False, profile_memory=False) as prof:
                        row = train_epoch(model, loader, optimizer, settings['accumulation_steps'], True)
                else:
                    row = train_epoch(model, loader, optimizer, settings['accumulation_steps'])
                val = evaluate(model, val_loader)
                if scheduler:
                    if settings['scheduler'] == 'ReduceLROnPlateau':
                        scheduler.step(val['loss'])
                    else:
                        scheduler.step()
                row.update(epoch=epoch, lr=lr, next_lr=optimizer.param_groups[0]['lr'],
                           validation_loss=val['loss'], validation_accuracy=val['accuracy'])
                history.append(row)
                mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, (int, float))}, step=epoch)
                checkpoint(folder / 'checkpoint.pt', epoch, model, optimizer, scheduler, generator,
                           history, config, settings, run_id)
                if spec.get('interrupt_after') == epoch:
                    checkpoint(folder / 'checkpoint_epoch3.pt', epoch, model, optimizer, scheduler,
                               generator, history, config, settings, run_id)
                    write_csv(folder / 'history.csv', history)
                    mlflow.set_tag('interrupted_at_epoch', epoch)
                    mlflow.log_artifact(str(folder / 'checkpoint_epoch3.pt'), 'checkpoints')
                    print(f"INTERRUPTED {spec['name']} after epoch {epoch}; saved complete state", flush=True)
                    return
                print(f"{spec['name']}: {epoch}/{settings['epochs']} loss={row['train_loss']:.4f} "
                      f"val={val['accuracy']:.2%} {row['train_seconds']:.2f}s", flush=True)
            final_test = evaluate(model, make_loader(test, 512, 0, False)) if spec.get('test') else None
        write_csv(folder / 'history.csv', history)
        write_json(folder / 'effective_config.json', dict(config=config, settings=settings))
        summary = dict(name=spec['name'], study=spec['study'], settings=settings, mlflow_run_id=run_id,
                       epochs_completed=len(history), final_validation_accuracy=history[-1]['validation_accuracy'],
                       final_validation_loss=history[-1]['validation_loss'], final_test=final_test,
                       throughput=sum(h['samples'] for h in history) / sum(h['train_seconds'] for h in history),
                       memory=memory.result())
        if prof is not None:
            phases = []
            for event in prof.key_averages():
                if event.key.startswith('phase/'):
                    phases.append(dict(phase=event.key.split('/')[1], seconds=event.cpu_time_total / 1e6,
                                       calls=event.count))
            total = sum(p['seconds'] for p in phases)
            for p in phases:
                p['share_of_four_phases_pct'] = 100 * p['seconds'] / total
                p['share_of_epoch_wall_pct'] = 100 * p['seconds'] / history[-1]['train_seconds']
            write_csv(RESULTS / 'profile_breakdown.csv', phases)
            summary['profile_other_wall_seconds'] = history[-1]['train_seconds'] - total
            (RESULTS / 'profile_operators.txt').write_text(
                prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=60), encoding='utf-8')
            trace = folder / 'profile_trace.json'
            prof.export_chrome_trace(str(trace))
            with trace.open('rb') as source, gzip.open(RESULTS / 'profile_trace.json.gz', 'wb') as dest:
                import shutil
                shutil.copyfileobj(source, dest)
            trace.unlink()
            mlflow.log_artifact(str(RESULTS / 'profile_breakdown.csv'), 'profile')
            mlflow.log_artifact(str(RESULTS / 'profile_trace.json.gz'), 'profile')
        write_json(folder / 'summary.json', summary)
        mlflow.log_metrics({'summary.throughput': summary['throughput'],
                            'summary.peak_tree_rss_mib': summary['memory']['peak_tree_rss_mib']})
        if final_test:
            mlflow.log_metrics({'test.' + k: v for k, v in final_test.items()})
        for file in ['history.csv', 'summary.json', 'effective_config.json']:
            mlflow.log_artifact(str(folder / file), 'evidence')
        mlflow.log_artifact(str(folder / 'checkpoint.pt'), 'checkpoints')
    return summary


def environment(config):
    installed = dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))
    write_json(RESULTS / 'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
        processor=platform.processor(), logical_cpus=os.cpu_count(), executable=os.sys.executable,
        torch=torch.__version__, device='cpu', threads=config['threads'], packages=installed,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    (LAB / 'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in installed.items())+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(LAB / 'config.yaml'))
    parser.add_argument('--spec', required=True, help='JSON experiment specification from orchestrator')
    args = parser.parse_args()
    run_experiment(load_config(args.config), json.loads(Path(args.spec).read_text(encoding='utf-8')))
