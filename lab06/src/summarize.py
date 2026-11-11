"""Build tables, plots and report strictly from saved measurements."""
from pathlib import Path
import csv
import hashlib
import html
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch

from pipeline import LAB, RESULTS, load_config, write_csv, write_json
from tracking import relocate_store


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def history(name):
    with (RESULTS / 'runs' / name / 'history.csv').open(encoding='utf-8') as f:
        return [{k: float(v) for k,v in row.items()} for row in csv.DictReader(f)]


def table(rows, keys, formats=None):
    formats = formats or {}
    result = ['| ' + ' | '.join(keys) + ' |', '| ' + ' | '.join('---' for _ in keys) + ' |']
    for row in rows:
        result.append('| ' + ' | '.join(formats.get(k, str)(row[k]) for k in keys) + ' |')
    return '\n'.join(result)


def savefig(name):
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=160)
    plt.close()


def compare_states(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a)


def summarize(config=None):
    config = config or load_config()
    summaries = [read_json(p) for p in sorted((RESULTS / 'runs').glob('*/summary.json'))]
    by_name = {r['name']: r for r in summaries}
    workers, batches, optimizers = [], [], []
    for r in summaries:
        name, settings = r['name'], r['settings']
        h = history(name)
        if r['study'] == 'workers':
            steady = h[1:]
            workers.append(dict(num_workers=settings['num_workers'], cold_epoch_seconds=h[0]['train_seconds'],
                steady_epoch_seconds=statistics.median(e['train_seconds'] for e in steady),
                steady_samples_per_second=statistics.median(e['throughput'] for e in steady),
                steady_data_wait_seconds=statistics.median(e['data_loading_seconds'] for e in steady),
                peak_tree_rss_mib=r['memory']['peak_tree_rss_mib']))
        elif r['study'] == 'batch':
            batches.append(dict(batch_size=settings['batch_size'], epochs=r['epochs_completed'],
                samples_per_second=r['throughput'], peak_rss_mib=r['memory']['native_process_peak_mib'],
                sampled_increment_mib=r['memory']['sampled_increment_mib'],
                validation_accuracy=r['final_validation_accuracy'], test_accuracy=r['final_test']['accuracy'],
                test_correct=r['final_test']['correct'], test_samples=r['final_test']['samples']))
        elif r['study'] == 'optimizer':
            optimizers.append(dict(optimizer=settings['optimizer'], scheduler=settings['scheduler'],
                epochs=r['epochs_completed'], final_validation_loss=r['final_validation_loss'],
                final_validation_accuracy=r['final_validation_accuracy'],
                initial_lr=h[0]['lr'], final_used_lr=h[-1]['lr'], next_lr=h[-1]['next_lr'],
                total_train_seconds=sum(e['train_seconds'] for e in h)))
    workers.sort(key=lambda x:x['num_workers'])
    batches.sort(key=lambda x:x['batch_size'])
    optimizers.sort(key=lambda x:(x['optimizer'],x['scheduler']))
    write_csv(RESULTS / 'worker_study.csv', workers)
    write_csv(RESULTS / 'batch_study.csv', batches)
    write_csv(RESULTS / 'optimizer_scheduler_study.csv', optimizers)

    a = torch.load(RESULTS / 'runs/effective256/checkpoint.pt', map_location='cpu', weights_only=False)
    b = torch.load(RESULTS / 'runs/micro64_accum4/checkpoint.pt', map_location='cpu', weights_only=False)
    accumulation = dict(max_parameter_difference=compare_states(a['model'],b['model']),
        max_train_loss_difference=max(abs(x['train_loss']-y['train_loss']) for x,y in zip(a['history'],b['history'])),
        max_validation_accuracy_difference=max(abs(x['validation_accuracy']-y['validation_accuracy'])
                                                for x,y in zip(a['history'],b['history'])),
        effective_batch=256, micro_batch=64, accumulation_steps=4,
        epochs=len(a['history']), optimizer_steps_per_epoch=a['history'][0]['optimizer_steps'],
        final_partial_group_samples=55000 % 256)
    accumulation['similar_behavior'] = (accumulation['max_train_loss_difference'] < 0.02 and
                                         accumulation['max_validation_accuracy_difference'] <= 0.01)
    write_json(RESULTS / 'accumulation_comparison.json', accumulation)
    ref = torch.load(RESULTS / 'runs/resume_reference/checkpoint.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(RESULTS / 'runs/resume_interrupted/checkpoint.pt', map_location='cpu', weights_only=False)
    numeric_keys = ['train_loss','train_accuracy','validation_loss','validation_accuracy','lr','next_lr']
    differences = {k:max(abs(x[k]-y[k]) for x,y in zip(ref['history'],resumed['history'])) for k in numeric_keys}
    resume = dict(interrupted_after_epoch=3, resumed_epochs=[4,5,6], final_epoch=resumed['epoch'],
                  max_parameter_difference=compare_states(ref['model'],resumed['model']),
                  curve_max_differences=differences,
                  exact_match=all(v==0 for v in differences.values()) and compare_states(ref['model'],resumed['model'])==0)
    write_json(RESULTS / 'resume_comparison.json', resume)
    all_history = [dict(name=r['name'], study=r['study'], **h) for r in summaries for h in history(r['name'])]
    write_csv(RESULTS / 'all_epoch_metrics.csv', all_history)

    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for key,label in [('cold_epoch_seconds','First epoch, includes worker startup'),
                      ('steady_epoch_seconds','Median of epochs 2-3')]:
        axes[0].plot([w['num_workers'] for w in workers],[w[key] for w in workers],'o-',label=label)
    axes[0].set(xlabel='num_workers',ylabel='Training epoch (s)',xticks=config['studies']['workers'])
    axes[0].legend(fontsize=8)
    axes[1].plot([w['num_workers'] for w in workers],[w['peak_tree_rss_mib'] for w in workers],'o-',color='#b45d30')
    axes[1].set(xlabel='num_workers',ylabel='Sampled peak process-tree RSS (MiB)',xticks=config['studies']['workers'])
    savefig('worker_study.png')
    fig, axes = plt.subplots(1,3,figsize=(12,3.8))
    for ax,key,label in zip(axes,['samples_per_second','peak_rss_mib','test_accuracy'],
                            ['Training samples/s','Peak process working set (MiB)','Test accuracy after 5 epochs']):
        ax.plot([b['batch_size'] for b in batches],[b[key] for b in batches],'o-')
        ax.set(xlabel='Batch size',ylabel=label)
        ax.set_xscale('log',base=2)
        ax.set_xticks(config['studies']['batch_sizes'],[str(x) for x in config['studies']['batch_sizes']])
    savefig('batch_study.png')
    fig, axes = plt.subplots(1,2,figsize=(11,4.3))
    for r in [r for r in summaries if r['study']=='optimizer']:
        h = history(r['name'])
        axes[0].plot([e['epoch'] for e in h],[e['validation_accuracy'] for e in h],label=r['name'])
        axes[1].plot([e['epoch'] for e in h],[e['lr'] for e in h],label=r['name'])
    axes[0].set(xlabel='Epoch',ylabel='Validation accuracy')
    axes[1].set(xlabel='Epoch',ylabel='Learning rate used (log scale)',yscale='log')
    axes[0].legend(fontsize=7)
    savefig('optimizer_schedules.png')
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for name,label in [('effective256','Physical batch 256'),('micro64_accum4','Batch 64, accumulation 4')]:
        h = history(name)
        axes[0].plot([e['epoch'] for e in h],[e['train_loss'] for e in h],'o-',label=label)
        axes[1].plot([e['epoch'] for e in h],[e['validation_accuracy'] for e in h],'o-',label=label)
    axes[0].set(xlabel='Epoch',ylabel='Training cross-entropy')
    axes[1].set(xlabel='Epoch',ylabel='Validation accuracy')
    axes[0].legend(fontsize=8)
    savefig('accumulation_curves.png')
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for name,style,label in [('resume_reference','o-','Uninterrupted'),('resume_interrupted','x--','Resumed in new process')]:
        h = history(name)
        for ax,key in zip(axes,['train_loss','validation_accuracy']):
            ax.plot([e['epoch'] for e in h],[e[key] for e in h],style,label=label)
            ax.axvline(3.5,color='grey',ls=':',lw=1)
    axes[0].set(xlabel='Epoch (restart between 3 and 4)',ylabel='Training cross-entropy')
    axes[1].set(xlabel='Epoch',ylabel='Validation accuracy')
    axes[0].legend(fontsize=8)
    savefig('resume_curves.png')
    with (RESULTS / 'profile_breakdown.csv').open() as f:
        profile=list(csv.DictReader(f))
    plt.figure(figsize=(7,4))
    plt.barh([p['phase'] for p in profile],[float(p['share_of_four_phases_pct']) for p in profile],color='#287883')
    plt.xlabel('Share of four non-overlapping CPU stage durations (%)')
    savefig('profile_breakdown.png')

    relocate_store(RESULTS / 'mlruns')
    mlflow.set_tracking_uri((RESULTS / 'mlruns').resolve().as_uri())
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config['tracking']['experiment'])
    active_ids = {r['mlflow_run_id'] for r in summaries}
    runs = [r for r in client.search_runs([experiment.experiment_id],max_results=1000) if r.info.run_id in active_ids]
    for run in runs:
        for source in [RESULTS / 'data_sha256.json', RESULTS / 'environment.json',
                       LAB / 'requirements-lock.txt', LAB / 'reference_record.json',
                       LAB / 'references/lab03_summary.csv', LAB / 'references/lab03_split_indices.npz']:
            client.log_artifact(run.info.run_id, str(source), 'provenance')
    write_json(RESULTS / 'mlflow_export.json', [dict(run_id=r.info.run_id,name=r.data.tags['mlflow.runName'],
        status=r.info.status,params=r.data.params,metrics=r.data.metrics,tags=r.data.tags,
        artifacts=[a.path for a in client.list_artifacts(r.info.run_id)]) for r in runs])
    rows=[]
    for r in runs:
        rows.append('<tr><td>'+html.escape(r.data.tags['mlflow.runName'])+'</td><td>'+r.info.run_id+
                    '</td><td>'+r.info.status+'</td><td>'+f"{r.data.metrics.get('validation_accuracy',0):.2%}"+'</td></tr>')
    page='''<!doctype html><html><meta charset="utf-8"><title>Lab 6 experiments</title>
    <style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#173f57}td,th{padding:9px;border-bottom:1px solid #ccd}img{max-width:100%}a{color:#287883}</style>
    <h1>Lab 6: measured experiments</h1><p>G M Robi Farhan Rojob | 73997</p>
    <p>This is an offline view of genuine MLflow FileStore runs. Parameters and metrics: <a href="mlflow_export.json">JSON export</a>. Raw records: <a href="mlruns/">mlruns</a>.</p>
    <table><tr><th>Experiment</th><th>MLflow run ID</th><th>Status</th><th>Final validation accuracy</th></tr>'''+''.join(rows)+'</table>'
    for image in ['profile_breakdown.png','worker_study.png','batch_study.png','optimizer_schedules.png','accumulation_curves.png','resume_curves.png']:
        page+=f'<h2>{html.escape(image.removesuffix(".png").replace("_"," ").title())}</h2><img src="{image}">'
    (RESULTS / 'experiment_dashboard.html').write_text(page+'</html>',encoding='utf-8')

    best_workers=min(workers,key=lambda x:x['steady_epoch_seconds'])
    best_opt=max(optimizers,key=lambda x:x['final_validation_accuracy'])
    dominant=max(profile,key=lambda p:float(p['share_of_four_phases_pct']))
    worker_bottleneck = ('At zero workers, data loading is the largest profiled stage. Extra workers reduce input wait, shifting the limit toward computation and IPC.' if dominant['phase']=='data_loading'
                         else f"The largest profiled stage is {dominant['phase']}; data-loading wait is not the largest component at zero workers.")
    report=f'''# Lab 6: Training pipelines and project proposal

**Student:** G M Robi Farhan Rojob | **ID:** 73997 | **Course:** CII5SE039CI, Group 01

## Goal and provenance

Restructure the best Lab 3 architecture into one YAML-driven reproducible pipeline, measure its bottlenecks, and demonstrate accumulation and restart recovery. The unchanged Lab 3 CNN has 20,490 parameters (Conv 1->16; pool; Conv 16->32; pool; Linear 1568->10). It had the highest Lab 3 validation accuracy (89.60%) and test accuracy (89.17%); the source summary and split are in references. This lab reinitializes that architecture with seed 42 for every independent comparison rather than continuing different pretrained states.

## Method

Fashion-MNIST: all 60,000 official training examples split 55,000/5,000 using random_split seed 42, identical to Lab 3; 10,000 official test examples remain separate. ToTensor and Normalize(mean=0.2860,std=0.3530), no augmentation. Raw data are loaded before training timers; conversion, normalization, collation and iterator waits are inside data-loading time. All runs use float32 CPU, four intra-operation threads, one inter-operation thread and deterministic algorithms. Python, NumPy, Torch, sampler and workers are seeded. All study values and budgets come from config.yaml; effective overrides are logged. Every independent experiment runs sequentially in a fresh Python process, avoiding competing CPU benchmarks and making process memory more comparable. No subset or early stopping is used.

Adam starts at 0.001 with default betas; SGD uses lr=0.01 and momentum=0.9; weight decay is zero. These fixed learning rates are declared choices, not separately tuned optima. StepLR halves lr every three epochs. CosineAnnealingLR has T_max=10 and eta_min=0. ReduceLROnPlateau monitors validation loss (factor 0.5, patience 1, relative threshold 0.001). Schedulers step after validation; the logged lr is the rate actually used in that epoch, and next_lr is distinguished. Six optimizer/scheduler combinations each receive exactly 10 epochs. Only batch-size runs use the official test set, once at the end of each required five-epoch run; scheduler comparisons and selection use validation only.

Throughput is training samples divided by measured training-loop wall time, including iterator startup, transformations, forward/loss, backward, updates and batch scoring. Validation, checkpoint serialization, MLflow writes and process startup are excluded; this is training throughput, not job completion throughput. Profiling overhead is excluded from all study results. Jobs execute in the explicit plan order, without randomized repeated timing blocks; background load and thermal changes may affect rankings. Memory is CPU process memory, not CUDA allocation or model-weight size: batch studies report Windows' native process peak working set in fresh processes; sampled main RSS, baseline/increment and process-tree RSS are also retained. RSS includes interpreter, libraries, dataset, tensors and allocator caching. Sampling is every 20 ms and can miss brief peaks; process-tree sums may double-count shared pages. Native peak includes setup and final evaluation. CPU/GPU figures must not be interchanged.

## 1. YAML, checkpoints and MLflow

main.py derives the full study from one config object and launches src/pipeline.py. Every run logs YAML/effective parameters, per-epoch loss, accuracy, lr, throughput, timing, checkpoints, source and evidence artifacts into genuine local MLflow records. There are {len(runs)} current completed runs; the interrupted run resumes under the same MLflow ID. The offline experiment_dashboard.html and mlflow_export.json expose the records without a server. mlflow-skinny 2.14.1 provides the actual MLflow tracking client/FileStore APIs; the web UI/server is not installed. The full course requirements are preserved elsewhere; its unrelated mlflow/pyarrow conflict is avoided by this explicit tracking-only subset. requirements-lock.txt records actual installed versions. Each epoch checkpoint stores model, optimizer, scheduler, Python/NumPy/Torch RNG, loader generator, epoch, history, effective config and MLflow ID.

## 2. Full-epoch PyTorch profile

One complete epoch (55,000 samples, 430 batches) is captured by torch.profiler on CPU. Non-overlapping record_function ranges enclose next(iterator), forward plus cross-entropy, backward, and gradient normalization/optimizer step/zeroing. The inclusive CPU duration of each labelled parent range includes its child operators; summing all nested operator totals would double-count. The four-stage shares below sum to 100%; the separate epoch-wall shares leave Python scoring/loop overhead outside those four stages. This is elapsed time of CPU ranges, not aggregate work across all cores. Iterator construction is charged to the first data-loading range.

{table(profile,['phase','seconds','share_of_four_phases_pct','share_of_epoch_wall_pct'],{k:lambda x:f'{float(x):.3f}' for k in ['seconds','share_of_four_phases_pct','share_of_epoch_wall_pct']})}

The unassigned epoch-wall residual is {by_name['profile']['profile_other_wall_seconds']:.3f} seconds. Full operator table: profile_operators.txt; compressed Chrome trace: profile_trace.json.gz (decompress to inspect in a compatible trace viewer).

![Profile breakdown](results/profile_breakdown.png)

## 3. DataLoader worker study

Each worker count trains three complete epochs at batch 128. The first includes Windows spawn/startup and warm-up; the median of epochs 2-3 is the steady measurement with persistent workers. A median of two values is their arithmetic midpoint. No disk image reads occur per batch: Fashion-MNIST raw arrays are already in memory. Worker count trades parallel per-sample processing against IPC, memory and CPU contention; eight workers do not mean eight times the throughput. The first-epoch shuffle is identical across worker counts. In later epochs, zero-worker and persistent-worker loaders consume their shared seed generator differently, so their sample orders and trained weights can differ. Every epoch still processes exactly the same 55,000 examples through a fixed-cost dense CNN; this worker study compares time and memory, not accuracy. Each individual setting is reproducible.

{table(workers,['num_workers','cold_epoch_seconds','steady_epoch_seconds','steady_samples_per_second','steady_data_wait_seconds','peak_tree_rss_mib'],{k:lambda x:f'{x:.2f}' for k in workers[0] if k!='num_workers'})}

Best observed steady epoch uses {best_workers['num_workers']} workers. {worker_bottleneck} Compare the stage timings and worker curve together: more workers can remove input wait yet add IPC and memory cost. With only two steady epochs and one seed, close timing rankings are observations on this computer, not proof of a universal optimum.

![Worker study](results/worker_study.png)

## 4. Batch size: five full epochs each

Worker count stays zero, Adam/no schedule, same starting state and epoch shuffle sequence. Batch size changes update count as well as memory and arithmetic utilization; an equal epoch budget does not mean an equal optimizer-update budget. Final partial batches are retained.

{table(batches,['batch_size','epochs','samples_per_second','peak_rss_mib','validation_accuracy','test_accuracy'],{'samples_per_second':lambda x:f'{x:.1f}','peak_rss_mib':lambda x:f'{x:.1f}','validation_accuracy':lambda x:f'{x:.2%}','test_accuracy':lambda x:f'{x:.2%}'})}

Each test accuracy has denominator 10,000; raw correct counts and memory increments are in batch_study.csv. Accuracy and memory need not vary monotonically: optimization paths, convolution workspaces and allocator reuse matter.

![Batch-size study](results/batch_study.png)

## 5. Optimizer and learning-rate schedule: equal ten-epoch budgets

{table(optimizers,['optimizer','scheduler','epochs','final_validation_loss','final_validation_accuracy','initial_lr','final_used_lr'],{'final_validation_loss':lambda x:f'{x:.4f}','final_validation_accuracy':lambda x:f'{x:.2%}','initial_lr':lambda x:f'{x:.6f}','final_used_lr':lambda x:f'{x:.6f}'})}

Highest observed final validation accuracy: {best_opt['optimizer']} + {best_opt['scheduler']} ({best_opt['final_validation_accuracy']:.2%}). This comparison includes the declared starting learning rates and one seed, so it does not establish an optimizer's inherent superiority. A plateau scheduler is conditional: it may keep the same rate if validation loss continues improving. See per-epoch lr and next_lr records rather than assuming every schedule changes lr every epoch.

![Optimizer and schedule curves](results/optimizer_schedules.png)

## 6. Gradient accumulation

Compare physical batch 256 to micro-batch 64 accumulated four times, using SGD+momentum, the same sample order, initialization and three-epoch budget. Sum cross-entropy gradients over a group, divide by the actual number of examples, then step once. The last effective group contains 216 examples (not 256), so blindly dividing every mean micro-loss by four would be wrong. Both paths make {accumulation['optimizer_steps_per_epoch']} updates per epoch. There are no BatchNorm/dropout layers whose behavior would otherwise differ with micro-batch size.

Maximum training-loss difference: {accumulation['max_train_loss_difference']:.3e}; maximum validation-accuracy difference: {accumulation['max_validation_accuracy_difference']:.3e}; maximum final parameter difference: {accumulation['max_parameter_difference']:.3e}. Similar-behavior check (loss difference <0.02 and accuracy difference <=1 percentage point): **{accumulation['similar_behavior']}**. Float32 reduction order can create small differences; an independent float64 linear-model check includes an uneven final group and matches within 1e-14.

![Accumulation comparison](results/accumulation_curves.png)

## 7. Real process interruption after epoch three

An uninterrupted six-epoch Adam+StepLR run is the reference. A separate process trains epochs 1-3, writes checkpoint_epoch3.pt and exits. A fresh Python process restores the complete state and trains epochs 4-6 under the same MLflow run ID. The checkpoint is saved after scheduler.step, so epoch four uses the correct next learning rate. Epoch metrics 1-3 are preserved and metrics 4-6 append without re-running earlier epochs.

Maximum final parameter difference: {resume['max_parameter_difference']:.3e}; per-curve maximum differences: {json.dumps(differences)}. Exact state/curve match: **{resume['exact_match']}**. Timing fields are intentionally excluded from equality because execution time varies. The exact recovery demonstration uses zero workers and deterministic transforms; stochastic worker-local state and prefetched queues would require additional handling. This verifies recovery at an epoch boundary; a mid-batch interruption would need sampler position and any accumulated gradients as well.

![Restart comparison](results/resume_curves.png)

## 8. Individual project proposal

project_proposal.pdf is a separate one-page proposal with name and ID. It proposes an offline Fashion-MNIST classifier, a validation-led training plan and INT8 quantization, with p95 CPU latency <=50 ms and an inference artifact <=5,000,000 bytes. These are proposed acceptance criteria, not achieved deployment claims. Instructor approval remains pending; no approval is fabricated.

## Conclusions

1. The largest profiled stage is {dominant['phase']} ({float(dominant['share_of_four_phases_pct']):.1f}% of the four labelled stages). Worker tuning must consider startup, steady time and whole-process memory together.
2. Batch size and learning-rate policy change both efficiency and optimization. Select using validation and deployment constraints; freeze project choices before the final project test evaluation.
3. Correct accumulation needs actual sample weighting, and reproducible resumption needs optimizer, scheduler and RNG/shuffle state as well as model weights. Both are backed by independent checks and full-dataset comparisons here.

## References and scope

- Supplied Lab Manual, Lab 6, printed page 8; copied to references. Its instructions define the assignment and do not authorize publication or imply instructor approval.
- Supplied Reddi textbook, July 31, 2025: section 5.5 pp.215-218; chapter 8 pp.385-468, especially 8.4 pp.408-422, 8.5 pp.422-439 and 8.7 pp.457-460; section 13.3 pp.859-875. Viewer page = printed page +52. See reference_record.json and study_guide.md.
- [MLflow 2.14.1 tracking documentation](https://mlflow.org/docs/2.14.1/tracking.html) and installed tracking/FileStore APIs.
- [PyTorch 2.3 profiler](https://docs.pytorch.org/docs/2.3/profiler.html), [DataLoader](https://docs.pytorch.org/docs/2.3/data.html), and installed 2.3.1 implementations.

AI-assisted preparation with measured evidence. Personal understanding, prescribed reading, attendance, instructor approval and any eventual submission remain student actions. Earlier labs are preserved.
'''
    (LAB / 'report.md').write_text(report,encoding='utf-8')
    print(f'SUMMARY: {len(summaries)} experiment summaries, {len(all_history)} epoch rows; report and figures written.',flush=True)


if __name__ == '__main__':
    summarize()
