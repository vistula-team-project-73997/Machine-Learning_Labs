"""Generate measured figures, report and offline dashboard from saved evidence."""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAB = Path(__file__).resolve().parents[1]
R = LAB/'results'


def md_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def summarize():
    results=json.loads((R/'summary.json').read_text()); by={r['name']:r for r in results}
    teacher=by['teacher']; plain=by['student_plain']; kd=by['student_distilled']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.grid':True,'grid.alpha':.18})
    colors={'global':'#187f95','layerwise':'#ce6935','structured':'#5e4fa2'}
    fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for method in ('global','layerwise'):
        rows=[r for r in results if r['method']==method]
        x=[100*r['actual_weight_sparsity'] for r in rows]
        ax[0].plot(x,[100*r['pre_validation_accuracy'] for r in rows],'--o',color=colors[method],alpha=.55,label=method+' before FT')
        ax[0].plot(x,[100*r['validation_accuracy'] for r in rows],'-o',color=colors[method],label=method+' after 3 epochs')
        ax[1].plot(x,[100*r['test_accuracy'] for r in rows],'-o',color=colors[method],label=method+' after 3 epochs')
    for a in ax:
        a.set(xlabel='Actual zero-weight sparsity (%)',ylabel='Accuracy (%)',xticks=[0,25,50,75,90,95]);a.legend(fontsize=8)
    ax[0].set_title('Validation: recovery with fixed masks');ax[1].set_title('Final test: same sparsity sweep')
    fig.savefig(R/'sparsity_curves.png',dpi=180);plt.close(fig)
    struct=[teacher]+[r for r in results if r['method']=='structured']
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    x=[100*r['target'] for r in struct]
    ax[0].plot(x,[100*r['validation_accuracy'] for r in struct],'-o',label='validation after FT',color=colors['structured'])
    ax[0].plot(x,[100*r['test_accuracy'] for r in struct],'--s',label='test after FT',color='#187f95')
    ax[0].set(xlabel='Requested filters removed per conv layer (%)',ylabel='Accuracy (%)',title='Physical filter removal');ax[0].legend(fontsize=8)
    ax[1].plot(x,[100*(1-r['parameters']/teacher['parameters']) for r in struct],'-o',label='parameters removed')
    ax[1].plot(x,[100*(1-r['dense_mac']/teacher['dense_mac']) for r in struct],'-s',label='dense MAC removed')
    ax[1].set(xlabel='Requested filters removed per conv layer (%)',ylabel='Reduction from baseline (%)',title='Whole filters change tensor shapes');ax[1].legend(fontsize=8)
    fig.savefig(R/'structured_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in ('global','layerwise'):
        rows=[r for r in results if r['method']==method];x=[100*r['target'] for r in rows]
        ax[0].plot(x,[r['dense_npz_bytes']/2**20 for r in rows],'--',color=colors[method],label=method+' dense')
        ax[0].plot(x,[r['csr_npz_bytes']/2**20 for r in rows],'-o',color=colors[method],label=method+' CSR')
        y=np.array([r['latency_b1_p50_ms'] for r in rows]);lo=np.array([r['latency_b1_min_block_ms'] for r in rows]);hi=np.array([r['latency_b1_max_block_ms'] for r in rows])
        ax[1].plot(x,y,'-o',label=method,color=colors[method]);ax[1].fill_between(x,lo,hi,color=colors[method],alpha=.12)
    ax[0].set(xlabel='Requested zero-weight sparsity (%)',ylabel='File size (MiB)',title='Same uncompressed NPZ container');ax[0].legend(fontsize=8)
    ax[1].axhline(teacher['latency_b1_p50_ms'],ls='--',color='black',label='baseline')
    ax[1].set(xlabel='Requested zero-weight sparsity (%)',ylabel='CPU batch-one median (ms)',title='Dense kernels still process dense tensors');ax[1].legend(fontsize=8)
    fig.savefig(R/'storage_latency.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,label,color in [('student_plain','plain student','#ce6935'),('student_distilled','distilled student','#187f95')]:
        epochs=pd.read_csv(R/'runs'/name/'epochs.csv')
        ax[0].plot(epochs.epoch,epochs.validation_accuracy*100,'-o',label=label,color=color)
        ax[1].plot(epochs.epoch,epochs.cross_entropy,'-o',label=label,color=color)
    ax[0].axhline(teacher['validation_accuracy']*100,ls='--',color='black',label='teacher final')
    ax[0].set(xlabel='Epoch',ylabel='Validation accuracy (%)',title='Equal initialization, order and training budget')
    ax[1].set(xlabel='Epoch',ylabel='Hard-label training CE',title='Compare CE; training objectives differ')
    for a in ax:a.legend(fontsize=8)
    fig.savefig(R/'distillation_curves.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    for method,marker in [('baseline','*'),('global','o'),('layerwise','s'),('structured','D'),('plain','^'),('distilled','v')]:
        rows=[r for r in results if r['method']==method]
        ax.scatter([r['latency_b1_p50_ms'] for r in rows],[r['test_accuracy']*100 for r in rows],label=method,marker=marker,s=65)
    ax.set(xlabel='CPU batch-one median forward latency (ms)',ylabel='Test accuracy (%)',title='Accuracy and measured latency: all 18 models');ax.legend(fontsize=8)
    fig.savefig(R/'accuracy_latency.png',dpi=180);plt.close(fig)
    comparison=md_table(['Model','Test %','Parameters','Zero weights %','Dense / CSR MiB','MAC M','b1 p50 / p95 ms'],[
        [r['name'],f"{r['test_accuracy']*100:.2f}",f"{r['parameters']:,}",f"{r['actual_weight_sparsity']*100:.2f}",
         f"{r['dense_npz_bytes']/2**20:.3f} / {r['csr_npz_bytes']/2**20:.3f}",f"{r['dense_mac']/1e6:.3f}",
         f"{r['latency_b1_p50_ms']:.3f} / {r['latency_b1_p95_ms']:.3f}"] for r in results])
    efficiency_rows=[]
    for r in results:
        unique=55000
        extra=r['epochs']*unique
        if r['method'] in ('global','layerwise','structured'):data_text=f'55k labels; +{extra:,} FT exposures'
        elif r['method']=='distilled':data_text=f'55k labels/logits; {extra:,} exposures'
        else:data_text=f'55k labels; {extra:,} exposures'
        efficiency_rows.append([r['name'],data_text,
            f"{r['parameters']:,}; {r['dense_npz_bytes']/2**20:.3f}/{r['csr_npz_bytes']/2**20:.3f} MiB",
            f"{r['dense_mac']/1e6:.3f} M; {r['latency_b1_p50_ms']:.3f} ms; {r['train_seconds']:.1f} s",
            f"{r['estimated_inference_mJ_at_35W']:.2f} mJ; {r['estimated_training_Wh_at_35W']:.3f} Wh"])
    pillars=md_table(['Model','Data: labels and image exposures','Model: params; dense/CSR','Compute: MAC; b1; training loop','Energy: estimated at 35 W'],efficiency_rows)
    students=md_table(['Model','Val %','Test %','Parameters','Dense MiB','b1 p50 ms','Train loop s'],[
        [r['name'],f"{r['validation_accuracy']*100:.2f}",f"{r['test_accuracy']*100:.2f}",f"{r['parameters']:,}",
         f"{r['dense_npz_bytes']/2**20:.3f}",f"{r['latency_b1_p50_ms']:.3f}",f"{r['train_seconds']:.1f}"] for r in (teacher,plain,kd)])
    global90=by['global_0.9'];structured50=by['structured_0.5'];delta=100*(kd['test_accuracy']-plain['test_accuracy'])
    cache=json.loads((R/'teacher_cache.json').read_text())
    refs=json.loads((LAB/'reference_record.json').read_text()) if (LAB/'reference_record.json').exists() else {}
    hardware=json.loads((R/'hardware.json').read_text()) if (R/'hardware.json').exists() else {}
    environment=json.loads((R/'environment.json').read_text())
    cpu_name=(hardware['cpu_model'] if hardware.get('cpu_identifier')==environment['processor']
              else environment['processor'])
    body=f'''# Lab 7: Compression - pruning and distillation

**Student:** G M Robi Farhan Rojob | **ID:** 73997 | **Course:** CII5SE039CI, Group 01

## Goal and evidence

Compare global, layer-wise and physical structured pruning with a compact student, trained with and without a teacher. These are full-data measurements from this CPU, not anticipated results. Eighteen models complete 75 epochs (10 baseline + 36 unstructured fine-tuning + 9 structured fine-tuning + 20 student). The manual's bracketed task marks sum to 115; this report retains all five tasks without rescaling them.

## Method and reproducibility

Fashion-MNIST has 60,000 official training and 10,000 test images. The seed-42 permutation splits training into 55,000/5,000; split_indices.npz records exact indices, matching the Lab 3/6 random_split convention. Pixels are divided by 255 and normalized with mean 0.2860/std 0.3530. Normalized FP32 tensors are cached once in RAM; this avoids per-example conversion overhead. No augmentation, subset or extra labelled data is used. This setup differs from Lab 6's timed transforms, so training speeds across labs should not be directly ranked.

Hardware: **{cpu_name}** (Windows, exact platform/build in environment.json). CPU FP32, four intra-op threads, one inter-op thread, deterministic algorithms, seed 42. Every model processes 55,000 images per epoch in 430 batches (batch 128, final batch 88). Each independent training call resets its sample-order generator to 43. Adam uses lr 0.001 for fresh teacher/student training and 0.0001 for fine-tuning; default betas, zero weight decay, no scheduler or early stopping. The final fixed-budget epoch supplies each model. All hyperparameters were fixed before observing these test results; validation is logged before/after pruning and every epoch. The test set measures the required comparisons after each final model; it never drives optimization or mask selection. One seed is used, so small accuracy gaps do not establish significance.

Code/config fingerprints, versions, architecture, initial-parameter hashes, checkpoint hashes, epoch logs and exact split indices are recorded. Running main.py continues complete fingerprint-matching experiments; an interrupted experiment restarts from its beginning. Use --rerun for deliberate retraining. Repeated invocation refreshes final timing; latency is expected to vary.

## 1. Oversized baseline [15]

Teacher: Conv(1,16,3,pad1)-ReLU-MaxPool2; Conv(16,32,3,pad1)-ReLU-MaxPool2; Flatten; Linear(1568,640)-ReLU-Linear(640,10). No BatchNorm or dropout. The parameter count is 160 + 4,640 + 1,004,160 + 6,410 = **{teacher['parameters']:,}**. The large hidden layer deliberately puts most parameters in the dense head. Ten epochs produce **{teacher['test_accuracy']:.2%} test accuracy**.

Inference-only model.pt stores architecture and state, without optimizer or pruning reparameterizations. Dense/CSR NPZ files below are a separate like-for-like storage comparison.

## 2. Controlled pruning and recovery [35]

Global pruning sorts absolute weights across all convolutional/linear weight tensors. Layer-wise pruning ranks within each tensor. Biases remain trainable and are excluded from the pruning denominator. Targets are exactly 0, 0.25, 0.5, 0.75, 0.9 and 0.95. We remove round(target * eligible weights), with stable flattened-index tie-breaking. Layer-wise rounding occurs independently per tensor; actual sparsity is reported. Each point starts from the same trained teacher, receives a new optimizer, and fine-tunes three full epochs at 0.0001. This is a one-shot independent sweep, not iterative pruning or cumulative training.

Masked gradients and post-update projection both preserve fixed zeros; assertions check for regrowth after every epoch. Saved masks and per-layer sparsities support verification. The 0% cases are three-epoch dense fine-tuning controls; they have three more epochs than the original baseline. Validation curves distinguish immediate pruning damage from recovery. Final test curves compare all frozen outcomes.

![Unstructured sparsity curves](results/sparsity_curves.png)

Structured pruning independently ranks each original convolution's output filters by L2 norm and removes round(target * filters) at 10%, 25% and 50%. Retained filters stay in original index order. Conv1 outputs/Conv2 inputs, Conv2 outputs and corresponding fc1 input columns are physically sliced, together with convolution biases. Integer rounding means requested filter removal can differ slightly from actual per-layer removal. The classifier hidden width stays 640. The compact model is checked against a dense network with removed filter outputs zeroed before fine-tuning; all differences must be below 2e-4. It then receives the same three-epoch fine-tuning recipe. Its remaining tensors are dense, so its zero-weight fraction is near zero: removed filters are absent, not stored zeros. Filter removal, parameter reduction and zero-weight sparsity are separate quantities.

![Structured removal and reduction](results/structured_curves.png)

## 3. Storage, hardware latency and compute [15]

{comparison}

Dense NPZ stores all FP32 coefficients, including zeros. Custom CSR flattens each weight tensor to (output channels, remaining dimensions), stores nonzero FP32 values, int32 column indices/row pointers and int32 shape metadata; biases remain dense. Both files use the same **uncompressed** NPZ container. Every sparse file is decoded and compared bit-for-bit with its dense weights. Architecture is in model.pt/summary. File sizes include container headers; model.pt has its own measured byte count in comparison.csv. MiB = 1,048,576 bytes. Stored masks are training evidence and excluded from inference-file sizes.

For an m-by-n FP32 matrix, dense payload is 4mn bytes, whereas this CSR payload is approximately 8*nnz + 4*(m+1) bytes plus shape/container metadata. Index overhead means CSR can exceed dense storage at low sparsity; crossing roughly 50% sparsity is only a payload approximation. At 90% global sparsity, dense storage is {global90['dense_npz_bytes']/2**20:.3f} MiB and CSR is {global90['csr_npz_bytes']/2**20:.3f} MiB. Restoring CSR to dense for inference does not preserve its RAM savings; no sparse execution speed is claimed.

Benchmark: CPU eager PyTorch, eval plus inference_mode, pre-created contiguous normalized inputs, batches 1 and 128, same four threads. Five blocks independently shuffle model order with seed 142. Before each model/block/batch, 30 warm-up calls run. Each block records 100 individual batch-one calls or 20 batch-128 calls using perf_counter_ns. Reported p50/p95 pool 500/100 calls per model; latency_blocks.csv retains block medians and order, latency_raw.csv every observation. Timings include Python dispatch and tensor allocation inside forward, and exclude preprocessing, data transfer, disk I/O, model loading and argmax. Inputs are reused and models remain resident; this is warm forward latency, not a cold-start or production request SLA. Synchronous CPU calls need no CUDA synchronization. Background scheduling, cache and thermal effects can alter close rankings; p95 is descriptive, not a confidence interval. Shaded bands show minimum/maximum block medians, not statistical confidence.

![Storage and measured latency](results/storage_latency.png)

Unstructured pruning retains identical tensor shapes and dense kernels: the baseline and global-90% model both require {teacher['dense_mac']:,} nominal dense MAC per image. Their observed batch-one medians are {teacher['latency_b1_p50_ms']:.3f} and {global90['latency_b1_p50_ms']:.3f} ms. Any difference here cannot be credited to skipping zero multiplies because this execution path does not skip them. Real sparse acceleration needs compatible kernels, a suitable sparsity pattern and enough work to repay index/gather overhead. Physically compact structured-50% uses {structured50['dense_mac']:,} MAC and measures {structured50['latency_b1_p50_ms']:.3f} ms, versus the baseline {teacher['latency_b1_p50_ms']:.3f} ms. Reduced MAC counts still need hardware measurement to establish a benefit.

MAC counts cover convolution and linear weight multiplies per image (biases, activations and pooling excluded); FLOPs use 2 per multiply-add. potential_nonzero_mac is an idealized zero-skipping count, not executed CPU work. Parameter sparsity can differ from MAC sparsity because a convolution weight is reused at many spatial positions. Batch-128 throughput, p50/p95 and all file sizes are in comparison.csv.

## 4. Plain and distilled student [30]

Student: conv channels 8 and 16, flatten 784, hidden width 128, ten classes. Parameter count = 80 + 1,168 + 100,480 + 1,290 = **{plain['parameters']:,}**. Both students start from bit-identical weights, use identical epoch orders/Adam settings, and train for ten epochs. Plain loss is hard-label cross entropy. The teacher is frozen in eval mode, and its logits are cached **only for the 55,000 training images**, taking {cache['seconds']:.2f} seconds and {cache['bytes']:,} bytes. Cache preparation is an extra distillation cost, excluded from student epoch timing. Teacher training and cache generation must be included when considering lifecycle cost; they are not needed during student inference.

The manual explicitly writes KL(p_student || p_teacher), which is the reverse of the common teacher-target KL(p_teacher || p_student). This implementation follows the manual literally:

`p_s = softmax(student_logits / 4); log_p_t = log_softmax(teacher_logits / 4)`

`KL = mean_over_batch(sum_over_classes(p_s * (log_p_s - log_p_t)))`

`loss = 0.5 * CE(student_logits, labels) + 0.5 * 4^2 * KL`

Teacher logits are detached; the student probability term remains differentiable, including its entropy. Class contributions are summed and examples averaged. PyTorch's usual kl_div(log_student, teacher_probs, reduction='batchmean') would compute the other direction. This choice is documented for instructor discussion instead of silently changing the supplied equation. A numerical gradient check tests the literal direction. Temperature 4 softens relative class scores; T^2 compensates the temperature-dependent gradient scale. At inference, the student uses its ordinary logits without the teacher or temperature adjustment.

{students}

The distilled-minus-plain test difference is **{delta:+.2f} percentage points** ({kd['test_correct']-plain['test_correct']:+d} correctly classified images out of 10,000). This is the observed result for the stated manual-direction loss and one fixed seed. Distillation is not guaranteed to improve accuracy. Both students have the same deployment shape, MAC count and storage; training differences cannot intrinsically make one execute a different number of dense operations.

![Student training comparison](results/distillation_curves.png)

## 5. Efficiency pillars and conclusion [20]

The table uses the manual's data/model/compute/energy organization. In this textbook edition, section 9.3 explicitly names algorithmic, compute and data efficiency; energy and sustainability also enter the broader system trade-offs. Here model is the compression/representation view, and energy is reported separately as required by the laboratory.

{pillars}

**Energy is not measured.** The fixed-power scenarios 15/35/65 W are illustrative assumptions, not CPU TDP readings, telemetry or confidence bounds. Inference estimate E_mJ = P_W * latency_ms. Training estimate E_Wh = P_W * training-plus-validation wall_seconds / 3600. The table shows 35 W; all three scenarios are saved. Cache generation at 35 W would add {35*cache['seconds']/3600:.4f} Wh and teacher preparation must also be counted. Data preparation, process startup, final test, serialization and failed/verification work are excluded from these estimates; whole-job energy is therefore not established. Actual power can differ by model, load and time, so latency-scaled estimates cannot prove energy savings or emissions reductions. A power meter or calibrated platform energy counter would be needed.

1. Global pruning trades zero-weight storage against accuracy. At 90%, it reaches {global90['test_accuracy']:.2%} test accuracy and a {global90['dense_npz_bytes']/global90['csr_npz_bytes']:.2f}x dense-to-CSR file-size ratio, while leaving dense execution shapes unchanged.
2. Physical structured pruning and the compact student reduce dense tensor dimensions and nominal work. Structured-50% retains {structured50['test_accuracy']:.2%} test accuracy with {100*(1-structured50['parameters']/teacher['parameters']):.1f}% fewer parameters. Measure latency on the target deployment device before selecting a model; storage, MAC and latency rankings need not match.
3. The student uses {teacher['parameters']/plain['parameters']:.2f}x fewer parameters than the teacher. Distillation changes its test accuracy by {delta:+.2f} points in this run and incurs teacher preparation cost. These measurements support a device-specific accuracy/size/latency decision, with energy still requiring direct measurement.

![All models: accuracy and latency](results/accuracy_latency.png)

## Sources and review

Lab Manual: Labs 07, printed page 9, preserved in references/. The supplied Reddi textbook edition is July 31, 2025; Chapter 9 is printed pp. 479-540 (viewer 531-592), section 10.3 pp. 547-550 (viewer 599-602), and section 10.4 pp. 550-584 (viewer 602-636). Boundaries include following headings; pruning begins p.551, distillation p.565. This edition calls 10.4 Model Representation Optimization. Source hashes and reading map are in reference_record.json. The study guide includes official PyTorch references and oral-practice questions.

This is AI-assisted preparation with measured evidence. Review the choices and conclusions so you can explain them during class. No submission or instructor approval is claimed.
'''
    (LAB/'report.md').write_text(body,encoding='utf-8')
    # Self-contained, offline view with sorting and filtering.
    head=['Model','Test %','Parameters','Dense MiB','CSR MiB','b1 p50 ms','b1 p95 ms','MAC M','Est. mJ at 35 W']
    rows=[[r['name'],f"{r['test_accuracy']*100:.2f}",r['parameters'],f"{r['dense_npz_bytes']/2**20:.3f}",f"{r['csr_npz_bytes']/2**20:.3f}",
           f"{r['latency_b1_p50_ms']:.3f}",f"{r['latency_b1_p95_ms']:.3f}",f"{r['dense_mac']/1e6:.3f}",f"{r['estimated_inference_mJ_at_35W']:.2f}"] for r in results]
    table='<table><thead><tr>'+''.join(f'<th onclick="sortTable({i})">{html.escape(v)}</th>' for i,v in enumerate(head))+'</tr></thead><tbody id="body">'
    table+=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lab 7 | 73997</title><style>
body{font:16px/1.5 system-ui;margin:36px auto;max-width:1250px;padding:0 24px;background:#f4f8fa;color:#183546}h1{font-size:32px}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{padding:9px;border-bottom:1px solid #d8e2e9;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#dfeef2;cursor:pointer;position:sticky;top:0}input{padding:10px;width:280px;margin:12px 0}img{max-width:100%;background:white;border-radius:8px;margin:12px 0}a{color:#166c84}.note{padding:16px;background:#fff3db;border-left:4px solid #b47526}.scroll{overflow:auto}</style>
<h1>Lab 7: Compression, measured</h1><p>G M Robi Farhan Rojob | 73997 | 18 models, 75 full-data epochs | CPU FP32, four threads</p>
<p class="note">Energy values are estimates under an assumed constant 35 W, not measurements. Unstructured models run dense kernels. Small timing differences can be noise.</p>
<p><a href="../report.md">Measured report</a> / <a href="comparison.csv">Full CSV</a> / <a href="latency_raw.csv">Raw latency</a> / <a href="../lab7%20explanation.pdf">Explanation PDF</a></p>
<label>Filter models <input id="filter" oninput="filterRows()" placeholder="e.g. student or global"></label><p>Click column headings to sort.</p><div class="scroll">'''+table+'''</div>
<h2>Pruning and fine-tuning</h2><img src="sparsity_curves.png" alt="Accuracy against sparsity"><img src="structured_curves.png" alt="Structured pruning curves">
<h2>Storage and CPU latency</h2><img src="storage_latency.png" alt="Storage and latency"><h2>Plain versus distilled</h2><img src="distillation_curves.png" alt="Student training"><img src="accuracy_latency.png" alt="Accuracy latency scatter">
<script>let direction={};function sortTable(i){let b=document.getElementById('body'),rs=Array.from(b.rows);direction[i]=-(direction[i]||-1);rs.sort((a,b)=>{let x=a.cells[i].textContent,y=b.cells[i].textContent;return direction[i]*(i?Number(x)-Number(y):x.localeCompare(y))});rs.forEach(r=>b.appendChild(r))}function filterRows(){let s=document.getElementById('filter').value.toLowerCase();Array.from(document.getElementById('body').rows).forEach(r=>r.hidden=!r.cells[0].textContent.toLowerCase().includes(s))}</script></html>'''
    (R/'experiment_dashboard.html').write_text(page,encoding='utf-8')
    print('Generated report, five measured figures and offline dashboard.',flush=True)


if __name__=='__main__':summarize()
