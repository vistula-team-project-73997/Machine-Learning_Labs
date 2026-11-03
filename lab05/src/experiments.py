"""Reproducible Lab 5 experiments. All timings are measured on this machine."""
import csv
import gc
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from autodiff import Value

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED, EPOCHS, LR = 42, 150, 0.5


def write_json(name, data):
    (RESULTS / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initial_weights():
    rng = np.random.default_rng(SEED)
    return [rng.normal(0, .6, (8, 2)), np.zeros(8),
            rng.normal(0, .4, (1, 8)), np.zeros(1)]


def scalar_model(arrays):
    flat = [Value(v) for a in arrays for v in a.ravel()]
    w1 = [flat[i*2:i*2+2] for i in range(8)]
    b1, w2, b2 = flat[16:24], flat[24:32], flat[32]
    def forward(row):
        hidden = [(sum(w*x for w, x in zip(ws, row)) + b).relu()
                  for ws, b in zip(w1, b1)]
        return (sum(w*h for w, h in zip(w2, hidden)) + b2).sigmoid()
    return flat, forward


def torch_model(arrays):
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(),
                                torch.nn.Linear(8, 1), torch.nn.Sigmoid()).double()
    with torch.no_grad():
        for p, a in zip(model.parameters(), arrays):
            p.copy_(torch.from_numpy(a))
    return model


def scalar_loss(forward, x, y):
    losses = []
    for row, target in zip(x, y):
        p = forward(row)
        losses.append(-(target * p.log() + (1-target) * (1-p).log()))
    return sum(losses) / len(losses)


def gradient_check():
    rng = np.random.default_rng(SEED)
    abc = rng.uniform(-.9, .9, (3, 8))
    values = [[Value(v) for v in row] for row in abc]
    terms = [((a*b+c)**2).relu() for a, b, c in zip(*values)]
    loss = sum(terms)
    loss.backward()
    t = [torch.tensor(row, dtype=torch.float64, requires_grad=True) for row in abc]
    reference = ((t[0]*t[1]+t[2])**2).relu().sum()
    reference.backward()
    def numeric(arr):
        return np.maximum((arr[0]*arr[1]+arr[2])**2, 0).sum()
    rows = []
    for j, name in enumerate("abc"):
        for k in range(8):
            plus, minus = abc.copy(), abc.copy()
            plus[j, k] += 1e-6
            minus[j, k] -= 1e-6
            numerical = (numeric(plus)-numeric(minus)) / 2e-6
            grad, pt = values[j][k].grad, t[j].grad[k].item()
            rows.append(dict(variable=name, index=k, engine=grad, pytorch=pt,
                             numerical=numerical, torch_abs_error=abs(grad-pt),
                             numerical_abs_error=abs(grad-numerical)))
    write_csv("gradient_check.csv", rows)
    summary = dict(expression="sum_i relu((a_i*b_i+c_i)**2)", dtype="float64",
                   epsilon=1e-6, coordinates=len(rows), threshold=1e-6,
                   engine_loss=loss.data, pytorch_loss=reference.item(),
                   max_torch_error=max(r["torch_abs_error"] for r in rows),
                   max_numerical_error=max(r["numerical_abs_error"] for r in rows))
    assert summary["max_torch_error"] < 1e-6
    assert summary["max_numerical_error"] < 1e-6
    write_json("gradient_summary.json", summary)
    nodes = loss.topological()
    ids = {node: i for i, node in enumerate(nodes)}
    lines = ["digraph scalar_graph {", 'rankdir="LR";']
    for node in nodes:
        lines.append(f'n{ids[node]} [label="{node.op or "leaf"}\\ndata={node.data:.4g}\\ngrad={node.grad:.4g}"];')
        for parent in node.parents:
            lines.append(f"n{ids[parent]} -> n{ids[node]};")
    (RESULTS / "scalar_graph.dot").write_text("\n".join(lines+["}"]), encoding="utf-8")
    return summary


def train_scalar(arrays, x, y, epochs, record=False):
    params, forward = scalar_model(arrays)
    history = []
    start = perf_counter()
    for epoch in range(epochs):
        loss = scalar_loss(forward, x, y)
        loss.backward(retain_graph=False)
        for p in params:
            p.data -= LR * p.grad
        if record:
            history.append(loss.data)
    # Release final graph debris inside timing, like ordinary Python execution.
    del loss
    gc.collect()
    elapsed = perf_counter()-start
    weights = np.array([p.data for p in params])
    return elapsed, weights, history


def train_torch(arrays, x, y, epochs, record=False):
    model = torch_model(arrays)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y[:, None])
    history = []
    start = perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy(model(tx), ty)
        loss.backward()
        optimizer.step()
        if record:
            history.append(loss.item())
    elapsed = perf_counter()-start
    weights = np.concatenate([p.detach().numpy().ravel() for p in model.parameters()])
    return elapsed, weights, history


def arrays_from_flat(flat):
    return [flat[:16].reshape(8, 2), flat[16:24], flat[24:32].reshape(1, 8), flat[32:]]


def training():
    x, y = make_moons(n_samples=160, noise=.15, random_state=SEED)
    xtr, xte, ytr, yte = train_test_split(x, y.astype(np.float64), test_size=.3,
                                        stratify=y, random_state=SEED)
    arrays = initial_weights()
    np.savez(RESULTS / "moons_split.npz", x_train=xtr, x_test=xte, y_train=ytr, y_test=yte)
    # Untimed warm-up uses throwaway models and identical steps.
    train_scalar(arrays, xtr, ytr, 2)
    train_torch(arrays, xtr, ytr, 2)
    rows, histories, saved = [], {}, {}
    for run in range(5):
        order = ["scalar", "pytorch"] if run % 2 == 0 else ["pytorch", "scalar"]
        for name in order:
            print(f"Training {name}: run {run+1}/5", flush=True)
            fn = train_scalar if name == "scalar" else train_torch
            elapsed, flat, history = fn(arrays, xtr, ytr, EPOCHS, record=True)
            model = torch_model(arrays_from_flat(flat)).eval()
            with torch.inference_mode():
                tr = model(torch.from_numpy(xtr)).squeeze(1)
                te = model(torch.from_numpy(xte)).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy(tr, torch.from_numpy(ytr)).item()
                train_acc = ((tr.numpy() >= .5) == ytr).mean()
                test_acc = ((te.numpy() >= .5) == yte).mean()
            rows.append(dict(engine=name, run=run+1, seconds=elapsed, epochs=EPOCHS,
                             final_bce=loss, train_accuracy=train_acc, test_accuracy=test_acc))
            if run == 0:
                histories[name], saved[name] = history, flat
    write_csv("training_runs.csv", rows)
    write_csv("loss_history.csv", [dict(epoch=i+1, scalar=histories["scalar"][i],
                                      pytorch=histories["pytorch"][i]) for i in range(EPOCHS)])
    max_error = float(np.max(np.abs(saved["scalar"]-saved["pytorch"])))
    assert max_error < 1e-9, max_error
    summary = dict(n_samples=160, train_samples=len(ytr), test_samples=len(yte),
                   seed=SEED, epochs=EPOCHS, learning_rate=LR, runs=5,
                   max_final_parameter_error=max_error,
                   scalar_median_seconds=statistics.median(r["seconds"] for r in rows if r["engine"]=="scalar"),
                   pytorch_median_seconds=statistics.median(r["seconds"] for r in rows if r["engine"]=="pytorch"),
                   test_accuracy=rows[0]["test_accuracy"], final_bce=rows[0]["final_bce"])
    write_json("training_summary.json", summary)
    np.savez(RESULTS / "trained_weights.npz", scalar=saved["scalar"], pytorch=saved["pytorch"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(histories["scalar"], label="Scalar AD", linewidth=2)
    ax.plot(histories["pytorch"], "--", label="PyTorch", linewidth=1.6)
    ax.set(xlabel="Epoch (zero indexed)", ylabel="Mean training BCE", title="Same initialization and updates")
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "loss_curves.png", dpi=160); plt.close(fig)
    return summary, torch_model(arrays_from_flat(saved["pytorch"])).eval()


def timed_alternating(functions, repeats, inner, group):
    rows = []
    names = list(functions)
    for name in names:
        for _ in range(20):
            functions[name]()
    for r in range(repeats):
        for name in (names if r % 2 == 0 else list(reversed(names))):
            start = perf_counter()
            for _ in range(inner):
                functions[name]()
            rows.append(dict(group=group, variant=name, repeat=r+1,
                             inner=inner, seconds_per_call=(perf_counter()-start)/inner))
    return rows


def execution(model):
    from torch.fx import symbolic_trace
    (RESULTS / "fx_forward_graph.txt").write_text(str(symbolic_trace(model).graph), encoding="utf-8")
    prediction = model(torch.zeros(2, 2, dtype=torch.float64))
    seen, lines = set(), []
    def walk(fn, depth=0):
        if fn is None or fn in seen:
            return
        seen.add(fn)
        lines.append("  "*depth + type(fn).__name__)
        for nxt, _ in fn.next_functions:
            walk(nxt, depth+1)
    walk(prediction.sum().grad_fn)
    (RESULTS / "autograd_backward_graph.txt").write_text("\n".join(lines), encoding="utf-8")
    generator = torch.Generator().manual_seed(SEED)
    example = torch.randn(32, 2, dtype=torch.float64, generator=generator)
    cold, rows, summaries = [], [], []
    with torch.inference_mode():
        start = perf_counter()
        traced = torch.jit.trace(model, example, check_inputs=[(example*2,)])
        trace_seconds = perf_counter()-start
        (RESULTS / "traced_graph.txt").write_text(str(traced.inlined_graph), encoding="utf-8")
        traced.save(str(RESULTS / "tiny_model_traced.pt"))
        compiled, error, wrapper_seconds = None, None, None
        if hasattr(torch, "compile"):
            try:
                # On this PyTorch build compiling Sequential directly can return
                # without capturing a graph. An explicit user-code frame plus
                # a capture-count check prevents benchmarking that as compiled.
                def forward(x):
                    return model(x)
                from torch._dynamo.utils import counters
                before = counters["stats"]["unique_graphs"]
                start = perf_counter()
                compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                wrapper_seconds = perf_counter()-start
                compiled(example)  # Actual compilation is lazy: record first execution.
                initial_compile_seconds = perf_counter()-start
                if counters["stats"]["unique_graphs"] <= before:
                    raise RuntimeError("torch.compile returned without capturing a graph")
            except Exception as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                (RESULTS / "compile_diagnostic.txt").write_text(diagnostic, encoding="utf-8")
                error = diagnostic.split("\n\n")[0].replace("\n", " ")
                compiled = None
        else:
            error = "torch.compile is not available in this PyTorch version"
        for batch in (1, 32, 256):
            x = torch.randn(batch, 2, dtype=torch.float64, generator=generator)
            functions = {"eager": lambda: model(x), "traced": lambda: traced(x)}
            if compiled is not None:
                functions["compiled"] = lambda: compiled(x)
            first = {}
            for name, fn in functions.items():
                start = perf_counter(); output = fn(); first[name] = perf_counter()-start
                torch.testing.assert_close(output, model(x), rtol=1e-10, atol=1e-12)
            block = timed_alternating(functions, repeats=7, inner=300, group=f"batch_{batch}")
            rows.extend(block)
            medians = {name: statistics.median(r["seconds_per_call"] for r in block if r["variant"]==name)
                       for name in functions}
            summaries.append(dict(batch=batch, median_seconds=medians,
                                  speedup_vs_eager={k: medians["eager"]/v for k, v in medians.items()},
                                  first_call_seconds=first))
        cold = dict(selected="compiled" if compiled is not None else "traced", compile_error=error,
                    trace_creation_seconds=trace_seconds, compile_wrapper_seconds=wrapper_seconds,
                    compile_creation_and_first_call_seconds=initial_compile_seconds if compiled is not None else None,
                    note="First-use observations may reuse on-disk compiler caches; not guaranteed cache-cold.")
    write_csv("execution_timings.csv", rows)
    write_json("execution_summary.json", dict(setup=cold, results=summaries))
    return dict(setup=cold, results=summaries)


def contiguity():
    generator = torch.Generator().manual_seed(SEED)
    backing = torch.randn(1024, 2048, dtype=torch.float64, generator=generator)
    view = backing[:, ::2]
    contiguous = view.contiguous()
    assert not view.is_contiguous() and contiguous.is_contiguous()
    torch.testing.assert_close(view, contiguous, rtol=0, atol=0)
    def operation(x):
        return x.sin().sum()
    with torch.inference_mode():
        torch.testing.assert_close(operation(view), operation(contiguous))
        rows = timed_alternating({"contiguous": lambda: operation(contiguous),
                                  "noncontiguous_view": lambda: operation(view),
                                  "copy_then_operation": lambda: operation(view.contiguous()),
                                  "copy_only": lambda: view.contiguous()},
                                 repeats=7, inner=30, group="sin_sum_1024x1024")
    write_csv("contiguity_timings.csv", rows)
    tensors = []
    for name, t in [("backing", backing), ("view", view), ("contiguous", contiguous)]:
        tensors.append(dict(name=name, shape=list(t.shape), dtype=str(t.dtype),
                            stride=list(t.stride()), contiguous=t.is_contiguous(),
                            storage_offset=t.storage_offset(), bytes_per_element=t.element_size(),
                            shares_backing_storage=t.untyped_storage().data_ptr()==backing.untyped_storage().data_ptr()))
    medians = {name: statistics.median(r["seconds_per_call"] for r in rows if r["variant"]==name)
               for name in ["contiguous", "noncontiguous_view", "copy_then_operation", "copy_only"]}
    summary = dict(operation="sin(x).sum()", tensors=tensors, median_seconds=medians,
                   view_over_contiguous=medians["noncontiguous_view"]/medians["contiguous"])
    write_json("contiguity_summary.json", summary)
    return summary


def report(g, t, e, c):
    setup = e["setup"]
    speed = t["scalar_median_seconds"]/t["pytorch_median_seconds"]
    lines = ["# Lab 5 - Frameworks and automatic differentiation", "", "G M Robi Farhan Rojob | Student number 73997 | Machine Learning, Group 01", "", "This is a runnable preparation example. Review the choices and explanations before using it in class.", "",
             "## Gradient correctness", "",
             f"24 coordinates of the manual's expression were tested in float64. Maximum absolute error: PyTorch **{g['max_torch_error']:.3e}**, central differences **{g['max_numerical_error']:.3e}**. Both are below 1e-6. Epsilon was 1e-6.", "",
             "The scalar engine uses one Value per element and sums the eight output Values. ReLU after a square cannot exercise a negative ReLU input, so independent tests cover negative, zero and positive ReLU inputs, reused nodes, duplicate edges, a deep graph, and all 33 model-parameter gradients.", "",
             "## Tiny model and training speed", "",
             f"2 -> 8 (ReLU) -> 1 (sigmoid), probability-space BCE, full-batch SGD, learning rate {LR}, {EPOCHS} epochs. make_moons: 160 points, noise 0.15, stratified 70/30 split, seed 42. NumPy default_rng(42) initializes first-layer weights from N(0, 0.6**2) and second-layer weights from N(0, 0.4**2); all biases are zero. These hyperparameters and the hidden ReLU are explicit choices where the manual is silent.", "",
             "Both engines start from identical 33 parameters for every run; use identical data, float64 CPU arithmetic, loss reduction, updates and epoch count. Five timed runs alternate engine order, after two untimed warm-up epochs on throwaway models. Data/model/optimizer setup and evaluation are outside timing; forward, backward and SGD updates are inside. Both record per-epoch loss; scalar graph disposal and final garbage collection are included.", "",
             "| Engine | Median training time |", "|---|---:|",
             f"| Scalar AD | {t['scalar_median_seconds']:.6f} s |",
             f"| PyTorch eager | {t['pytorch_median_seconds']:.6f} s |", "",
             f"PyTorch was {speed:.2f}x faster. Final parameter difference: {t['max_final_parameter_error']:.3e}; test accuracy: {t['test_accuracy']:.2%}; final train BCE: {t['final_bce']:.6f}. This comparison includes Python scalar objects versus vectorized native tensor operations; it does not isolate the cost of autodiff alone.", "",
             "![Training curves](results/loss_curves.png)", "", "## Execution modes", "",
             "Inference uses the same trained model, float64 CPU inputs and one PyTorch thread. Each input shape is warmed up separately (20 calls per variant), then measured in seven blocks of 300 calls with alternating order. The table reports median time per call; raw blocks are retained. Every variant's outputs are checked against eager. Training timings above are a separate experiment.", "",
             "| Batch | Eager (us) | Traced (us) | Compiled (us) |", "|---:|---:|---:|---:|"]
    for r in e["results"]:
        m = r["median_seconds"]
        comp = f"{m['compiled']*1e6:.3f}" if "compiled" in m else "unsupported"
        lines.append(f"| {r['batch']} | {m['eager']*1e6:.3f} | {m['traced']*1e6:.3f} | {comp} |")
    lines += ["", f"Selected main comparison: **{setup['selected']}**. Trace creation: {setup['trace_creation_seconds']:.6f} s."]
    if setup["compile_error"]:
        lines += ["", f"Compilation failed; tracing is the manual-approved fallback. Diagnostic: `{setup['compile_error']}`."]
    else:
        lines += ["", f"Compiler wrapper plus initial batch-32 execution: {setup['compile_creation_and_first_call_seconds']:.6f} s. Later first calls for batches 1 and 256 can include shape specialization; they are logged separately and excluded from steady-state timings. On-disk compiler caches may be reused, so this is observed first-use overhead, not a guaranteed clean-cache build time."]
    traced_speedups = [r["speedup_vs_eager"]["traced"] for r in e["results"]]
    lines += ["", f"Tracing was {min(traced_speedups):.2f}x to {max(traced_speedups):.2f}x as fast as eager across the measured batches. The compilation probe uses an explicit forward function and verifies graph capture, because a returning callable alone is insufficient evidence of compilation.", "",
              "### Short note: when compilation helps", "",
              "Compilation can help when repeated work saves enough interpreter/dispatch or kernel cost to repay setup. Stable shapes and many small operations can provide opportunities; a tiny model can instead be dominated by wrapper and guard overhead. For this run, use the measured speedup rather than assuming compilation wins. The break-even estimate is N > setup_time / (eager_time - optimized_time), only when the denominator is positive; changing shapes or graph breaks can add setup again. This is an interpretation of the experiment, not a general performance guarantee.", "",
              "Tracing records the path exercised by example inputs. This model has no data-dependent Python branch. Tracing an if-statement on tensor values would require separate correctness analysis. FX forward, autograd backward and TorchScript graphs are saved separately: they serve different purposes.", "",
              "## Tensor layout", "",
              "The strided view is backing[:, ::2], shape (1024, 1024), float64, stride (2048, 2). Its contiguous copy has identical element values and stride (1024, 1). The view shares the backing storage; the copy owns different storage. View creation and the initial copy are excluded from direct-operation timing; copy-only and copy-plus-operation are separately measured.", "",
              "| Variant | Median sin(x).sum() time (ms) |", "|---|---:|"]
    for name, seconds in c["median_seconds"].items():
        lines.append(f"| {name} | {seconds*1000:.4f} |")
    lines += ["", f"The strided-view/direct-contiguous ratio is {c['view_over_contiguous']:.3f}x. This is a measured layout effect for one operation and one shape. Allocation, cache behavior, stride-aware kernels and internal copies can change the result for other operations. Copying is worthwhile only if later savings repay the copy.", "",
              "## Evidence and limits", "",
              "`results/environment.json` records the actual interpreter, versions, CPU, device, threads and timestamp. CSV files preserve every training run and timing block. No GPU synchronization is needed because all measured tensors are on CPU. Wall-clock results are sensitive to system load; seven blocks are descriptive samples, not confidence intervals. This 160-point dataset is an instructional example, not a broad model-quality evaluation.", "",
              "Manual: `references/Lab Manual_ Labs 05.pdf`, printed page 7. Reading links and a worked chain-rule example are in `study_guide.md`. This preparation does not establish attendance or replace the opening classroom instructions."]
    (ROOT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    write_json("environment.json", dict(timestamp=datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(),
        python=sys.version, executable=sys.executable, torch=torch.__version__, numpy=np.__version__,
        sklearn=sklearn.__version__, matplotlib=matplotlib.__version__, platform=platform.platform(),
        processor=platform.processor(), device="cpu", cuda_available=torch.cuda.is_available(),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), seed=SEED,
        compile_available=hasattr(torch, "compile"), trace_available=hasattr(torch.jit, "trace")))
    g = gradient_check()
    print("Gradient checks passed:", g, flush=True)
    t, model = training()
    print("Training finished:", t, flush=True)
    e = execution(model)
    print("Execution-mode comparison finished", flush=True)
    c = contiguity()
    report(g, t, e, c)
    print("Report:", ROOT / "report.md", flush=True)
