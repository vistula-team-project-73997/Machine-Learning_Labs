# Lab 5 preparation

Read [study_guide.md](study_guide.md) first, then [report.md](report.md) for actual measurements. A digital copy of the one-page manual is in `references/`.

Your course copy is inside the same folder as Labs 1-4:

`E:\semester 5\Lab01_GM_Robi_Farhan_Rojob_73997_submission\Lab01_GM_Robi_Farhan_Rojob_73997\lab05`

From that course repository root in PowerShell:

```powershell
& 'C:\Users\rojob\Documents\ChatGPT\Machine Learning\.venv-lab2\Scripts\python.exe' 'lab05\main.py'
```

In PyCharm, open your existing `Lab01_GM_Robi_Farhan_Rojob_73997_submission` project on E:. Expand `Lab01_GM_Robi_Farhan_Rojob_73997`, then `lab05`, open **main.py**, and choose **Run**. The prepared **Lab 5** run configuration selects the tested Python 3.11.8 interpreter explicitly. See `RUN_IN_PYCHARM.txt`. Lab 5 resolves output paths from its source location, so a different working directory is fine.

The launcher runs five independent correctness tests, checks the manual's expression against PyTorch and central differences, trains identical scalar/PyTorch models five times, compares warmed eager/compiled/traced inference, measures a non-contiguous view and its equal-valued copy, then regenerates the report. Expect several minutes for scalar training and first compilation. Progress appears after each training run. No dataset download is needed.

The existing environment already has all four required packages. `requirements-lab05.txt` pins the required course versions; `requirements-lock.txt` records the complete environment actually used, including packages from earlier labs. Neither is a claim that the full root course stack is installed. For a new environment:

```powershell
python -m venv .venv-lab05
& '.\.venv-lab05\Scripts\python.exe' -m pip install -r lab05/requirements-lab05.txt
& '.\.venv-lab05\Scripts\python.exe' lab05/main.py
```

Use Python 3.11.8 for those course pins. `torch.compile` is attempted with the default backend, an explicit forward function, an actual first execution and a graph-capture check. If unsupported, the program logs the error and uses `torch.jit.trace`, as permitted by the manual. `compile` existing as an attribute or returning a callable alone is not considered proof of support. On this machine the backend cannot find a C++ compiler, so the recorded comparison uses tracing. TorchScript is used here for compatibility with the prescribed PyTorch 2.3.1 stack.

Core files:

- `src/autodiff.py`: scalar Value, operators, iterative topological sort and reverse accumulation.
- `src/test_autodiff.py`: shared-node, repeated-edge, ReLU, long-graph, mixed-operation and full-model checks.
- `src/experiments.py`: validation, training, graphs, timings and report generation.
- `src/verify_results.py`: saved-evidence consistency checks and fresh-process traced-model reload; run automatically after experiments.
- `results/`: measured CSVs, JSON summaries, graph text, training plot, split and saved model weights.
- `TASK_CHECKLIST.md`: manual requirements mapped to evidence.

Rerunning overwrites generated results and the report. The material is an AI-assisted preparation example; understand the engine and adapt choices to opening classroom instructions. Bring the reference PDF on your device or print it. The session time was not provided, so no arrival reminder has been scheduled.
