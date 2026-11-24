# Lab 6: Training pipelines and project proposal

G M Robi Farhan Rojob | Student number 73997 | CII5SE039CI | Group 01

Use the same course project as Labs 1-5. Open `lab06/main.py` in PyCharm and run **Lab 6**, using the existing tested Python 3.11.8 interpreter:

```powershell
& 'C:\Users\rojob\Documents\ChatGPT\Machine Learning\.venv-lab2\Scripts\python.exe' lab06/main.py
```

The full assignment trains the unchanged Lab 3 CNN on all 55,000 training examples in sequential, separate processes. It runs 22 jobs representing 21 completed MLflow runs: one profile, five worker studies, five batch sizes, six optimizer/scheduler combinations, two accumulation comparisons, and a reference/interrupted/resumed pair. Allow substantial time for the 119 measured training epochs, Windows worker startup and profiler post-processing. Progress prints each epoch. The checked-in results are already generated; reading them does not require retraining.

Read `report.md` for measured results, `study_guide.md` for explanations, `lab6 explanation.pdf` for the companion guide and `project_proposal.pdf` for the one-page proposal. Open `results/experiment_dashboard.html` for an offline experiment overview. `TASK_CHECKLIST.md` maps every requirement to evidence. The supplied manual is preserved under `references/`.

Task 8 now proposes **ROJOB - Lightweight Fabric Defect Detection for Garment Quality Control**: an AITEX classification baseline, compact defect segmentation and INT8 evaluation, targeting <=50 ms p95 CPU latency per 256 x 256 patch and a <=5 MB inference model. `proposal.md` and `project_proposal.md` contain the same revised text; `project_proposal.pdf` is the one-page version for instructor review. These are planned project experiments. The completed Lab 6 training-pipeline studies below use Fashion-MNIST.

The single YAML config specifies the seed, data, model, optimizer settings, scheduler definitions, tracking directory and study budgets. Study-specific settings are derived from this object and saved/logged explicitly. The official Fashion-MNIST raw data are already in the existing course `data/` folder. Portable users must supply the same dataset or set `dataset.download: true` for torchvision to obtain it. Dataset archives are excluded from the submission ZIP.

To continue a partially completed study suite without rerunning completed experiments:

```powershell
& 'C:\Users\rojob\Documents\ChatGPT\Machine Learning\.venv-lab2\Scripts\python.exe' lab06/main.py --continue-studies
```

Run `src/verify_results.py` to check saved evidence and re-evaluate the batch-128 checkpoint without retraining. Run `src/test_pipeline.py` for the independent accumulation and restart tests.

`mlflow-skinny==2.14.1` supplies genuine local MLflow tracking, parameters, metrics, artifacts and client APIs. It is the tracking-only distribution, not an imitation of MLflow and not the optional web UI/server. The offline dashboard reads the exported real records. The full course stack's incompatible `mlflow==2.14.1` / `pyarrow==16.1.0` pins are not installed together. Use `requirements-lab06.txt` for this explicit subset and `requirements-lock.txt` for the exact measured environment. Earlier lab dependencies remain recorded in their own lock files.

For a fresh Python 3.11.8 environment:

```powershell
python -m venv .venv-lab06
& '.\.venv-lab06\Scripts\python.exe' -m pip install -r lab06/requirements-lab06.txt
& '.\.venv-lab06\Scripts\python.exe' lab06/main.py
```

MLflow FileStore metadata uses absolute artifact URIs. After copying the lab, `src/tracking.py` rebases only those location fields to the current folder; verification and summary generation do this automatically. Original run IDs, metrics and timestamps are preserved. For an optional MLflow web UI, install the full `mlflow==2.14.1` distribution in a compatible environment, rebase the store, then use `mlflow ui --backend-store-uri <absolute path to lab06/results/mlruns>`.

Rerunning refreshes generated evidence and the Markdown report. Rebuild the companion PDFs/ZIP using the workspace delivery builder if you want those to reflect a new run. Reports distinguish CPU process memory, four-stage profiling shares and wall-clock throughput. See the method before comparing numbers.

This is AI-assisted preparation. The individual proposal is a draft for instructor approval; no approval or submission is claimed. Review the implementation and assigned reading so you can explain it in class.
