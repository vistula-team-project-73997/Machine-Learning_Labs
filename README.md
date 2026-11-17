# Machine Learning Systems Laboratory Repository

This repository contains the reproducible work for the Machine Learning Systems laboratory sequence.

## Lab 1

Lab 1 creates a pinned Python environment, trains Logistic Regression and Random Forest baselines on the Breast Cancer Wisconsin diagnostic dataset, measures system cost, and evaluates deployment against Cloud, Edge, Mobile, and TinyML budgets.

Run the lab from the repository root with:

```text
python lab01/src/print_versions.py > lab01/results/versions.txt
python lab01/src/run_lab01.py
```

The required Python version is 3.11.8. Install the pinned dependencies with `pip install -r requirements.txt` inside an isolated environment.

## Lab 3

Lab 3 compares the exact MLP, CNN and attention architectures on Fashion-MNIST.
Run `python lab03/main.py` using Python 3.11.8 and the dependencies in
`lab03/requirements-lab03.txt`. The same tested interpreter used for Lab 2 now
also contains the required Lab 3 packages. See `lab03/RUN_IN_PYCHARM.txt`.

The report is `lab03/report.md`, measured evidence is in `lab03/results/`,
and the personal guide is `lab03/lab3 explanation.pdf`. Question 3 still
requires the student's handwritten calculations: follow `hand_calculations.md`
and add scans under `lab03/results/handwritten/`.

The exact architecture specifications conflict with the 100k-150k parameter
target. The report keeps the specified layers and explains the discrepancy.
It uses the supplied July 31, 2025 textbook edition and documents profiler
coverage and the existing course-wide package dependency conflict.

## Lab 4

Lab 4 builds a reproducible Adult data pipeline with validation, a 70/30
stratified split, train-only preprocessing, duplicate and label audits,
two derived features and a logistic comparison. Run `python lab04/main.py`
using Python 3.11.8 and `lab04/requirements-lab04.txt`.

See `lab04/report.md`, `lab04/data_card.md`, `lab04/results/`,
`lab04/RUN_IN_PYCHARM.txt` and `lab04/lab4 explanation.pdf`.
The digital manual is under `lab04/references/`. Source and processed data
are in ignored root `data/`. Both fits reproduce identical processed bytes
and predictions; fresh-process artifact checks also pass.

Baseline/engineered test accuracy: 0.853955/0.857777; F1: 0.660102/0.669626.
Five conflicting-label groups and remaining cross-split duplicate overlap
are documented. AI-assisted source and draft conclusions require student
review. Complete the 50-row manual worksheet yourself before submission;
completed human inspection is not claimed.

## Lab 5

Lab 5 implements a scalar automatic-differentiation engine, validates it
against PyTorch and numerical gradients, trains identical 2->8->1 models
on make_moons, compares warmed eager/traced execution and measures the
cost of non-contiguous tensor storage. Five-run training medians and raw
timing blocks are retained under `lab05/results/`.

Open `lab05/main.py` in this same course project and select the prepared
PyCharm run configuration `Lab 5`. Use the tested Python 3.11.8 environment
documented in `lab05/RUN_IN_PYCHARM.txt`. The requirements are recorded in
`lab05/requirements-lab05.txt`; the entire course stack is not required.

Read `lab05/report.md`, `lab05/study_guide.md`, `lab05/TASK_CHECKLIST.md`
and `lab05/lab5 explanation.pdf`. The digital manual is in
`lab05/references/`. The default compilation backend has no working C++
compiler here, so the manual-approved tracing fallback is used.

The maximum numerical-gradient error is 1.623e-10 and the engine matches
PyTorch on the required expression. Five correctness tests and saved-artifact
verification pass. This is AI-assisted preparation; understand the code,
review your prescribed reading and follow the opening classroom instructions.

## Lab 6

Lab 6 restructures the best Lab 3 CNN into a single YAML-driven training
pipeline with genuine local MLflow logs, checkpoints and full-epoch PyTorch
profiling. It measures workers 0/1/2/4/8, batches 32/64/128/256/512 over five
epochs, and all six SGD+momentum/Adam and scheduler pairs over ten epochs.
Gradient accumulation and a real epoch-three process interruption/restart
are independently checked against their reference runs.

Open lab06/main.py and select the Lab 6 PyCharm configuration. The existing
Python 3.11.8 environment now includes mlflow-skinny 2.14.1 and psutil 5.9.8
for genuine tracking and CPU memory measurements. See lab06/RUN_IN_PYCHARM.txt.
All 21 completed MLflow runs and 119 training epochs have measured evidence.

Read lab06/report.md, lab06/study_guide.md and lab06/lab6 explanation.pdf.
Open lab06/results/experiment_dashboard.html for the offline experiment view.
The one-page lab06/project_proposal.pdf covers a compact INT8 clothing-image
classifier with <=50 ms p95 CPU latency and <=5 MB artifact targets. These
are proposed constraints; instructor approval remains pending.

This is AI-assisted preparation. The recorded verification passes, the earlier
labs are preserved, and no work has been sent or published externally.

## Lab 7: Compression - pruning and distillation

Lab 7 adds an oversized 1,015,370-parameter Fashion-MNIST CNN, global and
layer-wise magnitude pruning at six sparsities, physical L2 filter pruning,
and plain/distilled 103,018-parameter students. All 18 experiments and 75
full-data epochs have saved evidence. Every pruning model fine-tunes for
three epochs with its specified structure/mask preserved.

Open lab07/main.py and select the Lab 7 PyCharm configuration with the same
tested Python 3.11.8 interpreter. Complete matching experiments are skipped;
final CPU latency is refreshed. Read lab07/report.md, lab07/study_guide.md and
lab07/lab7 explanation.pdf. Open lab07/results/experiment_dashboard.html for
the offline sortable comparison. Run lab07/src/verify_results.py to verify
saved evidence without retraining.

The comparison distinguishes dense execution from sparse disk storage and
physical shape reduction. It follows the manual's literal KL(student ||
teacher) at T=4 and alpha=0.5. Energy is reported as fixed-power estimates,
not measurements. AI-assisted preparation; review before submission.
