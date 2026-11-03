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
