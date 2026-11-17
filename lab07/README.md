# Lab 7: Compression - pruning and distillation

G M Robi Farhan Rojob | 73997 | CII5SE039CI | Group 01

Open `lab07/main.py` in the same PyCharm course project used for earlier labs. Use the existing Python 3.11.8 interpreter:

```powershell
& 'C:\Users\rojob\Documents\ChatGPT\Machine Learning\.venv-lab2\Scripts\python.exe' lab07/main.py
```

The full assignment trains 18 models across 75 full-data epochs: a 1,015,370-parameter teacher, 12 global/layer-wise pruning points, three physically compact filter-pruned networks, and plain/distilled 103,018-parameter students. Every pruning point fine-tunes for three epochs. Model training runs sequentially; final latency benchmarking runs after all training has stopped.

`main.py` skips complete experiments only if their code/config fingerprint matches. An interrupted experiment restarts that experiment. Use `--rerun` to retrain everything, or `--benchmark-only` to refresh timings after all 18 model results exist. Running the launcher refreshes timing evidence and the Markdown report; rebuild the companion PDF/ZIP afterwards. Reading the supplied report and plots requires no training.

Read `report.md` for measured results, `study_guide.md` for concepts and reading pages, `lab7 explanation.pdf` for the companion guide and `TASK_CHECKLIST.md` for requirement-to-evidence mapping. Open `results/experiment_dashboard.html` for the offline sortable/filterable overview. `results/comparison.csv` includes accuracy, parameters, nonzero counts, dense/CSR storage, batch-1/128 latency, MACs and energy scenarios. Raw latency, per-layer sparsity, epoch records and reloadable models are retained.

Fashion-MNIST is already in the shared course `data/` folder. On another machine, set `download` to `true` in `config.json` before the first run. Network access is then needed only to obtain the dataset. The normalization and 55k/5k split match the earlier course convention. Preprocessing is cached in RAM before training timers.

For a new Python 3.11.8 environment:

```powershell
python -m venv .venv-lab07
& '.\.venv-lab07\Scripts\python.exe' -m pip install -r lab07/requirements-lab07.txt
& '.\.venv-lab07\Scripts\python.exe' lab07/main.py
```

`requirements-lab07.txt` lists only the required course versions; `requirements-lock.txt` records the existing measured environment. CUDA is not used. No new packages are needed in the established course interpreter.

To verify all saved evidence without retraining:

```powershell
& 'C:\Users\rojob\Documents\ChatGPT\Machine Learning\.venv-lab2\Scripts\python.exe' lab07/src/verify_results.py
```

The manual's KL direction is followed literally: KL(student || teacher), T=4, alpha=0.5. This differs from the usual teacher-target direction; see the report before adapting tutorial code. Structured pruning slices tensors, whereas unstructured inference uses dense kernels. Energy is **estimated under fixed-power scenarios**, not measured.

This is AI-assisted preparation with measured evidence. Review the implementation and conclusions so you can explain them in class. Nothing is submitted externally by these scripts.
