# Lab 7 requirement-to-evidence checklist

The supplied one-page manual is preserved in `references/`. Its task weights add to 115; all five tasks are covered.

| Task | Implementation | Evidence |
|---|---|---|
| 1 [15]: oversized Fashion-MNIST baseline, about 1M parameters | `src/models.py`, `src/run_lab.py`; 1,015,370 parameters | `results/runs/teacher/`, report task 1, comparison CSV |
| 2 [35]: global/layer-wise magnitude pruning at 0/25/50/75/90/95% | Exact sorted masks over weight tensors; independent teacher copies | Saved `masks.pt`, `layer_sparsity.csv`, all 12 run folders |
| Three fine-tuning epochs with fixed masks | Gradient mask and post-update zero projection; per-epoch assertions | Each pruning `epochs.csv`: 3 epochs, zero mask violations; independent tests |
| Structured L2 filter pruning at 10/25/50% | Physical tensor slicing across adjacent layers | Three structured run folders, kept filter indices, reference-equivalence error |
| Accuracy-versus-sparsity plots | Validation before/after FT plus final test curves | `sparsity_curves.png`, `structured_curves.png` |
| 3 [15]: honest hardware latency | Five shuffled timing blocks, warm-up, batches 1/128, CPU four threads | `latency_raw.csv`, `latency_blocks.csv`, `comparison.csv`, report task 3 |
| Dense versus sparse on-disk storage | Same uncompressed NPZ, FP32/int32 CSR, exact reload checks | Each run: `dense.npz`, `csr.npz`, `model.pt`; `storage_latency.png` |
| Explain lack of dense-kernel speedup from unstructured sparsity | Same shapes and executed dense MAC; nonzero MAC clearly idealized | Report task 3 and explanation PDF |
| 4 [30]: about-100k student with/without teacher | 103,018 parameters; identical init, order, epochs and optimizer | Plain/distilled run folders, init hashes, `distillation_curves.png` |
| Temperature 4, alpha 0.5, manual's KL direction | Explicit KL(student \|\| teacher), detached teacher | Config, `src/models.py`, numerical-gradient test, report task 4 |
| 5 [20]: one comparison table by data/model/compute/energy | All 18 models, full-data exposures, measured compute/latency/storage, estimated energy | Report task 5, `comparison.csv`; no measured energy claim |
| Reproducibility and verification | Config/source/checkpoint hashes, exact split, versions | Environment JSON, lock file, `verification.txt`, artifact manifest |

All student/teacher training uses ten epochs; all 15 pruning experiments use three. The 0% cases are fine-tuning controls. The fixed dataset and extra distillation preparation costs are documented. Review before submission; no instructor approval or external submission is claimed.
