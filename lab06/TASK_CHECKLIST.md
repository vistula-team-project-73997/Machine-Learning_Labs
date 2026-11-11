# Lab 6 requirement map

Source: references/Lab Manual_ Labs 06.pdf, printed page 8. Document instructions are assignment requirements; they do not establish approval or authorize external submission.

| Task | Implementation and graded evidence |
|---|---|
| 1 [20] Best Lab 3 model, single YAML config, seed, checkpoints, MLflow params/metrics/artifacts | config.yaml; src/model.py; src/pipeline.py; references/lab03_summary.csv; results/mlruns; results/mlflow_export.json; per-run effective configs/checkpoints |
| 2 [15] Profile a complete epoch and four-stage time shares | results/profile_breakdown.csv; profile_operators.txt; profile_trace.json.gz; profile_breakdown.png; report method and interpretation |
| 3 [15] Workers 0,1,2,4,8, bottleneck and plot | results/worker_study.csv; worker_study.png; all three full epochs per setting in results/runs/workers_*/history.csv |
| 4 [15] Batches 32,64,128,256,512: throughput, peak memory, five-epoch accuracy | results/batch_study.csv; batch_study.png; fresh-process memory records and checkpoint per batch |
| 5 [15] StepLR, CosineAnnealingLR, ReduceLROnPlateau; SGD+momentum and Adam, equal ten epochs | Six combinations in results/optimizer_scheduler_study.csv; optimizer_schedules.png; per-epoch lr, next_lr and validation metrics |
| 6 [10] Physical batch 256 vs micro-batch 64 accumulated 4 | results/accumulation_comparison.json; accumulation_curves.png; independent uneven-tail unit test |
| 7 [10] Interrupt after epoch 3, resume and compare | Separate fresh processes in main.py; checkpoint_epoch3.pt; resume_comparison.json; resume_curves.png; full optimizer/scheduler/RNG verification |
| 8 [15] Individual one-page proposal with problem, data, success metric, <=50 ms, <=5 MB, optimization, student and ID | project_proposal.pdf; project_proposal.md; local course commit recorded in delivery record after delivery |
| Approved individual proposal | Prepared for review; instructor approval is pending and cannot be completed by the implementation |
| Prescribed reading and personal understanding | study_guide.md with verified reading map; personal reading remains the student's action |

The manual's task weights sum to 115; they are reproduced as supplied, without silently renormalizing. Actual measurements and verification status are in report.md and results/verification.txt. Nothing has been sent or published externally.
