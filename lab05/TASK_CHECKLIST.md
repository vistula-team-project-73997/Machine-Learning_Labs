# Manual requirements and preparation evidence

Source: `references/Lab Manual_ Labs 05.pdf`, printed page 7. Its instructions are assignment requirements, not additional authorization to submit or publish anything.

| Task | Evidence |
|---|---|
| 1. Value holds data, gradient, parents, local rule | `src/autodiff.py`, `Value` |
| 2. Addition, multiplication, powers, ReLU build graph | Operator methods; `results/scalar_graph.dot` |
| 3. Topological sort, seed 1, reverse visit, accumulation | `Value.topological`, `Value.backward`; reused-node/repeated-edge tests |
| 4. Match PyTorch and numerical gradients within 1e-6 | `results/gradient_check.csv`, `results/gradient_summary.json` |
| 5. make_moons, 2->8->1, sigmoid, BCE, five training runs | `src/experiments.py`; `results/training_runs.csv`, `results/loss_history.csv`, `report.md` |
| 6. Inspect PyTorch graph | `results/fx_forward_graph.txt`, `results/autograd_backward_graph.txt`, `results/traced_graph.txt` |
| 6. Warm eager vs compiled/traced timing | `results/execution_timings.csv`, `results/execution_summary.json` |
| 6. Shape, dtype, contiguity and layout timing | `results/contiguity_summary.json`, `results/contiguity_timings.csv` |
| Graded short note: when compilation helps | `report.md`, execution-mode discussion and break-even rule |
| Required environment and actual execution-mode support | `requirements-lab05.txt`, `results/environment.json`, `results/execution_summary.json` |
| Prescribed reading and fundamental concepts | Reading map and worked examples in `study_guide.md`; personal review remains yours |
| Bring manual | Digital copy provided in `references/`; bring the device or print it |
| Attendance and opening instructions | Personal action at the scheduled session; not claimed as completed |

The manual leaves sample count, noise, split, learning rate, epochs, initialization, hidden activation and detailed timing protocol unspecified. The preparation report states all those choices. Delivery follows the earlier labs: a local course-repository commit and a portable ZIP. Nothing has been sent to the instructor.
