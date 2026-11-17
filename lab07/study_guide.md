# Lab 7 study guide

Use the manual, measured report and code together. The explanations here prepare you to describe your own experiments.

## Reading map for your textbook

The supplied Reddi edition is July 31, 2025, at `E:\introduction-to-machine-learning-systems_compress.pdf`. Viewer page = printed page + 52. Section boundaries may share a page.

| Reading | Printed pages | Viewer pages | Focus |
|---|---|---|---|
| Chapter 9: Efficient AI | 479-540 | 531-592 | Efficiency and whole-system trade-offs |
| 9.3: pillars | 494-504 | 546-556 | Algorithmic, compute and data efficiency |
| 9.4-9.7 | 504-524 | 556-576 | Interdependencies, trade-offs and end-to-end thinking |
| 10.3: Model Optimization Dimensions | 547-550 | 599-602 | Representation, precision and architectural execution |
| 10.4: Model Representation Optimization | 550-584 | 602-636 | This edition's title for the assigned compression reading |
| 10.4.1: Pruning | 551-565 | 603-617 | Magnitude criteria, masks, structured removal, recovery |
| 10.4.2: Knowledge Distillation | 565-571 | 617-623 | Teacher/student, soft targets, temperature and combined loss |

The manual asks for data/model/compute/energy columns. The edition's actual named pillars in 9.3 are algorithmic, compute and data efficiency; the report uses the requested laboratory table and treats energy as a separate system concern.

## Follow the experiment

`main.py` loads JSON config and checks independent compression tests. It builds one exact Fashion-MNIST split, trains the teacher, makes independent teacher copies for each pruning experiment, and trains both students. Every experiment ends with saved weights, storage files, validation/test evaluation and a summary. Once all training stops, the script repeatedly times every model in shuffled order. `summarize.py` then builds the report, figures and dashboard; `verify_results.py` reloads all final checkpoints and checks the evidence.

The baseline is oversized mainly because Linear(1568,640) has 1,004,160 parameters. A convolution weight is reused at many spatial positions, whereas a dense-head weight contributes one MAC per image. Parameter counts and executed compute therefore tell different stories.

## Explain the three pruning methods

Suppose layer A has weights [0.01, 0.02] and layer B has [0.3, 0.4]. At 50% global pruning, both weights in A disappear; at 50% layer-wise pruning, one disappears in each layer. Global pruning allocates sparsity according to the shared magnitude ranking, while layer-wise pruning imposes a separate quota. Magnitude is a heuristic: scales and layer sensitivity can differ.

Sparsity is zero eligible weights divided by all eligible weights. Biases are excluded here. Exact ranking avoids threshold ties accidentally pruning too many weights. Stable sorting gives deterministic tie-breaking. Masks stay fixed for all three fine-tuning epochs; neither ranking nor target is recomputed.

Setting weights to zero once is insufficient because updates or optimizer momentum can make them nonzero again. This code masks gradients and projects masked positions to zero after every update. It uses a fresh optimizer for every pruning point. The saved masks and epoch assertions make this verifiable.

Structured pruning ranks whole convolution output filters by L2 norm, removes the lowest-scoring filters and builds smaller convolution/linear layers. If Conv1 loses a channel, Conv2 must lose its matching input slice. If Conv2 loses a channel, the flattened fc1 input loses its 7x7=49 corresponding columns. Biases for removed outputs must also disappear. The pre-fine-tuning equivalence test catches channel-order and bias mistakes.

Requested filter removal is rounded separately per layer. With 16 and 32 output channels, 10% removal rounds to 2 and 3 filters. A compact network can have almost no stored zeros after removing half its original filters; filter-removal rate and weight-zero sparsity are distinct.

## Storage is a format decision

An FP32 dense matrix with a million entries occupies about four million payload bytes even when 90% are zero. A CSR representation stores values plus their column positions and row boundaries. With FP32 values and int32 indices, the approximate cost is 8 bytes per nonzero plus row pointers. At low sparsity it can be larger than dense storage. At high sparsity it can save disk space.

The laboratory compares the same uncompressed NPZ container for dense and sparse arrays, avoiding gzip efficiency as a confounder. Header/shape overhead is included. Each sparse archive is decoded back to the exact dense tensor. Model checkpoints, mask evidence, runtime RAM and disk archives are different quantities.

## Hardware latency must be measured

Dense PyTorch convolution/linear kernels process the same tensor shapes after unstructured masking. Zero values do not select a sparse kernel automatically. A nominal nonzero MAC count is an upper-level opportunity, not a CPU instruction count. A real speedup needs an execution path that exploits the sparsity, enough work to amortize index overhead, and compatible hardware.

Smaller structured dimensions reduce dense MACs and weight traffic, but alignment, kernel selection, threading and fixed overhead still affect latency. Batch 128 throughput answers a different question from batch-one response time. The benchmark records both, uses warm-up and repeated shuffled blocks, and saves the raw samples. It measures a warm model forward with pre-created inputs, not loading or image preprocessing.

Five hundred batch-one observations can describe timing variation, but they are correlated measurements on one computer. A p95 is not a guarantee or a confidence interval. Inspect the block medians before interpreting a small difference. The report never assumes speedup simply from a lower parameter count.

## Work through distillation

Hard labels tell the student the correct class. Soft teacher probabilities also reveal relative alternatives: an ambiguous shirt might look more like a T-shirt than a sneaker. Temperature T divides logits before softmax. Larger T usually produces softer class probabilities; T=4 and alpha=0.5 are fixed by the manual. T squared compensates the temperature scaling of gradients, especially in the usual high-temperature approximation.

The plain student minimizes CE on its unscaled logits. The distilled student uses `0.5 * CE + 0.5 * 16 * KL`, with class sums and a batch mean. It has the same architecture, initial weights, sample order, optimizer and epoch budget as the plain student, so teacher guidance is the intended changed factor.

**Direction matters.** The manual literally specifies KL(student || teacher). The implementation sums `p_student * (log_p_student - log_p_teacher)`. The student probabilities remain in the gradient graph. Common tutorials use KL(teacher || student), as does `kl_div(log_student, teacher_probabilities)`. These losses are not interchangeable. This package follows the supplied equation, states the choice and tests its gradient numerically. Discuss a different instructor convention before changing it. The textbook presents the common teacher-to-student direction.

Teacher training is an extra cost. This implementation computes frozen teacher logits once for the training split and stores the temporary cache in RAM. Cache generation time is recorded separately. Labels and validation/test images are never passed to a training teacher. At inference the student is completely independent: no teacher, soft-target cache or temperature 4 is needed. Distillation can help, hurt or tie; the measured comparison must say which occurred.

## Compute and energy

MAC counts are calculated from actual convolution and dense shapes. A multiply-add is counted as two FLOPs; biases, pooling and activations are excluded. Nonzero-weight MACs estimate an ideal zero-skipping algorithm and are not substituted for dense executed work.

There is no power meter in this experiment. At an assumed 35 W and measured 1 ms inference, E = 35 * 0.001 = 0.035 J = 35 mJ. At a measured 100 s training-plus-validation wall time, E = 35 * 100 / 3600 = 0.972 Wh. The package saves 15/35/65 W scenarios. None is a measured hardware power or a confidence range. Different models may draw different power, so this exercise cannot establish actual energy savings. Count teacher preparation, student training and eventual deployment volume when considering a lifecycle trade-off.

## Questions to practise aloud

1. Why do global and layer-wise pruning produce different layer sparsities?
2. What keeps zeros fixed when Adam updates the weights?
3. Why is structured pruning more than setting filters to zero?
4. Why can sparse storage be larger at 25% sparsity?
5. Why might 90% fewer nonzero parameters give no dense CPU speedup?
6. What exactly is included in the latency timer?
7. Why are the plain and distilled student initial hashes identical?
8. Which KL direction does the manual require, and which does the usual PyTorch call compute?
9. Why must teacher preparation cost be distinguished from student inference cost?
10. What can and cannot be concluded from fixed-power energy estimates and one random seed?

## Official API references

- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html): masks, reparameterization and pruning semantics. The measured code uses explicit masks rather than forward-hook reparameterization.
- [PyTorch KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html): input/target direction and batchmean reduction. The measured code explicitly implements the manual's reverse direction.
- [PyTorch 2.3 benchmark utilities](https://docs.pytorch.org/docs/2.3/benchmark_utils.html): warm-up, thread count, repeats and timing variation. The experiment uses a fixed repeat protocol with perf_counter_ns.
- [Torchvision FashionMNIST](https://docs.pytorch.org/vision/0.18/generated/torchvision.datasets.FashionMNIST.html): dataset loading and download option.

These web links are supporting API references; the experiment runs the installed course versions recorded in `results/environment.json`. Review the manual and assigned textbook rather than treating this guide as a substitute.
