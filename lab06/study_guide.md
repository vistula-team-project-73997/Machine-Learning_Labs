# Lab 6 study guide

Use this with the measured report and the supplied manual. Practise explaining the experiments without reading the implementation.

## Reading map for your supplied textbook

The source is your July 31, 2025 Reddi textbook at `E:\introduction-to-machine-learning-systems_compress.pdf`. Its PDF viewer pages are printed pages plus 52. Boundaries can include the start of the next section.

| Reading | Printed pages | Viewer pages | Focus for this lab |
|---|---|---|---|
| 5.5 Model Development | 215-218 | 267-270 | Controlled experiments, validation and model selection |
| 8.2 Training Systems | 387-391 | 439-443 | Training is a coupled data/compute/optimization system |
| 8.3.2 Optimization Algorithms | 399-405 | 451-457 | SGD, momentum and adaptive updates |
| 8.4 Pipeline Architecture | 408-422 | 460-474 | Data loading, forward, backward and parameter updates |
| 8.5 Pipeline Optimizations | 422-439 | 474-491 | Prefetch, overlap, accumulation and checkpointing |
| 8.7 Optimization Techniques | 457-460 | 509-512 | Measure bottlenecks before changing settings |
| 13.3 MLOps Key Components | 859-875 | 911-927 | Repeatable pipelines, experiment records and observability |

## Trace one experiment

The root `main.py` reads one YAML object, derives an effective setting and starts a separate training process. `src/pipeline.py` seeds the environment, creates the same Lab 3 CNN and data split, constructs an optimizer/scheduler, logs the configuration to MLflow, trains each epoch, validates, steps the scheduler, logs metrics and saves a complete checkpoint. Source and measured evidence are artifacts of that real MLflow run.

The CNN is reused as an architecture. Every independent comparison starts from identical seed-42 parameters; the Lab 3 pretrained weights are not mixed into only some settings. The model has 20,490 parameters: first convolution 16*(1*3*3+1)=160; second 32*(16*3*3+1)=4,640; linear 10*(32*7*7+1)=15,690. Total 20,490.

## Explain a profile correctly

A parent forward range contains many child tensor operations. Adding its inclusive time to all those children's inclusive times counts the same interval repeatedly. This implementation adds only four disjoint parent ranges. It reports their share of the four-stage sum, and separately their share of epoch wall time. Batch scoring and Python loop bookkeeping occupy the residual.

Data-loading time means time inside iterator creation and `next(iterator)`. At zero workers, it includes transformations/collation; with workers, it mostly measures waits on already-prefetched work and IPC. It does not measure the sum of worker CPU usage. Raw Fashion-MNIST arrays are loaded once before the timer, so these are not per-image disk-read measurements.

Profiler instrumentation adds overhead. Use the unprofiled worker and batch runs for throughput. A Chrome trace is compressed to reduce size; decompress the `.json.gz` before opening it in a trace viewer.

## Why worker count can hurt

Windows starts fresh Python worker processes. Each imports libraries, receives dataset state and communicates batches through multiprocessing queues. Persistent workers pay that startup once per experiment. They can overlap transformations with model computation, but IPC, extra process memory and CPU contention can outweigh the benefit for small in-memory images. Compare the first epoch to epochs two and three, and inspect both time and process-tree memory. The best count is measured on this workload.

## Batch size changes several things at once

Training throughput is samples divided by training-loop seconds. Larger batches can use kernels more efficiently and require more activation memory. They also mean fewer optimizer steps per epoch. At 55,000 examples, batch 32 makes 1,719 updates per epoch; batch 512 makes 108. Five epochs expose every image five times in both cases, but the optimizer follows different paths. Do not compare final accuracy as though batch size only changed memory.

CPU RSS/working set includes interpreter, libraries, dataset, tensors and allocator caches. It is not the model's serialized size and is not GPU memory. Fresh processes make comparisons cleaner; baseline, sampled increments and native peak help explain what is counted. Sampled process-tree memory can double-count shared pages.

## Optimizers and schedules

SGD uses the current gradient; momentum carries a decayed update direction across steps. Adam tracks first and second moments, so the optimizer checkpoint contains more than the model's parameters. The comparison fixes starting rates at 0.01 for SGD and 0.001 for Adam; this is not a search proving either optimizer universally best.

StepLR halves the rate after every three epochs. A rate of 0.001 is used for epochs 1-3, 0.0005 for 4-6 and 0.00025 for 7-9. CosineAnnealingLR smoothly reduces the rate over ten epochs. ReduceLROnPlateau monitors validation loss and reduces only after its configured plateau condition; it may never trigger if loss keeps improving. The rate logged as `lr` is used in that epoch; `next_lr` is the rate after the scheduler update.

## Work through accumulation and its last batch

For four equal micro-batches of 64 examples, the desired gradient is the mean over 256 examples. Accumulate gradients of summed per-example losses, divide the accumulated gradient by 256, and update once. Do not zero the gradients between those four backward passes.

The dataset has 55,000 examples, and 55,000 mod 256 = 216. The last group has micro-batches 64,64,64,24. Its correct mean uses denominator 216. Averaging four micro-batch means gives the 24-example micro-batch too much weight. The code divides by the actual group count, and the independent test uses 19 examples to force an uneven tail.

The exact Lab 3 CNN has no BatchNorm or dropout. BatchNorm computes statistics over the physical micro-batch, so accumulation would not generally reproduce a larger physical batch in a model that contains it. Float32 reduction order can still create small differences here; the report compares actual curves.

## Recover training, not just weights

Suppose training ends after epoch three and StepLR halves the rate. The saved model is only part of the state. Restore optimizer moments, scheduler counters/rate, next epoch, Python/NumPy/Torch random states and the independent DataLoader generator. Otherwise the next shuffle or update changes. Save after the scheduler step so the checkpoint knows the rate for epoch four.

The experiment uses three separate process executions: six epochs uninterrupted; three epochs and exit; reload and run four through six. Matching full curves and final model/optimizer/RNG states provides stronger evidence than visually smooth curves alone. Timing cannot match exactly. Recovery is at an epoch boundary; mid-batch recovery would also need sampler position and any partially accumulated gradient group.

## Understand the proposal

The proposed project compares an FP32 clothing-image classifier with an INT8 version. Its targets are not measured achievements. The final inference artifact must be <=5,000,000 bytes, and measured p95 batch-one CPU response must be <=50 ms. Model file size and process RSS are separate quantities. Calibration uses training data; validation selects options; the final test set evaluates frozen choices.

The one-page proposal still needs instructor approval. Read it and decide whether you can explain and carry out the stated plan.

## Questions to practise

1. Where does the YAML become the optimizer's actual learning rate?
2. Why does summing nested profiler times exaggerate the total?
3. Why might eight workers be slower and use more memory?
4. Why do equal epochs not imply equal optimizer-update budgets?
5. What denominator does the final accumulation group use, and why?
6. What does an Adam checkpoint need beyond the weights?
7. Why is the final project test set kept out of quantization calibration?

Official API references: [MLflow 2.14.1 Tracking](https://mlflow.org/docs/2.14.1/tracking.html), [PyTorch 2.3 data loading](https://docs.pytorch.org/docs/2.3/data.html), [profiler](https://docs.pytorch.org/docs/2.3/profiler.html), [learning-rate schedules](https://docs.pytorch.org/docs/2.3/optim.html#how-to-adjust-learning-rate). The measured code uses the locally installed 2.3.1 implementations.
