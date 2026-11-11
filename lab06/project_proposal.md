# Individual project proposal

**Student:** G M Robi Farhan Rojob | **Student ID:** 73997  
**Course:** Machine Learning CII5SE039CI, Group 01  
**Title:** Compact clothing-image classification with CPU INT8 quantization  
**Status:** Proposal for instructor approval

## Problem and intended use

Build an offline classifier that maps a centered 28x28 grayscale clothing image to one of ten Fashion-MNIST categories. The deliverable is a reproducible local inference demo and an accuracy/latency/storage comparison of an FP32 baseline and an INT8 model. Scope is the benchmark's image distribution; this is not a claim of reliable real-store photograph recognition.

## Dataset and evaluation

Use the official Fashion-MNIST dataset: 60,000 training images and 10,000 test images. Preserve the seed-42 55,000/5,000 training/validation split used in Labs 3 and 6. Normalize using mean 0.2860 and standard deviation 0.3530. Use validation only to select a training checkpoint and quantization settings. Calibrate on a fixed 1,000-image training subset, never on validation/test. Freeze choices, then report final test accuracy, macro-F1, per-class recall and confusion matrices. Repeat baseline training for three seeds within a fixed 20-epoch budget per seed; report variation rather than selecting a seed by test score.

## Model and planned optimization

Start with the 20,490-parameter Lab 3 CNN: two 3x3 convolution/ReLU/pooling blocks and a ten-class linear head. Train an FP32 baseline using the Lab 6 YAML/MLflow/checkpoint pipeline. Apply post-training static INT8 quantization to supported convolution/linear blocks using the PyTorch x86 CPU backend, fusing eligible convolution/ReLU operations and recording calibration indices. If operator support or accuracy is inadequate, investigate quantization-aware fine-tuning for at most five additional epochs. Report unsuccessful results as well as improvements.

## Success metrics and deployment constraints

Target test accuracy >=90% and macro-F1 >=0.88, with INT8 accuracy degradation <=1 percentage point relative to the frozen FP32 baseline. Require serialized inference artifact <=5,000,000 bytes and p95 batch-one CPU latency <=50 ms on this Windows laptop using four inference threads. Measure 50 warm-up calls and 1,000 timed calls; include in-memory normalization and inference, excluding disk/UI input. Record median, p95, model bytes and peak process RSS. Aim for at least 2x smaller artifact, measuring speed rather than assuming INT8 is faster. The existing small FP32 model may already satisfy both hard constraints; optimization still needs an honest benefit/cost analysis.

## Feasibility, risks and deliverables

Stage 1: lock data/splits and reproduce baseline. Stage 2: calibrate/quantize and validate operator support. Stage 3: freeze choices, run final accuracy and latency checks, then package the demo, configs, MLflow evidence and a concise report. CPU quantized-kernel support, calibration sensitivity and benchmark-to-photo domain shift are the main risks. Instructor approval is requested before treating this plan as the approved individual project; the targets above are proposed, not achieved results.
