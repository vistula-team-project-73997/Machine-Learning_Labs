# ROJOB - Fabric defect detection

**Student:** G M Robi Farhan Rojob | **Student ID:** 73997

**Course:** Machine Learning CII5SE039CI | **Group:** 01

**Title:** ROJOB - Lightweight Fabric Defect Detection for Garment Quality Control

**Status:** Revised proposal for instructor approval

## 1. Problem and intended use

Fashion brands need consistent garment quality, but manual inspection can miss small fabric defects. Inspired by quality-control challenges encountered while developing ROJOB, this project will build an image-based inspection prototype. Given a standardized close-up photograph of fabric, it will predict whether a visible defect is present and highlight suspicious regions for human review. The scope covers surface defects such as broken yarn, knots and contamination. The research question is whether a compact neural network can identify and localize fabric defects within strict CPU latency and model-size limits.

## 2. Dataset and preparation

Use the AITEX Fabric Image Database, described by its publisher as 245 images across seven fabric types, with normal images, defective images and pixel-level defect masks. Check duplicates and create fixed-seed training, validation and test partitions before extracting 256 x 256 patches; all patches from one original image must stay in the same partition. Training-only augmentation and class weighting will address limited data and imbalanced defect regions. Where available, separately labelled ROJOB fabric photographs will provide a pilot evaluation, reported separately from benchmark performance. Dataset source: https://www.aitex.es/afid/

## 3. Proposed method and experiments

Train a small CNN baseline for normal-versus-defective classification, then a compact U-Net-style segmentation model to predict defect regions. The segmentation output will produce a highlighted mask and an image-level inspection decision. Compare classification performance across both models and evaluate segmentation quality for the latter. Evaluate the segmentation model before and after INT8 quantization to measure quality, storage and latency trade-offs. Calibration will use training data only. Record configurations, checkpoints and experiments using YAML and MLflow. Select decision thresholds and model settings on validation data, then evaluate frozen choices on the untouched test set.

## 4. Proposed success metrics

Target defect recall >=90%, image-level macro-F1 >=0.85 and defect-region Dice score >=0.60, with Dice measured on defective test patches. Require a final serialized inference model <=5,000,000 bytes (5 MB) and batch-one p95 CPU latency <=50 ms for one 256 x 256 fabric patch. Measure preprocessing, inference and mask postprocessing using four CPU threads, 50 warm-up calls and 1,000 timed calls. Record median/p95 latency, model bytes and peak process memory. Report full-image scanning time separately.

## 5. Deliverables, feasibility and risks

Stage 1: audit data and reproduce the classification baseline. Stage 2: train segmentation and evaluate quantization. Stage 3: complete final evaluation and package reproducible code, trained models, experiment records, an upload-and-inspect demo and a comparative report. Main risks are limited data, very small defects, quantized operator support and differences between benchmark fabrics and ROJOB garments. The prototype supports human inspection. All targets are proposed outcomes to be verified experimentally; instructor approval is pending.
