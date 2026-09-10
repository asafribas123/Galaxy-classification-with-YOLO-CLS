# Galaxy Morphology Classification with YOLOv8-CLS

Reproducible pipeline for automated galaxy morphology classification using the public **Galaxy10 DECaLS** dataset and **YOLOv8n-CLS**.

This repository accompanies the work:

**Deep Learning-Based Morphological Analysis of Galaxies from SDSS and DECaLS Using YOLO-CLS**
Asaf Ribas — Universidade Federal do Rio Grande (FURG)

## Overview

The pipeline performs the complete experimental workflow required to reproduce the classification analysis:

1. downloads and verifies the Galaxy10 DECaLS dataset;
2. reads the HDF5 images, labels, and available metadata;
3. aggregates the 10 original Galaxy10 classes into three morphological macroclasses;
4. balances the macroclasses by random undersampling without replacement;
5. creates stratified train/validation/test splits;
6. trains YOLOv8n-CLS using multiple epoch budgets and random seeds;
7. selects the final configuration using validation performance only;
8. evaluates the selected model on the held-out test set;
9. performs resolution and augmentation ablations;
10. evaluates performance as a function of redshift;
11. compares YOLOv8n-CLS with reference architectures;
12. saves manifests, predictions, metrics, figures, checkpoints, and a complete execution report.

The test set is not used for hyperparameter selection or for defining redshift boundaries.

## Dataset

The pipeline uses the public **Galaxy10 DECaLS** dataset:

* File: `Galaxy10_DECals_NoDuplicated.h5`
* DOI: `10.5281/zenodo.10845026`
* Expected MD5: `a920094d5e470e3705f4691c0d6dff54`
* Image channels: `g`, `r`, and `z`
* Class labels: HDF5 field `ans`
* Additional metadata, when available: right ascension, declination, redshift, and pixel scale

The dataset is downloaded automatically by the notebook when it is not available locally. File integrity is checked before the analysis continues.

No additional catalog cross-match or external quality filtering is applied by this pipeline.

## Morphological Taxonomy

The 10 original Galaxy10 DECaLS classes are aggregated into three macroclasses:

| Original ID | Original class                   | Macroclass          |
| ----------: | -------------------------------- | ------------------- |
|           0 | Disturbed Galaxies               | `disturbed_merging` |
|           1 | Merging Galaxies                 | `disturbed_merging` |
|           2 | Round Smooth Galaxies            | `smooth`            |
|           3 | In-between Round Smooth Galaxies | `smooth`            |
|           4 | Cigar Shaped Smooth Galaxies     | `smooth`            |
|           5 | Barred Spiral Galaxies           | `disk_spiral`       |
|           6 | Unbarred Tight Spiral Galaxies   | `disk_spiral`       |
|           7 | Unbarred Loose Spiral Galaxies   | `disk_spiral`       |
|           8 | Edge-on Galaxies without Bulge   | `disk_spiral`       |
|           9 | Edge-on Galaxies with Bulge      | `disk_spiral`       |

The `disk_spiral` macroclass includes edge-on systems and therefore does not assume direct visual identification of spiral arms in those objects.

## Experimental Design

### Sampling and splits

* Class balancing: random undersampling without replacement
* Target size: size of the least populated macroclass
* Balancing seed: `42`
* Stratified split:

  * training: `70%`
  * validation: `15%`
  * test: `15%`
* Split seed: `42`

The balanced sample is an experimental dataset and must not be interpreted as representing the cosmological abundance of the three morphological classes.

### YOLOv8n-CLS configuration

| Parameter               | Value               |
| ----------------------- | ------------------- |
| Initial weights         | `yolov8n-cls.pt`    |
| Main image size         | `128 × 128` px      |
| Resolution ablation     | `128`, `256` px     |
| Epoch budgets           | `30`, `50`, `100`   |
| Random seeds            | `42`, `123`, `2026` |
| Batch size              | `16`                |
| Optimizer               | `AdamW`             |
| Initial learning rate   | `1e-3`              |
| Weight decay            | `5e-4`              |
| LR schedule             | cosine              |
| Early-stopping patience | `15` epochs         |
| Workers                 | `4`                 |
| Pretrained weights      | enabled             |
| Deterministic mode      | enabled             |

The final YOLO configuration is selected from the validation results using macro F1-score, balanced accuracy, and training time as ranking criteria.

### Data augmentation

The default physically motivated augmentation configuration is:

* rotation: up to `180°`;
* horizontal flip probability: `0.5`;
* vertical flip probability: `0.5`;
* translation: `0.05`;
* scale variation: `0.10`;
* saturation variation: `0.10`;
* brightness/value variation: `0.20`;
* hue variation: disabled;
* automatic augmentation policies: disabled;
* random erasing: disabled.

A no-augmentation configuration is also evaluated as an ablation experiment.

### Reference models

The same train/validation/test partitions are used to evaluate:

* ResNet18;
* EfficientNet-B0;
* ViT-Tiny (`vit_tiny_patch16_224`).

These models use ImageNet-pretrained weights and an output layer adapted to the three macroclasses.

## Repository Structure

```text
.
├── Galaxy10_YOLOv8_pipeline_reprodutivel.ipynb
├── requirements.txt
└── README.md
```

During execution, the notebook creates:

```text
galaxy10_revision/
├── data/
├── dataset_balanced_3classes/
├── manifests/
├── results/
├── figures/
├── yolo_runs/
├── redshift_datasets/
└── baseline_runs/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/asafribas123/Galaxy-classification-with-YOLO-CLS.git
cd Galaxy-classification-with-YOLO-CLS
```

A dedicated Python environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a pinned `requirements.txt` is not available, the dependencies used by the notebook are:

```bash
pip install ultralytics h5py requests tqdm scikit-learn pandas matplotlib pillow joblib torch torchvision timm
```

For local notebook execution, install Jupyter if necessary:

```bash
pip install jupyterlab
```

## Reproduction

Open the notebook:

```bash
jupyter lab Galaxy10_YOLOv8_pipeline_reprodutivel.ipynb
```

Then execute the cells sequentially from top to bottom.

The complete workflow can be controlled through the `Config` dataclass near the beginning of the notebook. Computationally expensive analyses can be enabled or disabled using:

```python
run_epoch_sweep
run_resolution_ablation
run_augmentation_ablation
run_redshift_specialists
run_baselines
```

For a minimal execution test, set:

```python
quick_test = True
```

This reduces the experiment to one epoch and one random seed and is intended only to verify that the pipeline executes successfully.

## Reproducibility and Data Leakage Control

The pipeline implements the following controls:

* fixed seeds for Python, NumPy, PyTorch, sampling, splitting, and bootstrap procedures;
* deterministic PyTorch execution when supported;
* dataset integrity verification using MD5;
* explicit storage of the balanced sample and split manifests;
* stratified train/validation/test partitions;
* validation-only model selection;
* redshift boundaries estimated from the training data only;
* held-out test evaluation after model selection;
* explicit augmentation parameters;
* software and hardware information recorded during execution;
* automatic saving of the final experimental configuration and metrics.

The main execution record is written to:

```text
galaxy10_revision/results/complete_run_report.json
```

This file contains the dataset checksum, sample sizes, class counts, split sizes, taxonomy, redshift boundaries, selected YOLO configuration, test metrics, software versions, platform information, and experimental parameters.

## Main Outputs

Relevant outputs are stored under `galaxy10_revision/`, including:

* balanced dataset manifests;
* train/validation/test membership;
* validation and test predictions;
* classification metrics;
* confusion matrices;
* bootstrap estimates;
* redshift-dependent analyses;
* resolution and augmentation ablations;
* baseline-model comparisons;
* YOLO checkpoints;
* figures;
* `complete_run_report.json`.

Because model metrics may vary across software versions, hardware, and numerical backends, numerical results are not hard-coded in this README. The values generated by a specific execution should be taken from the saved result files and execution report.

## Citation

If you use this repository, please cite the associated work and the Galaxy10 DECaLS dataset.

```bibtex
@software{ribas_galaxy_yolocls,
  author = {Ribas, Asaf},
  title  = {Galaxy Morphology Classification with YOLOv8-CLS},
  url    = {https://github.com/asafribas123/Galaxy-classification-with-YOLO-CLS}
}
```

Dataset:

**Galaxy10 DECaLS** — DOI: `10.5281/zenodo.10845026`

## Author

**Asaf Ribas**
Universidade Federal do Rio Grande — FURG
Research interests: galaxy morphology, machine learning, and astronomical spectroscopy.
