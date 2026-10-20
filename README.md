# HER2 prediction pipeline

This repository provides the code and instructions to reproduce the HER2
prediction experiments.

Experiment results are summarized in
[`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md). Environment setup,
checkpoint download, training, inference, and analysis instructions are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Included workflow

1. `her2pred/extract_features.py`: WSI tile feature extraction.
2. `her2pred/build_filtered.py`: coordinate-matched tumor filtering.
3. `scripts/create_training_view.py`: prepares PyTorch feature files for training.
4. `scripts/train_final.py`: trains one fold of `ndpi_adversarial`,
   `zero_shot`, or `few_shot`.
5. `her2pred/inference_tcga_ensemble.py`: runs ensemble inference on TCGA.
   `her2pred/inference_external_ensemble.py` evaluates SVS/KFB.
6. `analysis/clinical_confounding/`: contains cohort summaries and clinical
   covariate analysis.
7. `analysis/reproduce_prediction_statistics.py`: recomputes ROC/PR metrics,
   confidence intervals, and paired P values.

## Public checkpoint

Five-fold model checkpoints are available at
[`zhezi12138/her2-prediction-checkpoints`](https://huggingface.co/zhezi12138/her2-prediction-checkpoints).
Experiment names follow the checkpoint folder names:

| Experiment | Description |
|---|---|
| `ndpi_vanilla` | NDPI-only vanilla baseline |
| `ndpi_adversarial` | NDPI-only adversarial model |
| `internal_vanilla` | Internal-data vanilla baseline |
| `macenko` | NDPI-only stain-normalized baseline |
| `zero_shot` | Prov-GigaPath zero-shot model |
| `resnet` | ResNet50 zero-shot model |
| `plip` | PLIP zero-shot model |
| `few_shot` | Few-shot model trained with 36 TCGA cases |

Training uses cached appearance targets stored at
`ndpi_adversarial/appearance_cache.npz` in the checkpoint repository. Download
the file and set `HER2PRED_APPEAR_FULL_CACHE` to its local path. The same cache
supports `ndpi_adversarial`, `zero_shot`, and `few_shot` training.

## Data

The TCGA labels and split are provided in
[`manifests/tcga_standard_b_few_shot.csv`](manifests/tcga_standard_b_few_shot.csv).
The `few_shot` experiment uses 36 cases for training and 288 for evaluation.

The PyTorch feature datasets are available at
[`zhezi12138/her2pred_pt_filtered_vista03`](https://huggingface.co/datasets/zhezi12138/her2pred_pt_filtered_vista03)
and
[`zhezi12138/her2pred_pt_vista03`](https://huggingface.co/datasets/zhezi12138/her2pred_pt_vista03).

## Feature extraction

Extract GigaPath features from a directory of HER2-positive NDPI slides:

```bash
python her2pred/extract_features.py \
  --format ndpi --encoder gigapath \
  --slide-dir /path/to/ndpi_positive --label 1 \
  --out extracted/ndpi_data/original --gpu 0
```

Use `--label 0` for negative slides. Each `.pt` file contains patch features,
coordinates, the slide ID, and its label. The default resolution is 0.5 µm/px.

| Option | Usage |
|---|---|
| `--format` | `ndpi`, `svs`, or `kfb` |
| `--encoder` | `gigapath`, `resnet50`, or `plip` |
| `--coords-from` | Extract patches at coordinates from matching reference `.pt` files |
| `--stain-norm macenko` | Apply Macenko stain normalization |
| `--augment` | Generate augmented features |

For encoder comparisons on the released tumor patches, use
`--coords-from features/ndpi_data/original` and select `resnet50` or `plip`.
Reference files are matched by slide filename. Save augmented copies to
`extracted/ndpi_data/augmented`. For KFB slides, use `--workers 0`.
GigaPath and PLIP pretrained weights are loaded from Hugging Face.

## Run experiments

Follow the [reproducibility guide](docs/REPRODUCIBILITY.md) to install the
environment and download the features and appearance cache. Then train all
five folds of an experiment:

```bash
export HER2PRED_APPEAR_FULL_CACHE="$PWD/checkpoints/ndpi_adversarial/appearance_cache.npz"
for fold in 1 2 3 4 5; do
  python scripts/train_final.py \
    --mode zero_shot --fold "$fold" --gpu 0 \
    --data-root ./features \
    --view training_view/zero_shot --out results/zero_shot
done
```

Use `ndpi_adversarial` or `few_shot` as the mode and directory names to run
those experiments. The guide includes [ensemble evaluation commands](docs/REPRODUCIBILITY.md#5-run-inference)
for the trained models.

To recompute the published metrics from the prediction tables:

```bash
pip install -r requirements-analysis.txt
python analysis/reproduce_prediction_statistics.py
```
