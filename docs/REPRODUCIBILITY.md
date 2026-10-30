# Reproducibility guide

## 1. Environment

We use Python 3.10 and CUDA 12.1:

```bash
git clone https://github.com/srzer/her2pred_repo.git
cd her2pred_repo
conda env create -f environment.yml
conda activate her2pred
python -m pip check
```

An existing CUDA-enabled Python 3.10 environment can use
`python -m pip install -r requirements.txt`.
OpenSlide also needs `libopenslide0` on Debian/Ubuntu or `openslide` from
Homebrew on macOS. KFB extraction uses the Linux-only ASlide backend, included
in the Conda environment. `requirements-kfb.txt` installs the pinned `v1.7.0`
version separately; the import is `from Aslide import Slide`. Native-library
setup is described in the
[ASlide installation guide](https://github.com/MrPeterJin/ASlide/blob/main/docs/installation.md).

## 2. Data and checkpoints

You can train models with the commands below or download the published
five-checkpoint ensembles from
[Hugging Face](https://huggingface.co/zhezi12138/her2-prediction-checkpoints).
Available ensembles are `ndpi_vanilla`, `ndpi_adversarial`, `internal_vanilla`,
`macenko`, `zero_shot`, `resnet`, `plip`, and `few_shot`. You can select one
with `mode`:

```bash
mode=zero_shot
hf download zhezi12138/her2pred_pt_filtered_vista03 \
  --repo-type dataset --local-dir features
hf download zhezi12138/her2-prediction-checkpoints \
  --repo-type model --include "$mode/*.pth" --local-dir checkpoints
```

TCGA labels and the few-shot split are in
[`tcga_split.csv`](../manifests/tcga_split.csv).
The [README](../README.md#feature-extraction) covers WSI feature extraction.
ResNet50, PLIP, and Macenko checkpoints require features from the corresponding
encoder or stain normalization.

The training launcher supports `ndpi_vanilla`, `ndpi_adversarial`, `zero_shot`,
and `few_shot`. This example trains one fold of `zero_shot` for 16 epochs:

```bash
mode=zero_shot
python scripts/train_final.py \
  --mode "$mode" --fold 1 --gpu 0 \
  --data-root ./features \
  --view "training_view/$mode" --out "results/$mode"
```

You can select a model with `mode` and a fold from 1 to 5 with `--fold`.
Each run writes `ckpts/fold_<fold>.pth` and `predictions_valid.csv` under
`results/<mode>/final_<mode>_fold<fold>/`. Training settings are in
[`final_protocol.json`](../configs/final_protocol.json).

## 3. Ensemble inference

Both programs average the five checkpoint probabilities. These examples use
training outputs; downloaded checkpoints use
`checkpoints/<mode>/fold_<fold>.pth` for each `--ckpt` instead. TCGA inference
also requires the matching OOF validation files; SVS/KFB inference can use
`--valid-threshold` instead of those files, with frozen values recorded in
[`final_protocol.json`](../configs/final_protocol.json).

For TCGA:

```bash
mode=zero_shot
args=()
for fold in 1 2 3 4 5; do
  run_dir="results/$mode/final_${mode}_fold${fold}"
  args+=(--ckpt "$run_dir/ckpts/fold_${fold}.pth")
  args+=(--result-dir "$run_dir")
  args+=(--valid-pred "$run_dir/predictions_valid.csv")
done
python her2pred/inference_tcga_ensemble.py "${args[@]}" \
  --manifest manifests/tcga_split.csv \
  --feature-root ./features/tcga_data \
  --cohort full --out-dir "evaluation/$mode"
```

For `few_shot`, you can use `mode=few_shot` with `--cohort heldout`.

For SVS/KFB:

```bash
mode=ndpi_adversarial
args=()
for fold in 1 2 3 4 5; do
  run_dir="results/$mode/final_${mode}_fold${fold}"
  args+=(--ckpt "$run_dir/ckpts/fold_${fold}.pth")
  args+=(--valid-pred "$run_dir/predictions_valid.csv")
done
python her2pred/inference_external_ensemble.py "${args[@]}" \
  --root-dir ./features --out-dir "evaluation/$mode"
```

The same command supports `mode=ndpi_vanilla`. Both programs save predictions,
summary metrics, and ROC/PR plots. Classification thresholds come from OOF
validation predictions.

## 4. Analysis and verification

The published statistics reproduce directly from the prediction tables in
`results/`, without model training or inference:

```bash
python -m pip install -r requirements-analysis.txt
python analysis/reproduce_prediction_statistics.py
python -m compileall -q her2pred scripts analysis
git status --short
```

The statistics command regenerates `results/prediction_statistics.json`;
[`results/README.md`](../results/README.md) describes the files. Clinical
analysis requires a separate patient-level CSV, as described in the
[clinical analysis README](../analysis/clinical_confounding/README.md).
