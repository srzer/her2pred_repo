# Reproducibility guide

Follow these steps to install dependencies, prepare features, train models, and
reproduce the results.

## 1. Create the environment

The reference environment uses Python 3.10 and CUDA 12.1:

```bash
git clone https://github.com/srzer/her2pred_repo.git
cd her2pred_repo
conda env create -f environment.yml
conda activate her2pred
python -m pip check
```

For an existing CUDA-enabled Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

To run the statistical analysis:

```bash
python -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
```

OpenSlide also needs its native library. On Debian/Ubuntu, install
`libopenslide0`. On macOS, install `openslide` with Homebrew.

### KFB / ASlide support

KFB feature extraction uses ASlide on Linux. The Conda environment includes
ASlide. To add it to an existing Linux Python 3.10 environment:

```bash
python -m pip install -r requirements-kfb.txt
python -c "from Aslide import Slide"
```

The requirements file installs ASlide version `v1.7.0`. Use the import spelling
`from Aslide import Slide` on Linux.

ASlide bundles the KFB reader's native libraries. If the import succeeds but a
KFB slide fails to open because a shared library cannot be found, initialize
the bundled runtime paths before importing `Slide`:

```python
from Aslide.bootstrap import setup_runtime_environment

setup_runtime_environment()
from Aslide import Slide
```

Alternatively, source the `setup_env.sh` installed inside the `Aslide` package
before running feature extraction. See the
[upstream ASlide installation guide](https://github.com/MrPeterJin/ASlide/blob/main/docs/installation.md)
for supported Linux distributions and backend-specific native dependencies.

## 2. Download checkpoints

Download the `zero_shot` and `ndpi_adversarial` checkpoints from
[Hugging Face](https://huggingface.co/zhezi12138/her2-prediction-checkpoints):

```bash
hf download zhezi12138/her2-prediction-checkpoints \
  --repo-type model \
  --include 'zero_shot/*.pth' \
  --local-dir checkpoints

hf download zhezi12138/her2-prediction-checkpoints \
  --repo-type model \
  --include 'ndpi_adversarial/*.pth' \
  --local-dir checkpoints
```

## 3. Prepare the dataset

Download the feature dataset:

```bash
hf download zhezi12138/her2pred_pt_filtered_vista03 \
  --repo-type dataset --local-dir features
```

The training script prepares the downloaded features automatically. The TCGA
labels and few-shot split are provided in
[`tcga_standard_b_few_shot.csv`](../manifests/tcga_standard_b_few_shot.csv).

For WSI feature extraction, see the [README example](../README.md#feature-extraction).
Use `python her2pred/extract_features.py --help` for all extraction options.

## 4. Train models

Run five folds for each experiment:

```bash
for mode in ndpi_adversarial zero_shot few_shot; do
  for fold in 1 2 3 4 5; do
    python scripts/train_final.py \
      --mode "$mode" --fold "$fold" --gpu 0 \
      --data-root ./features \
      --view "training_view/$mode" --out "results/$mode"
  done
done
```

Each fold saves its checkpoint and validation predictions in
`results/<mode>/final_<mode>_fold<fold>/`:

- `ckpts/fold_<fold>.pth`: model checkpoint.
- `predictions_valid.csv`: OOF validation predictions used for evaluation.

The training script uses 16 epochs and saves the final model from each fold.
Training settings are listed in
[`configs/final_protocol.json`](../configs/final_protocol.json).

## 5. Run inference

### TCGA

Run this command in Bash after training `zero_shot`. It averages the five
fold probabilities and calculates the evaluation metrics:

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
  --manifest manifests/tcga_standard_b_few_shot.csv \
  --feature-root ./features/tcga_data \
  --cohort full --out-dir evaluation/zero_shot
```

For `few_shot`, set `mode=few_shot`, use `--cohort heldout`, and change
`--out-dir` to `evaluation/few_shot`.

### SVS/KFB

Evaluate the trained `ndpi_adversarial` model in Bash:

```bash
mode=ndpi_adversarial
args=()
for fold in 1 2 3 4 5; do
  run_dir="results/$mode/final_${mode}_fold${fold}"
  args+=(--ckpt "$run_dir/ckpts/fold_${fold}.pth")
  args+=(--valid-pred "$run_dir/predictions_valid.csv")
done
python her2pred/inference_external_ensemble.py "${args[@]}" \
  --root-dir ./features --out-dir evaluation/ndpi_adversarial
```

Both programs save ensemble predictions, summary metrics, and ROC/PR plots in
the evaluation directory. The classification threshold is calculated from the
OOF validation predictions.

## 6. Reproduce prediction statistics

Recompute ROC-AUC, PR-AUC, confidence intervals, and paired statistical tests
from the prediction tables in `results/`:

```bash
python analysis/reproduce_prediction_statistics.py
```

The command regenerates `results/prediction_statistics.json`. See
[`results/README.md`](../results/README.md) for file descriptions.

## 7. Clinical analysis

The clinical analysis uses a CSV with one row per patient:

```bash
python analysis/clinical_confounding/analyze_clinical_confounding.py \
  --help
```

See the [clinical analysis README](../analysis/clinical_confounding/README.md)
for input columns and output files.

## 8. Verification

```bash
python -m compileall -q her2pred scripts analysis
git status --short
```
