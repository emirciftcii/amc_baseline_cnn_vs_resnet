# AMC Project

This project compares a baseline 1D CNN and a small 1D ResNet for Automatic Modulation Classification on the DeepSig RadioML 2016.10A dataset. The code is intentionally kept modular so team members can work on data loading, models, training, and reporting separately.

## Setup

```powershell
conda env create -f environment.yml
conda activate amc_project
```

For GPU training on this Windows machine, a separate environment was created on the D: drive.

```powershell
conda activate D:\conda_envs\amc_gpu
```

If the OpenMP duplicate runtime error appears on Windows, use this temporary workaround in the same terminal:

```powershell
set KMP_DUPLICATE_LIB_OK=TRUE
```

## Dataset

Download `RML2016.10a_dict.pkl` manually and place it here:

```text
code/data/RML2016.10a_dict.pkl
```

The loader reads the local pickle with `encoding="latin1"` and builds stratified train/validation/test splits. The test set also keeps SNR labels so accuracy vs SNR can be reported.

## Run

From the project root:

```powershell
cd code
python main.py --model baseline_cnn --data-path data/RML2016.10a_dict.pkl --epochs 10 --num-workers 0
python main.py --model resnet1d --data-path data/RML2016.10a_dict.pkl --epochs 10 --num-workers 0
```

GPU run example:

```powershell
cd code
set KMP_DUPLICATE_LIB_OK=TRUE
conda run -p D:\conda_envs\amc_gpu python main.py --model resnet1d --data-path data/RML2016.10a_dict.pkl --epochs 10 --batch-size 512 --num-workers 0 --device cuda --save-dir results/gpu_resnet10
```

## Colab Notebook

For a step-by-step Google Colab workflow, open:

```text
code/notebooks/amc_colab_step_by_step.ipynb
```

Upload `RML2016.10a_dict.pkl` to Colab or set the notebook's `DATA_PATH` to a Google Drive location.

To run both models in one command:

```powershell
python main.py --model all --data-path data/RML2016.10a_dict.pkl --epochs 10 --num-workers 0
```

Use a different `--save-dir` when you want to keep older runs:

```powershell
python main.py --model all --data-path data/RML2016.10a_dict.pkl --epochs 10 --num-workers 0 --save-dir results/epoch10
```

## Outputs

Each model writes results under `code/results/<model_name>/`:

- `best_model.pt`: best validation-loss checkpoint
- `history.json`: train/validation loss and accuracy per epoch
- `summary.json`: test metrics, best epoch, model size, training time, SNR accuracy, and confusion matrix values
- `history.png`: training curves
- `accuracy_vs_snr.png`: SNR-based test accuracy
- `confusion_matrix.csv` and `confusion_matrix.png`: class-level confusion matrix

When `--model all` is used, `code/results/comparison_summary.json` gives a compact baseline vs ResNet comparison.

## Project References

- T. J. O'Shea et al., "Over-the-Air Deep Learning Based Radio Signal Classification", IEEE JSTSP, 2018.
- DeepSig RadioML 2016.10A dataset.
- `radioML/examples` dataset ecosystem examples.
- `brysef/rfml` PyTorch-based AMC framework.
