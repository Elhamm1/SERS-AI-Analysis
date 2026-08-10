#!/usr/bin/env python3
"""SHAP analysis for the final optimized/preprocessed 80/20 CNN.

Place this file in:
  Finalized-Optimized-80-20Within-populationSplit/

Expected checkpoint:
  Unified_Raw_80_20_Comparison/Fold_1/best_cnn_validation_checkpoint.pth

Notes
-----
* This script is intended specifically for the finalized optimized/preprocessed
  80/20 run. The result-directory name still contains "Raw" because that is the
  historical output-folder name produced by the final optimized script. The SHAP
  analysis applies the same final CNN preprocessing:
  SG window 17, polynomial order 2, iterative ModPoly degree 2, no min-max.
* The exact 80/20 outer split is reconstructed with seed 42 and cultivar
  stratification, matching the final analysis.
* Put wavenumbers_1496.npy beside this script.
"""

from pathlib import Path
import argparse
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- configuration -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / "../../CrossValidationData").resolve()
RESULT_DIR = SCRIPT_DIR / "Unified_Raw_80_20_Comparison"
CHECKPOINT = RESULT_DIR / "Fold_1" / "best_cnn_validation_checkpoint.pth"
OUT_DIR = RESULT_DIR / "SHAP_Within_Cultivar"
AXIS_FILE = SCRIPT_DIR / "wavenumbers_1496.npy"

EXPECTED_PARENT_DIRNAME = "Finalized-Optimized-80-20Within-populationSplit"

INPUT_LENGTH = 1496
SEED = 42
N_BACKGROUND = 200
N_EXPLAIN = 300
TOP_K = 20

# Exact CNN preprocessing used in the final optimized/preprocessed 80/20 run.
SG_WINDOW = 17
SG_POLYORDER = 2
BASELINE_DEGREE = 2
BASELINE_KMAD = 2.5
BASELINE_MAX_ITER = 30
BASELINE_TOL = 1e-6
BASELINE_MIN_KEEP_FRACTION = 0.05

CULTIVAR_MEAN_CYS = {
    "AAC_Chrome": 0.317211, "AAC_Lacombe": 0.338561, "AAC_Liscard": 0.325405,
    "CDC_Amarillo": 0.358930, "CDC_Athabasca": 0.341203, "CDC_Canary": 0.314476,
    "CDC_Dakota": 0.332835, "CDC_Golden": 0.359066, "CDC_Greenwater": 0.346381,
    "CDC_Inca": 0.365342, "CDC_Jasper": 0.337341, "CDC_Striker": 0.344968,
    "CDC_Lewochko": 0.341267, "CDC_Meadow": 0.312012, "CDC_Patrick": 0.338404,
    "CDC_Saffron": 0.346789, "CDC_Spectrum": 0.373175, "CDC_Spruce": 0.345676,
    "CDC_Tetris": 0.342040, "Redbat88": 0.316257,
}

# Biochemical regions retained from the previous SHAP workflow.
BANDS = [
    (180, 300, "Ag-S / substrate"),
    (510, 518, "C-S stretch"),
    (660, 674, "S-S bridge"),
    (752, 762, "Tyr/Phe ring"),
    (825, 840, "Side chains (Tyr)"),
    (924, 936, "C-C backbone"),
    (996, 1008, "Phe marker"),
    (1080, 1126, "C-N / backbone"),
    (1200, 1270, "Amide III"),
    (1330, 1380, "CH2/CH3 deformation"),
    (1450, 1465, "CH bending"),
    (1590, 1610, "Aromatic ring"),
    (1645, 1665, "Amide I"),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_cultivar(filename):
    for cultivar in sorted(CULTIVAR_MEAN_CYS, key=len, reverse=True):
        if filename.startswith(cultivar):
            return cultivar
    return None


def standardize_loaded_array(arr, filename):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected shape {arr.shape} in {filename}")
    if arr.shape[1] != INPUT_LENGTH:
        raise ValueError(f"{filename}: length {arr.shape[1]} != {INPUT_LENGTH}")
    return arr.astype(np.float32, copy=False)


def load_all_spectra():
    files = sorted(DATA_DIR.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy spectra found in {DATA_DIR}")
    xs, rows = [], []
    for fp in files:
        cultivar = find_cultivar(fp.name)
        if cultivar is None:
            continue
        arr = standardize_loaded_array(np.load(fp, allow_pickle=False), fp.name)
        xs.append(arr)
        for array_index in range(arr.shape[0]):
            rows.append({"cultivar": cultivar, "filename": fp.name, "array_index": array_index})
    X = np.vstack(xs).astype(np.float32, copy=False)
    metadata = pd.DataFrame(rows)
    if len(X) != len(metadata):
        raise RuntimeError("Data/metadata length mismatch")
    return X, metadata


def _mad(values):
    median = np.median(values)
    return float(np.median(np.abs(values - median)) + 1e-12)


def iterative_modified_polynomial_baseline(spectrum, degree):
    y = np.asarray(spectrum, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    weights = np.ones(y.size, dtype=np.float64)
    previous = weights.copy()
    minimum_points = max(degree + 2, int(BASELINE_MIN_KEEP_FRACTION * y.size), 5)
    for _ in range(BASELINE_MAX_ITER):
        coefficients = np.polyfit(x, y, deg=degree, w=weights)
        baseline = np.polyval(coefficients, x)
        residual = y - baseline
        threshold = BASELINE_KMAD * _mad(residual)
        new_weights = (residual <= threshold).astype(np.float64)
        if int(new_weights.sum()) < minimum_points:
            keep = np.argsort(residual)[:minimum_points]
            new_weights = np.zeros_like(new_weights)
            new_weights[keep] = 1.0
        if np.mean(np.abs(new_weights - previous)) < BASELINE_TOL:
            weights = new_weights
            break
        previous = weights
        weights = new_weights
    coefficients = np.polyfit(x, y, deg=degree, w=weights)
    return np.polyval(coefficients, x)


def preprocess_cnn(X):
    processed = savgol_filter(
        X, window_length=SG_WINDOW, polyorder=SG_POLYORDER, axis=1, mode="interp"
    ).astype(np.float32)
    corrected = np.empty_like(processed, dtype=np.float32)
    for i in range(len(processed)):
        baseline = iterative_modified_polynomial_baseline(processed[i], BASELINE_DEGREE)
        corrected[i] = (processed[i].astype(np.float64) - baseline).astype(np.float32)
    if not np.all(np.isfinite(corrected)):
        raise FloatingPointError("Non-finite values after CNN preprocessing")
    return corrected


class Cys1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16); self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32); self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64); self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(128); self.pool4 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear((INPUT_LENGTH // 16) * 128, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def normalize_shap_array(values):
    if isinstance(values, list):
        values = values[0]
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    elif values.ndim == 3 and values.shape[2] == 1:
        values = values[:, :, 0]
    elif values.ndim != 2:
        values = values.reshape(values.shape[0], -1)
    return values.astype(np.float32, copy=False)


def sample_indices(indices, n, rng):
    indices = np.asarray(indices)
    n = min(int(n), len(indices))
    return rng.choice(indices, size=n, replace=False)


def save_outputs(axis, X_exp, shap_values, predictions, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:TOP_K]
    feature_names = np.array([f"f_{int(round(v))} cm^-1" for v in axis])

    np.save(out_dir / "shap_values.npy", shap_values)
    np.save(out_dir / "X_explained.npy", X_exp)
    np.save(out_dir / "predictions.npy", predictions)
    np.save(out_dir / "wavenumbers_used.npy", axis)

    pd.DataFrame({"wavenumber_cm^-1": axis, "mean_abs_shap": mean_abs}).to_csv(
        out_dir / "shap_global_importance.csv", index=False
    )
    pd.DataFrame({
        "rank": np.arange(1, len(top_idx) + 1),
        "feature_index": top_idx,
        "wavenumber_cm^-1": axis[top_idx],
        "mean_abs_shap": mean_abs[top_idx],
    }).to_csv(out_dir / "shap_top_features.csv", index=False)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_exp, feature_names=feature_names, max_display=TOP_K, show=False)
    plt.title("1D-CNN - Within cultivar")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(axis, mean_abs)
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Mean |SHAP|")
    plt.title("Within-cultivar CNN: global SHAP importance")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_importance.png", dpi=300)
    plt.close()

    avg_spec = np.mean(X_exp, axis=0)
    def norm01(a):
        a = np.asarray(a, dtype=float)
        r = a.max() - a.min()
        return (a - a.min()) / (r if r > 0 else 1.0)
    plt.figure(figsize=(12, 4))
    plt.plot(axis, norm01(avg_spec), label="Normalized spectrum")
    plt.plot(axis, norm01(mean_abs), label="Normalized SHAP")
    for lo, hi, label in BANDS:
        plt.axvspan(lo, hi, alpha=0.12, label=label)
    handles, labels = plt.gca().get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    plt.legend(h2, l2, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Normalized value")
    plt.title("Within-cultivar CNN: spectrum and SHAP importance")
    plt.tight_layout()
    plt.savefig(out_dir / "overlay_spectrum_shap_highlighted_bands.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-background", type=int, default=N_BACKGROUND)
    parser.add_argument("--n-explain", type=int, default=N_EXPLAIN)
    args = parser.parse_args()

    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    if SCRIPT_DIR.name != EXPECTED_PARENT_DIRNAME:
        raise RuntimeError(
            f"This script is configured to run from {EXPECTED_PARENT_DIRNAME}, "
            f"but it is currently located in {SCRIPT_DIR}. "
            "Move the script to the finalized optimized 80/20 directory."
        )
    if not AXIS_FILE.is_file():
        raise FileNotFoundError(
            f"Missing Raman axis: {AXIS_FILE}\nCopy wavenumbers_1496.npy beside this script."
        )
    axis = np.load(AXIS_FILE, allow_pickle=False).reshape(-1).astype(float)
    if len(axis) != INPUT_LENGTH:
        raise ValueError(f"Raman axis length {len(axis)} != {INPUT_LENGTH}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    # Guard against the historical output-directory naming collision: both the
    # raw and optimized/preprocessed scripts used the name
    # Unified_Raw_80_20_Comparison in some runs. Refuse to interpret a raw
    # checkpoint as the preprocessed CNN if model_selection.csv says otherwise.
    selection_path = RESULT_DIR / "Fold_1" / "model_selection.csv"
    if selection_path.is_file():
        sel = pd.read_csv(selection_path)
        cnn_rows = sel[sel["Model"].astype(str).str.upper() == "CNN"] if "Model" in sel.columns else pd.DataFrame()
        if not cnn_rows.empty and "Selected_Parameters" in cnn_rows.columns:
            text = " ".join(cnn_rows["Selected_Parameters"].astype(str).tolist()).lower()
            if "preprocessing" not in text or "sg_window" not in text:
                raise RuntimeError(
                    "The Fold_1 checkpoint appears to come from the RAW 80/20 run, not the "
                    "optimized/preprocessed run. The two runs historically shared the same output "
                    "directory name. Restore/rerun the optimized/preprocessed output before SHAP."
                )

    X_raw, metadata = load_all_spectra()
    all_idx = np.arange(len(X_raw))
    dev_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, random_state=SEED, stratify=metadata["cultivar"]
    )

    bg_idx = sample_indices(dev_idx, args.n_background, rng)
    exp_idx = sample_indices(test_idx, args.n_explain, rng)
    X_bg = preprocess_cnn(X_raw[bg_idx])
    X_exp = preprocess_cnn(X_raw[exp_idx])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cys1DCNN().to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    model.eval()

    bg_t = torch.as_tensor(X_bg, dtype=torch.float32, device=device).unsqueeze(1)
    exp_t = torch.as_tensor(X_exp, dtype=torch.float32, device=device).unsqueeze(1)
    explainer = shap.GradientExplainer(model, bg_t)
    shap_values = normalize_shap_array(explainer.shap_values(exp_t))

    with torch.no_grad():
        predictions = model(exp_t).detach().cpu().numpy().reshape(-1)

    save_outputs(axis, X_exp, shap_values, predictions, OUT_DIR)
    metadata.iloc[exp_idx].reset_index(drop=True).to_csv(OUT_DIR / "explained_spectra_metadata.csv", index=False)
    print(f"Done. Outputs: {OUT_DIR.resolve()}")
    print(f"Checkpoint: {CHECKPOINT.resolve()}")
    print(f"Background N={len(X_bg)}; explained N={len(X_exp)}")


if __name__ == "__main__":
    main()
