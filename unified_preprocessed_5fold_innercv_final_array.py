
"""
FINAL 20-LABEL PREPROCESSED VERSION — nested leave-one-cultivar-out validation.

- The modeled target is the mean HPLC cysteine concentration for each cultivar
  across the three field locations (20 distinct cultivar-level response values).
- One complete cultivar is held out in each of 20 outer folds.
- The held-out cultivar is never used for model hyperparameter or CNN epoch selection.
- Five grouped inner folds are formed from the 19 development cultivars.
- The same model-specific preprocessing configurations identified in the 80/20
  preprocessing study are applied here as fixed preprocessing.
- PLSR, RFR, and SVR hyperparameters are selected only within the five grouped
  inner folds of each outer LOCO fold.
- Each inner CNN is fitted for exactly 100 epochs; the epoch with the smallest
  mean validation loss across the five inner folds is selected.
- A fresh CNN is then fitted on all 19 development cultivars for that epoch count.
- The outer held-out cultivar is evaluated only after all inner selection is complete.
- Fold-specific R2 is undefined because each held-out cultivar has one constant target.
"""

import os
import argparse
import re
import glob
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Configuration
# ============================================================

INPUT_LENGTH = 1496
DATA_DIR = Path("../../CrossValidationData")
SEED = 42

# Fixed preprocessing settings carried over from the completed 80/20
# preprocessing optimization. These settings are not re-optimized in LOCO.
OPTIMIZED_PREPROCESSING = {
    "LR": {"sg_window": 11, "sg_polyorder": 3, "baseline_degree": 3, "minmax": True},
    "PLSR": {"sg_window": 5, "sg_polyorder": 4, "baseline_degree": 2, "minmax": False},
    "RFR": {"sg_window": 17, "sg_polyorder": 2, "baseline_degree": 2, "minmax": True},
    "SVR": {"sg_window": 5, "sg_polyorder": 4, "baseline_degree": 2, "minmax": True},
    "CNN": {"sg_window": 17, "sg_polyorder": 2, "baseline_degree": 2, "minmax": False},
}

BASELINE_KMAD = 2.5
BASELINE_MAX_ITER = 30
BASELINE_TOL = 1e-6
BASELINE_MIN_KEEP_FRACTION = 0.05

# CNN settings
CNN_MAX_EPOCHS = 100
CNN_INNER_FOLDS = 5
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-4
CNN_WEIGHT_DECAY = 1e-4
CNN_HUBER_BETA = 0.02
CNN_USE_AMP = False

# Small, practical candidate grids for the diagnostic comparison.
PLS_COMPONENTS = [2, 3, 5, 7, 10, 15, 20]
RFR_CANDIDATES = [
    {"n_estimators": 300, "max_depth": 12, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 20, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 2},
]
SVR_CANDIDATES = [
    {"C": 0.5, "epsilon": 0.01, "gamma": "scale"},
    {"C": 1.0, "epsilon": 0.01, "gamma": "scale"},
    {"C": 2.0, "epsilon": 0.02, "gamma": "scale"},
    {"C": 5.0, "epsilon": 0.02, "gamma": "scale"},
    {"C": 2.0, "epsilon": 0.05, "gamma": 0.01},
]

LOCATION_CODE_TO_NAME = {
    "01": "Limerick",
    "02": "Rosthern",
    "03": "Sutherland",
}

# Mean HPLC cysteine concentration across the three field locations.
CULTIVAR_MEAN_CYS = {
    "AAC_Chrome":     0.317211,
    "AAC_Lacombe":    0.338561,
    "AAC_Liscard":    0.325405,
    "CDC_Amarillo":   0.358930,
    "CDC_Athabasca":  0.341203,
    "CDC_Canary":     0.314476,
    "CDC_Dakota":     0.332835,
    "CDC_Golden":     0.359066,
    "CDC_Greenwater": 0.346381,
    "CDC_Inca":       0.365342,
    "CDC_Jasper":     0.337341,
    "CDC_Striker":    0.344968,
    "CDC_Lewochko":   0.341267,
    "CDC_Meadow":     0.312012,
    "CDC_Patrick":    0.338404,
    "CDC_Saffron":    0.346789,
    "CDC_Spectrum":   0.373175,
    "CDC_Spruce":     0.345676,
    "CDC_Tetris":     0.342040,
    "Redbat88":       0.316257,
}


MODEL_ORDER = ["LR", "PLSR", "RFR", "SVR", "CNN"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# Data loading and metadata
# ============================================================

def find_cultivar_from_filename(filename: str):
    for cultivar in sorted(CULTIVAR_MEAN_CYS, key=len, reverse=True):
        if filename.startswith(cultivar):
            return cultivar
    return None


def parse_spectrum_filename(filename: str):
    cultivar = find_cultivar_from_filename(filename)
    if cultivar is None:
        raise ValueError(f"Cannot identify cultivar in filename: {filename}")

    location_match = re.search(r"_(001|002|003|01|02|03)_", filename)
    spot_match = re.search(r"_(R[123])_(\d+)\.npy$", filename)

    if location_match is None:
        raise ValueError(f"Cannot identify location code in filename: {filename}")
    if spot_match is None:
        raise ValueError(f"Cannot identify spot and replicate in filename: {filename}")

    location = {"001": "01", "002": "02", "003": "03",
                "01": "01", "02": "02", "03": "03"}[location_match.group(1)]
    spot = spot_match.group(1)
    replicate = int(spot_match.group(2))
    return cultivar, location, spot, replicate


def standardize_loaded_array(arr: np.ndarray, filename: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected shape {arr.shape} in {filename}")

    if arr.shape[1] != INPUT_LENGTH:
        raise ValueError(
            f"{filename}: spectrum length {arr.shape[1]} does not equal {INPUT_LENGTH}"
        )
    return arr.astype(np.float32, copy=False)


def load_all_spectra():
    file_paths = sorted(DATA_DIR.glob("*.npy"))
    if not file_paths:
        raise RuntimeError(f"No .npy files found in {DATA_DIR.resolve()}")

    X_parts, y_parts, rows = [], [], []
    for file_path in file_paths:
        filename = file_path.name
        cultivar, location, spot, replicate = parse_spectrum_filename(filename)
        arr = standardize_loaded_array(np.load(file_path), filename)

        target = float(CULTIVAR_MEAN_CYS[cultivar])

        X_parts.append(arr)
        y_parts.append(np.full(arr.shape[0], target, dtype=np.float32))

        for array_index in range(arr.shape[0]):
            rows.append({
                "filename": filename,
                "cultivar": cultivar,
                "location": location,
                "location_name": LOCATION_CODE_TO_NAME[location],
                "spot": spot,
                "replicate": replicate,
                "array_index": array_index,
                "sample_group": f"{cultivar}_{location}",
                "spot_group": f"{cultivar}_{location}_{spot}",
            })

    X = np.vstack(X_parts).astype(np.float32, copy=False)
    y = np.concatenate(y_parts).astype(np.float32, copy=False)
    metadata = pd.DataFrame(rows)

    if len(X) != len(metadata):
        raise RuntimeError("Data and metadata lengths do not match.")
    return X, y, metadata



# ============================================================
# Fixed optimized preprocessing
# ============================================================

def _mad(values):
    median = np.median(values)
    return float(np.median(np.abs(values - median)) + 1e-12)


def iterative_modified_polynomial_baseline(
    spectrum,
    degree,
    kmad=BASELINE_KMAD,
    max_iter=BASELINE_MAX_ITER,
    tol=BASELINE_TOL,
    min_keep_fraction=BASELINE_MIN_KEEP_FRACTION,
):
    y = np.asarray(spectrum, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    weights = np.ones(y.size, dtype=np.float64)
    previous = weights.copy()
    minimum_points = max(degree + 2, int(min_keep_fraction * y.size), 5)

    for _ in range(max_iter):
        coefficients = np.polyfit(x, y, deg=degree, w=weights)
        baseline = np.polyval(coefficients, x)
        residual = y - baseline
        threshold = kmad * _mad(residual)
        new_weights = (residual <= threshold).astype(np.float64)

        if int(new_weights.sum()) < minimum_points:
            keep = np.argsort(residual)[:minimum_points]
            new_weights = np.zeros_like(new_weights)
            new_weights[keep] = 1.0

        if np.mean(np.abs(new_weights - previous)) < tol:
            weights = new_weights
            break

        previous = weights
        weights = new_weights

    coefficients = np.polyfit(x, y, deg=degree, w=weights)
    return np.polyval(coefficients, x)


def apply_baseline_correction(X, degree):
    corrected = np.empty_like(X, dtype=np.float32)
    for index in range(len(X)):
        baseline = iterative_modified_polynomial_baseline(X[index], degree=degree)
        corrected[index] = (X[index].astype(np.float64) - baseline).astype(np.float32)
    return corrected


def apply_per_spectrum_minmax(X):
    X = np.asarray(X, dtype=np.float32)
    row_min = np.min(X, axis=1, keepdims=True)
    row_max = np.max(X, axis=1, keepdims=True)
    denominator = np.where((row_max - row_min) > 1e-12, row_max - row_min, 1.0)
    return ((X - row_min) / denominator).astype(np.float32)


def preprocess_for_model(X, model_name):
    """Apply the fixed model-specific preprocessing from the 80/20 study."""
    config = OPTIMIZED_PREPROCESSING[model_name]

    processed = savgol_filter(
        X,
        window_length=config["sg_window"],
        polyorder=config["sg_polyorder"],
        axis=1,
        mode="interp",
    ).astype(np.float32)

    processed = apply_baseline_correction(
        processed,
        degree=config["baseline_degree"],
    )

    if config["minmax"]:
        processed = apply_per_spectrum_minmax(processed)

    if not np.all(np.isfinite(processed)):
        raise FloatingPointError(
            f"Non-finite values produced during preprocessing for {model_name}"
        )

    return processed.astype(np.float32, copy=False)


# ============================================================
# Metrics and output helpers
# ============================================================

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return np.nan
    return float(r2_score(y_true, y_pred))


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
        "Bias": float(np.mean(y_pred - y_true)),
        "N": int(len(y_true)),
    }


def make_prediction_table(model_name, fold_name, y_true, y_pred, metadata):
    out = metadata.copy().reset_index(drop=True)
    out.insert(0, "Model", model_name)
    out.insert(1, "Fold", fold_name)
    out["True_Cys"] = np.asarray(y_true, dtype=float)
    out["Predicted_Cys"] = np.asarray(y_pred, dtype=float)
    out["Residual"] = out["Predicted_Cys"] - out["True_Cys"]
    return out


def aggregate_predictions(prediction_df, unit_columns):
    grouped = (
        prediction_df
        .groupby(["Model", "Fold"] + list(unit_columns), as_index=False)
        .agg(
            True_Cys=("True_Cys", "first"),
            Mean_Predicted_Cys=("Predicted_Cys", "mean"),
            SD_Predicted_Cys=("Predicted_Cys", "std"),
            N_Spectra=("Predicted_Cys", "size"),
        )
    )
    grouped["SD_Predicted_Cys"] = grouped["SD_Predicted_Cys"].fillna(0.0)
    grouped["Residual"] = grouped["Mean_Predicted_Cys"] - grouped["True_Cys"]
    return grouped


def plot_model_comparison(comparison_df, output_path, title):
    ordered = comparison_df.set_index("Model").reindex(MODEL_ORDER).reset_index()
    x = np.arange(len(ordered))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ordered["MAE"], width, label="MAE")
    ax.bar(x + width / 2, ordered["RMSE"], width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["Model"])
    ax.set_ylabel("Error (g/100 g)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_predictions(unit_df, output_path, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    for model_name in MODEL_ORDER:
        sub = unit_df[unit_df["Model"] == model_name]
        if sub.empty:
            continue
        ax.scatter(
            sub["True_Cys"],
            sub["Mean_Predicted_Cys"],
            alpha=0.75,
            label=model_name,
        )

    lo = float(min(unit_df["True_Cys"].min(), unit_df["Mean_Predicted_Cys"].min()))
    hi = float(max(unit_df["True_Cys"].max(), unit_df["Mean_Predicted_Cys"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", label="Identity")
    ax.set_xlabel("Reference Cys (g/100 g)")
    ax.set_ylabel("Mean predicted Cys (g/100 g)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================
# Conventional models
# ============================================================

def fit_lr(X_dev, y_dev):
    model = LinearRegression()
    model.fit(X_dev, y_dev)
    return model, "No hyperparameter selection; fixed optimized LR preprocessing used"


def select_plsr_grouped_cv(X_dev, y_dev, inner_splits):
    max_allowed = max(1, min(X_dev.shape[0] - 1, X_dev.shape[1]))
    candidates = [k for k in PLS_COMPONENTS if k <= max_allowed] or [1]
    rows = []
    for k in candidates:
        fold_rmse = []
        for train_rel, val_rel in inner_splits:
            model = PLSRegression(n_components=k, scale=False, max_iter=1000)
            model.fit(X_dev[train_rel], y_dev[train_rel])
            pred = model.predict(X_dev[val_rel]).ravel()
            fold_rmse.append(np.sqrt(mean_squared_error(y_dev[val_rel], pred)))
        rows.append((k, float(np.mean(fold_rmse))))
    best_k, best_mean = min(rows, key=lambda item: item[1])
    return best_k, best_mean


def fit_plsr(X_dev, y_dev, best_k):
    model = PLSRegression(n_components=int(best_k), scale=False, max_iter=1000)
    model.fit(X_dev, y_dev)
    return model


def select_rfr_grouped_cv(X_dev, y_dev, inner_splits):
    rows = []
    for params in RFR_CANDIDATES:
        fold_rmse = []
        for inner_fold, (train_rel, val_rel) in enumerate(inner_splits, start=1):
            model = RandomForestRegressor(
                random_state=SEED + inner_fold, n_jobs=-1, **params
            )
            model.fit(X_dev[train_rel], y_dev[train_rel])
            pred = model.predict(X_dev[val_rel])
            fold_rmse.append(np.sqrt(mean_squared_error(y_dev[val_rel], pred)))
        rows.append((params.copy(), float(np.mean(fold_rmse))))
    best_params, best_mean = min(rows, key=lambda item: item[1])
    return best_params, best_mean


def fit_rfr(X_dev, y_dev, params):
    model = RandomForestRegressor(
        random_state=SEED,
        n_jobs=-1,
        **params,
    )
    model.fit(X_dev, y_dev)
    return model


def select_svr_grouped_cv(X_dev, y_dev, inner_splits):
    rows = []
    for params in SVR_CANDIDATES:
        fold_rmse = []
        for train_rel, val_rel in inner_splits:
            model = SVR(kernel="rbf", cache_size=2000, **params)
            model.fit(X_dev[train_rel], y_dev[train_rel])
            pred = model.predict(X_dev[val_rel])
            fold_rmse.append(np.sqrt(mean_squared_error(y_dev[val_rel], pred)))
        rows.append((params.copy(), float(np.mean(fold_rmse))))
    best_params, best_mean = min(rows, key=lambda item: item[1])
    return best_params, best_mean


def fit_svr(X_dev, y_dev, params):
    model = SVR(kernel="rbf", cache_size=2000, **params)
    model.fit(X_dev, y_dev)
    return model


# ============================================================
# CNN
# ============================================================

class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Cys1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)

        flattened_size = (INPUT_LENGTH // 16) * 128
        self.fc1 = nn.Linear(flattened_size, 128)
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


def _make_cnn_components(device, epochs, steps_per_epoch):
    model = Cys1DCNN().to(device)
    criterion = nn.SmoothL1Loss(beta=CNN_HUBER_BETA)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CNN_LEARNING_RATE,
        weight_decay=CNN_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CNN_LEARNING_RATE * 10,
        epochs=max(1, epochs),
        steps_per_epoch=max(1, steps_per_epoch),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e4,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(CNN_USE_AMP and device.type == "cuda")
    )
    return model, criterion, optimizer, scheduler, scaler


def _cnn_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=(CNN_USE_AMP and device.type == "cuda")
        ):
            pred = model(xb)
            loss = criterion(pred, yb)

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total += float(loss.item()) * len(xb)

    return total / len(loader.dataset)


def fit_cnn_record_history(
    X_train,
    y_train,
    X_val,
    y_val,
    fold_seed,
):
    """Fit one temporary inner-fold CNN and return losses for all epochs."""
    set_seed(fold_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        SpectraDataset(X_train, y_train),
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        SpectraDataset(X_val, y_val),
        batch_size=CNN_BATCH_SIZE,
        shuffle=False,
    )

    model, criterion, optimizer, scheduler, grad_scaler = _make_cnn_components(
        device, CNN_MAX_EPOCHS, len(train_loader)
    )

    history = []
    for epoch in range(1, CNN_MAX_EPOCHS + 1):
        train_loss = _cnn_epoch(
            model, train_loader, criterion, device,
            optimizer, scheduler, grad_scaler
        )
        with torch.no_grad():
            val_loss = _cnn_epoch(model, val_loader, criterion, device)
        history.append((epoch, train_loss, val_loss))

    return pd.DataFrame(
        history,
        columns=["Epoch", "Train_Loss", "Validation_Loss"],
    )


def select_cnn_epoch_by_grouped_cv(X_dev, y_dev, metadata_dev, fold_seed, output_dir):
    """Select the epoch with minimum mean validation loss over five cultivar folds."""
    groups = metadata_dev["cultivar"].to_numpy()
    splitter = GroupKFold(n_splits=CNN_INNER_FOLDS)
    histories = []

    for inner_fold, (train_rel, val_rel) in enumerate(
        splitter.split(X_dev, y_dev, groups=groups), start=1
    ):
        train_cultivars = set(groups[train_rel])
        val_cultivars = set(groups[val_rel])
        assert train_cultivars.isdisjoint(val_cultivars)

        history = fit_cnn_record_history(
            X_train=X_dev[train_rel],
            y_train=y_dev[train_rel],
            X_val=X_dev[val_rel],
            y_val=y_dev[val_rel],
            fold_seed=fold_seed + inner_fold,
        )
        history.insert(0, "Inner_Fold", inner_fold)
        history["Training_Cultivars"] = ";".join(sorted(train_cultivars))
        history["Validation_Cultivars"] = ";".join(sorted(val_cultivars))
        histories.append(history)

    all_history = pd.concat(histories, ignore_index=True)
    mean_history = (
        all_history.groupby("Epoch", as_index=False)
        .agg(
            Mean_Train_Loss=("Train_Loss", "mean"),
            Mean_Validation_Loss=("Validation_Loss", "mean"),
            SD_Validation_Loss=("Validation_Loss", "std"),
        )
    )
    selected_row = mean_history.loc[mean_history["Mean_Validation_Loss"].idxmin()]
    selected_epoch = int(selected_row["Epoch"])
    selected_mean_val = float(selected_row["Mean_Validation_Loss"])

    all_history.to_csv(output_dir / "cnn_inner_fold_histories.csv", index=False)
    mean_history.to_csv(output_dir / "cnn_mean_validation_history.csv", index=False)
    return selected_epoch, selected_mean_val, all_history, mean_history


def fit_final_cnn_on_all_development(X_dev, y_dev, selected_epoch, fold_seed, checkpoint_path):
    """Fit a fresh CNN on all 19 development cultivars for selected_epoch epochs.

    The OneCycleLR horizon remains CNN_MAX_EPOCHS so epoch k uses the same
    schedule position as epoch k in the temporary inner-fold CNNs.
    """
    set_seed(fold_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        SpectraDataset(X_dev, y_dev),
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
    )
    model, criterion, optimizer, scheduler, grad_scaler = _make_cnn_components(
        device, CNN_MAX_EPOCHS, len(train_loader)
    )

    history = []
    for epoch in range(1, selected_epoch + 1):
        train_loss = _cnn_epoch(
            model, train_loader, criterion, device,
            optimizer, scheduler, grad_scaler
        )
        history.append((epoch, train_loss))

    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    return model, device, pd.DataFrame(history, columns=["Epoch", "Train_Loss"])


def predict_cnn(model, device, X):
    loader = DataLoader(
        SpectraDataset(X, np.zeros(len(X), dtype=np.float32)),
        batch_size=CNN_BATCH_SIZE,
        shuffle=False,
    )

    predictions = []
    model.eval()

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            with torch.cuda.amp.autocast(
                enabled=(CNN_USE_AMP and device.type == "cuda")
            ):
                predictions.append(model(xb).squeeze(1).cpu().numpy())

    return np.concatenate(predictions)


# ============================================================
# One fold: common training and testing for all models
# ============================================================

def run_all_models_for_fold(
    fold_name,
    X_dev,
    y_dev,
    meta_dev,
    inner_splits,
    X_test,
    y_test,
    meta_test,
    fold_output_dir,
    fold_seed,
):
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    prediction_tables = []
    selection_rows = []

    # LR
    start = time.time()
    X_dev_lr = preprocess_for_model(X_dev, "LR")
    X_test_lr = preprocess_for_model(X_test, "LR")
    lr_model, lr_note = fit_lr(X_dev_lr, y_dev)
    lr_pred = lr_model.predict(X_test_lr)
    selection_rows.append({
        "Fold": fold_name, "Model": "LR",
        "Selected_Parameters": f"{lr_note}; preprocessing={OPTIMIZED_PREPROCESSING['LR']}",
        "Validation_RMSE": np.nan,
        "Runtime_Seconds": time.time() - start,
    })
    prediction_tables.append(
        make_prediction_table("LR", fold_name, y_test, lr_pred, meta_test)
    )

    # PLSR
    start = time.time()
    X_dev_plsr = preprocess_for_model(X_dev, "PLSR")
    X_test_plsr = preprocess_for_model(X_test, "PLSR")
    best_k, validation_rmse = select_plsr_grouped_cv(
        X_dev_plsr, y_dev, inner_splits
    )
    pls_model = fit_plsr(X_dev_plsr, y_dev, best_k)
    pls_pred = pls_model.predict(X_test_plsr).ravel()
    selection_rows.append({
        "Fold": fold_name, "Model": "PLSR",
        "Selected_Parameters": (
            f"n_components={best_k}; preprocessing={OPTIMIZED_PREPROCESSING['PLSR']}"
        ),
        "Validation_RMSE": validation_rmse,
        "Runtime_Seconds": time.time() - start,
    })
    prediction_tables.append(
        make_prediction_table("PLSR", fold_name, y_test, pls_pred, meta_test)
    )

    # RFR
    start = time.time()
    X_dev_rfr = preprocess_for_model(X_dev, "RFR")
    X_test_rfr = preprocess_for_model(X_test, "RFR")
    best_rf, validation_rmse = select_rfr_grouped_cv(
        X_dev_rfr, y_dev, inner_splits
    )
    rf_model = fit_rfr(X_dev_rfr, y_dev, best_rf)
    rf_pred = rf_model.predict(X_test_rfr)
    selection_rows.append({
        "Fold": fold_name, "Model": "RFR",
        "Selected_Parameters": (
            f"{best_rf}; preprocessing={OPTIMIZED_PREPROCESSING['RFR']}"
        ),
        "Validation_RMSE": validation_rmse,
        "Runtime_Seconds": time.time() - start,
    })
    prediction_tables.append(
        make_prediction_table("RFR", fold_name, y_test, rf_pred, meta_test)
    )

    # SVR
    start = time.time()
    X_dev_svr = preprocess_for_model(X_dev, "SVR")
    X_test_svr = preprocess_for_model(X_test, "SVR")
    best_svr, validation_rmse = select_svr_grouped_cv(
        X_dev_svr, y_dev, inner_splits
    )
    svr_model = fit_svr(X_dev_svr, y_dev, best_svr)
    svr_pred = svr_model.predict(X_test_svr)
    selection_rows.append({
        "Fold": fold_name, "Model": "SVR",
        "Selected_Parameters": (
            f"{best_svr}; preprocessing={OPTIMIZED_PREPROCESSING['SVR']}"
        ),
        "Validation_RMSE": validation_rmse,
        "Runtime_Seconds": time.time() - start,
    })
    prediction_tables.append(
        make_prediction_table("SVR", fold_name, y_test, svr_pred, meta_test)
    )

    # CNN
    start = time.time()
    X_dev_cnn = preprocess_for_model(X_dev, "CNN")
    X_test_cnn = preprocess_for_model(X_test, "CNN")
    checkpoint_path = fold_output_dir / "final_cnn_all_19_cultivars.pth"

    selected_epoch, mean_validation_loss, _, _ = select_cnn_epoch_by_grouped_cv(
        X_dev=X_dev_cnn,
        y_dev=y_dev,
        metadata_dev=meta_dev,
        fold_seed=fold_seed,
        output_dir=fold_output_dir,
    )

    cnn_model, cnn_device, final_history = fit_final_cnn_on_all_development(
        X_dev=X_dev_cnn,
        y_dev=y_dev,
        selected_epoch=selected_epoch,
        fold_seed=fold_seed + 1000,
        checkpoint_path=checkpoint_path,
    )
    final_history.to_csv(
        fold_output_dir / "cnn_final_all_19_training_history.csv",
        index=False,
    )

    cnn_pred = predict_cnn(
        model=cnn_model,
        device=cnn_device,
        X=X_test_cnn,
    )

    selection_rows.append({
        "Fold": fold_name,
        "Model": "CNN",
        "Selected_Parameters": (
            f"grouped_5fold_mean_best_epoch={selected_epoch}; "
            f"preprocessing={OPTIMIZED_PREPROCESSING['CNN']}"
        ),
        "Validation_RMSE": np.nan,
        "Validation_SmoothL1": mean_validation_loss,
        "Runtime_Seconds": time.time() - start,
    })
    prediction_tables.append(
        make_prediction_table("CNN", fold_name, y_test, cnn_pred, meta_test)
    )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    predictions.to_csv(fold_output_dir / "spectrum_predictions.csv", index=False)
    selections.to_csv(fold_output_dir / "model_selection.csv", index=False)
    return predictions, selections


def finalize_results(all_predictions, all_selections, unit_columns, output_dir, strategy_name):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions.to_csv(output_dir / "all_spectrum_predictions.csv", index=False)
    all_selections.to_csv(output_dir / "all_model_selection.csv", index=False)

    unit_predictions = aggregate_predictions(all_predictions, unit_columns)
    unit_predictions.to_csv(output_dir / "all_unit_mean_predictions.csv", index=False)

    # Fold-level metrics at both spectrum and independent-unit levels.
    fold_rows = []
    for (model_name, fold_name), sub in all_predictions.groupby(["Model", "Fold"]):
        spectrum_metrics = metric_dict(sub["True_Cys"], sub["Predicted_Cys"])
        unit_sub = unit_predictions[
            (unit_predictions["Model"] == model_name)
            & (unit_predictions["Fold"] == fold_name)
        ]
        unit_metrics = metric_dict(
            unit_sub["True_Cys"], unit_sub["Mean_Predicted_Cys"]
        )
        fold_rows.append({
            "Model": model_name,
            "Fold": fold_name,
            **{f"Spectrum_{k}": v for k, v in spectrum_metrics.items()},
            **{f"Unit_Mean_{k}": v for k, v in unit_metrics.items()},
        })

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    # Pooled out-of-fold comparison at independent-unit level.
    comparison_rows = []
    for model_name in MODEL_ORDER:
        sub = unit_predictions[unit_predictions["Model"] == model_name]
        metrics = metric_dict(sub["True_Cys"], sub["Mean_Predicted_Cys"])
        comparison_rows.append({"Model": model_name, **metrics})

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "model_comparison_unit_level.csv", index=False)

    # Pooled spectrum-level table as secondary output.
    spectrum_rows = []
    for model_name in MODEL_ORDER:
        sub = all_predictions[all_predictions["Model"] == model_name]
        spectrum_rows.append({
            "Model": model_name,
            **metric_dict(sub["True_Cys"], sub["Predicted_Cys"]),
        })
    pd.DataFrame(spectrum_rows).to_csv(
        output_dir / "model_comparison_spectrum_level.csv", index=False
    )

    plot_model_comparison(
        comparison,
        output_dir / "model_comparison_errors.png",
        f"{strategy_name}: independent-unit error comparison",
    )
    plot_predictions(
        unit_predictions,
        output_dir / "predicted_vs_reference_all_models.png",
        f"{strategy_name}: pooled out-of-fold predictions",
    )

    with open(output_dir / "README_results.txt", "w", encoding="utf-8") as handle:
        handle.write(
            f"Strategy: {strategy_name}\n"
            f"Data directory: {DATA_DIR}\n"
            "Input handling: fixed model-specific preprocessing from the 80/20 preprocessing study is applied.\n"
            f"Independent unit columns: {list(unit_columns)}\n"
            "Primary comparison: model_comparison_unit_level.csv\n"
            "Secondary comparison: model_comparison_spectrum_level.csv\n"
            "The outer test subset is not used for model selection.\n"
            "The CNN uses five grouped inner folds to select the epoch with the smallest mean validation loss.\n"
            "A fresh CNN is then fitted on all 19 development cultivars for the selected epoch count before outer testing.\n"
        )

    print("\nFinal independent-unit comparison")
    print(comparison.to_string(index=False))
    print(f"\nResults saved to: {output_dir.resolve()}")


def make_grouped_inner_folds(dev_idx, metadata):
    """Return five relative train/validation index pairs grouped by cultivar."""
    metadata_dev = metadata.iloc[dev_idx].reset_index(drop=True)
    groups = metadata_dev["cultivar"].to_numpy()
    splitter = GroupKFold(n_splits=CNN_INNER_FOLDS)
    dummy_X = np.zeros((len(metadata_dev), 1), dtype=np.float32)
    return list(splitter.split(dummy_X, groups=groups))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Zero-based outer-fold index. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    args = parser.parse_args()

    output_dir = Path("./Unified_Preprocessed_5Fold_InnerCV_LOCO_Comparison")
    X, y, metadata = load_all_spectra()

    cultivars = list(CULTIVAR_MEAN_CYS.keys())
    fold_index = args.fold_index
    if fold_index is None:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is None:
            raise RuntimeError(
                "Provide --fold-index or run this script as a Slurm array task."
            )
        fold_index = int(task_id)

    if not 0 <= fold_index < len(cultivars):
        raise ValueError(
            f"fold-index must be between 0 and {len(cultivars) - 1}; got {fold_index}."
        )

    fold_number = fold_index + 1
    test_cultivar = cultivars[fold_index]
    test_idx = np.flatnonzero(
        metadata["cultivar"].eq(test_cultivar).to_numpy()
    )
    dev_idx = np.flatnonzero(
        ~metadata["cultivar"].eq(test_cultivar).to_numpy()
    )

    meta_dev = metadata.iloc[dev_idx].reset_index(drop=True)
    inner_splits = make_grouped_inner_folds(dev_idx, metadata)

    test_cultivars = set(metadata.iloc[test_idx]["cultivar"])
    dev_cultivars = set(meta_dev["cultivar"])
    assert dev_cultivars.isdisjoint(test_cultivars)
    assert len(inner_splits) == CNN_INNER_FOLDS

    fold_name = f"Holdout_{test_cultivar}"
    print(
        f"Running outer fold {fold_number}/{len(cultivars)}: {fold_name}",
        flush=True,
    )
    run_all_models_for_fold(
        fold_name=fold_name,
        X_dev=X[dev_idx], y_dev=y[dev_idx],
        meta_dev=meta_dev,
        inner_splits=inner_splits,
        X_test=X[test_idx], y_test=y[test_idx],
        meta_test=metadata.iloc[test_idx].reset_index(drop=True),
        fold_output_dir=output_dir / fold_name,
        fold_seed=SEED + fold_number,
    )


if __name__ == "__main__":
    main()
