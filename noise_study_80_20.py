# ===== Corrected 1D-CNN SERS cysteine simulated scan-count sweep =====
#
# This version preserves the structure of the original noise study while
# addressing the reviewer comments:
#
# 1. Uses ../../CrossValidationData.
# 2. Uses one fixed cultivar-stratified split for every scan-count condition:
#       64% training, 16% internal validation, 20% final test.
# 3. Applies each simulated scan-count condition independently to training,
#    validation, and test spectra.
# 4. Selects the CNN checkpoint using internal validation loss only.
# 5. Uses the final test set only after checkpoint selection.
# 6. Reports cultivar-level RMSE, MAE, bias, and R².
# 7. Saves a manuscript-style summary table for nominal scan counts
#    64, 32, 16, 8, 4, 2, and 1.
#
# Important interpretation:
# The value 512 is retained only as a mathematical simulation-scaling reference,
# not as an experimentally acquired condition. The resulting rows represent
# nominal simulated scan-count noise conditions, not experimentally measured
# scan counts.

from __future__ import annotations

import gc
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

INPUT_LENGTH = 1496
DATA_DIR = Path("../../CrossValidationData")
OUTPUT_DIR = Path("./Corrected_Simulated_Scan_Count_Sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

OUTER_TEST_FRACTION = 0.20
INNER_VALIDATION_FRACTION_OF_DEVELOPMENT = 0.20

NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
HUBER_BETA = 0.02
USE_AMP = True

# Retained only as a simulation-scaling reference.
SIMULATION_REFERENCE_SCAN_COUNT = 512

NOMINAL_SCAN_COUNTS = [64, 32, 16, 8, 4, 2, 1]

# Number of noisy copies of each training spectrum.
NUM_TRAIN_AUGMENTS = 100

# Validation and test receive one independently simulated version per spectrum.
NUM_VALIDATION_REALIZATIONS = 1
NUM_TEST_REALIZATIONS = 5


# ============================================================
# Cultivar labels
# ============================================================

CULTIVAR_CYS_MAP: Dict[str, float] = {
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


# ============================================================
# Reproducibility
# ============================================================

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
# Data loading
# ============================================================

def find_cultivar_from_filename(filename: str) -> str | None:
    for cultivar in sorted(CULTIVAR_CYS_MAP, key=len, reverse=True):
        if filename.startswith(cultivar):
            return cultivar
    return None


def parse_metadata(filename: str) -> Tuple[str, str, str, int]:
    cultivar = find_cultivar_from_filename(filename)
    if cultivar is None:
        raise ValueError(f"Cannot identify cultivar in filename: {filename}")

    location_match = re.search(r"_(001|002|003|01|02|03)_", filename)
    spot_match = re.search(r"_(R[123])_(\d+)\.npy$", filename)

    location = location_match.group(1) if location_match else "Unknown"
    if location in {"001", "002", "003"}:
        location = location[-2:]

    spot = spot_match.group(1) if spot_match else "Unknown"
    replicate = int(spot_match.group(2)) if spot_match else -1

    return cultivar, location, spot, replicate


def standardize_array_shape(arr: np.ndarray, filename: str) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected array shape {arr.shape} in {filename}")

    if arr.shape[1] != INPUT_LENGTH:
        raise ValueError(
            f"{filename}: spectrum length {arr.shape[1]} != {INPUT_LENGTH}"
        )

    return arr.astype(np.float32, copy=False)


def load_raw_data() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    file_paths = sorted(DATA_DIR.glob("*.npy"))
    if not file_paths:
        raise RuntimeError(f"No .npy files found in {DATA_DIR.resolve()}")

    X_parts = []
    y_parts = []
    metadata_rows = []

    for file_path in file_paths:
        cultivar, location, spot, replicate = parse_metadata(file_path.name)
        arr = standardize_array_shape(np.load(file_path), file_path.name)

        X_parts.append(arr)
        y_parts.append(
            np.full(arr.shape[0], CULTIVAR_CYS_MAP[cultivar], dtype=np.float32)
        )

        for array_index in range(arr.shape[0]):
            metadata_rows.append({
                "filename": file_path.name,
                "cultivar": cultivar,
                "location": location,
                "spot": spot,
                "replicate": replicate,
                "array_index": array_index,
            })

    X = np.vstack(X_parts).astype(np.float32, copy=False)
    y = np.concatenate(y_parts).astype(np.float32, copy=False)
    metadata = pd.DataFrame(metadata_rows)

    if not (len(X) == len(y) == len(metadata)):
        raise RuntimeError("X, y, and metadata lengths are inconsistent.")

    return X, y, metadata


# ============================================================
# Noise simulation
# ============================================================

def robust_baseline_noise(signal: np.ndarray) -> float:
    """
    Estimate high-frequency noise using a moving-average residual and MAD.
    This is used only for noise estimation, not as preprocessing.
    """
    window = 21

    if window >= len(signal):
        window = max(3, (len(signal) // 2) * 2 - 1)

    smooth = np.convolve(
        signal.astype(np.float64),
        np.ones(window, dtype=np.float64) / window,
        mode="same",
    )
    residual = signal.astype(np.float64) - smooth
    mad = np.median(np.abs(residual - np.median(residual)))

    return float(max(1.4826 * mad, np.finfo(np.float32).eps))


def compute_baseline_noise_from_training(X_train: np.ndarray) -> float:
    values = [robust_baseline_noise(spectrum) for spectrum in X_train]
    return float(np.mean(values))


def nominal_noise_std(
    baseline_noise: float,
    nominal_scan_count: int,
) -> float:
    """
    Original study scaling:
        sigma_N = sigma_baseline * sqrt(512 / N)

    The 512 value is a simulation reference only.
    """
    if nominal_scan_count <= 0:
        raise ValueError("nominal_scan_count must be positive")

    return float(
        baseline_noise
        * np.sqrt(SIMULATION_REFERENCE_SCAN_COUNT / nominal_scan_count)
    )


def add_noise_copies(
    X: np.ndarray,
    noise_std: float,
    copies: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if copies < 1:
        raise ValueError("copies must be >= 1")

    blocks = []

    for _ in range(copies):
        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=X.shape,
        ).astype(np.float32)

        blocks.append((X + noise).astype(np.float32, copy=False))

    return np.vstack(blocks)


# ============================================================
# CNN
# ============================================================

class Cys1DCNN(nn.Module):
    def __init__(self, input_length: int = 1496):
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

        self.flattened_size = (input_length // 16) * 128
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)


class SpectraDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(
            np.asarray(X, dtype=np.float32)
        ).unsqueeze(1)
        self.y = torch.from_numpy(
            np.asarray(y, dtype=np.float32)
        ).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        SpectraDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Training
# ============================================================

def train_with_validation_checkpoint(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    condition_seed: int,
):
    set_seed(condition_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cys1DCNN(INPUT_LENGTH).to(device)

    criterion = nn.SmoothL1Loss(beta=HUBER_BETA)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    validation_loader = make_loader(
        X_validation,
        y_validation,
        shuffle=False,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 10.0,
        epochs=NUM_EPOCHS,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e4,
    )

    amp_enabled = bool(USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=amp_enabled,
    )

    best_state = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    training_losses = []
    validation_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_training_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_training_loss += loss.item() * X_batch.size(0)

        epoch_training_loss = (
            running_training_loss / len(train_loader.dataset)
        )
        training_losses.append(epoch_training_loss)

        model.eval()
        running_validation_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)

                running_validation_loss += loss.item() * X_batch.size(0)

        epoch_validation_loss = (
            running_validation_loss / len(validation_loader.dataset)
        )
        validation_losses.append(epoch_validation_loss)

        print(
            f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | "
            f"Train loss: {epoch_training_loss:.6f} | "
            f"Validation loss: {epoch_validation_loss:.6f}"
        )

        if epoch_validation_loss < best_validation_loss:
            best_validation_loss = epoch_validation_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}; "
                f"best epoch = {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("No validation-selected checkpoint was created.")

    model.load_state_dict(best_state)
    model.eval()

    return (
        model,
        best_epoch,
        best_validation_loss,
        training_losses,
        validation_losses,
    )


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []

    model.eval()
    amp_enabled = bool(USE_AMP and device.type == "cuda")

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                output = model(X_batch)

            predictions.append(output.detach().cpu().numpy().ravel())
            targets.append(y_batch.numpy().ravel())

    return np.concatenate(predictions), np.concatenate(targets)


# ============================================================
# Metrics
# ============================================================

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def cultivar_level_metrics(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    frame = metadata.copy().reset_index(drop=True)
    frame["True_Cys"] = y_true
    frame["Predicted_Cys"] = y_pred

    cultivar_table = (
        frame.groupby("cultivar", as_index=False)
        .agg(
            True_Cys=("True_Cys", "first"),
            Mean_Predicted_Cys=("Predicted_Cys", "mean"),
            SD_Predicted_Cys=("Predicted_Cys", "std"),
            N_Spectra=("Predicted_Cys", "size"),
        )
    )

    cultivar_table["Residual"] = (
        cultivar_table["Mean_Predicted_Cys"]
        - cultivar_table["True_Cys"]
    )

    metrics = regression_metrics(
        cultivar_table["True_Cys"].to_numpy(),
        cultivar_table["Mean_Predicted_Cys"].to_numpy(),
    )

    return cultivar_table, metrics


# ============================================================
# Fixed split used for every nominal scan count
# ============================================================

X_all, y_all, metadata_all = load_raw_data()
all_indices = np.arange(len(X_all))

development_indices, test_indices = train_test_split(
    all_indices,
    test_size=OUTER_TEST_FRACTION,
    random_state=SEED,
    stratify=metadata_all["cultivar"],
)

training_indices, validation_indices = train_test_split(
    development_indices,
    test_size=INNER_VALIDATION_FRACTION_OF_DEVELOPMENT,
    random_state=SEED + 1,
    stratify=metadata_all.iloc[development_indices]["cultivar"],
)

X_train_raw = X_all[training_indices]
y_train_raw = y_all[training_indices]
metadata_train = metadata_all.iloc[training_indices].reset_index(drop=True)

X_validation_raw = X_all[validation_indices]
y_validation_raw = y_all[validation_indices]
metadata_validation = metadata_all.iloc[validation_indices].reset_index(drop=True)

X_test_raw = X_all[test_indices]
y_test_raw = y_all[test_indices]
metadata_test = metadata_all.iloc[test_indices].reset_index(drop=True)

print("\nFixed split")
print(f"Training spectra:   {len(X_train_raw)}")
print(f"Validation spectra: {len(X_validation_raw)}")
print(f"Test spectra:       {len(X_test_raw)}")

split_assignment = pd.concat(
    [
        metadata_train.assign(split="training"),
        metadata_validation.assign(split="internal_validation"),
        metadata_test.assign(split="final_test"),
    ],
    ignore_index=True,
)
split_assignment.to_csv(
    OUTPUT_DIR / "fixed_data_split.csv",
    index=False,
)


# Baseline noise is estimated once from the raw training subset only.
baseline_noise = compute_baseline_noise_from_training(X_train_raw)

print(
    f"\nBaseline noise estimated from training spectra only: "
    f"{baseline_noise:.8f}"
)


# ============================================================
# Scan-count sweep
# ============================================================

summary_rows = []
all_test_predictions = []

for scan_index, scan_count in enumerate(NOMINAL_SCAN_COUNTS):
    print("\n" + "=" * 84)
    print(
        f"Nominal simulated scan count: {scan_count} "
        f"({scan_index + 1}/{len(NOMINAL_SCAN_COUNTS)})"
    )
    print("=" * 84)

    condition_seed = SEED + 1000 * (scan_index + 1)
    noise_std = nominal_noise_std(baseline_noise, scan_count)

    print(f"Simulation noise SD: {noise_std:.8f}")

    training_rng = np.random.default_rng(condition_seed + 11)
    validation_rng = np.random.default_rng(condition_seed + 22)

    X_train_simulated = add_noise_copies(
        X_train_raw,
        noise_std=noise_std,
        copies=NUM_TRAIN_AUGMENTS,
        rng=training_rng,
    )
    y_train_simulated = np.tile(
        y_train_raw,
        NUM_TRAIN_AUGMENTS,
    )

    X_validation_simulated = add_noise_copies(
        X_validation_raw,
        noise_std=noise_std,
        copies=NUM_VALIDATION_REALIZATIONS,
        rng=validation_rng,
    )
    y_validation_simulated = np.tile(
        y_validation_raw,
        NUM_VALIDATION_REALIZATIONS,
    )

    condition_directory = (
        OUTPUT_DIR / f"nominal_scans_{scan_count}"
    )
    condition_directory.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    (
        model,
        best_epoch,
        best_validation_loss,
        training_losses,
        validation_losses,
    ) = train_with_validation_checkpoint(
        X_train_simulated,
        y_train_simulated,
        X_validation_simulated,
        y_validation_simulated,
        condition_seed=condition_seed,
    )

    torch.save(
        model.state_dict(),
        condition_directory / "best_validation_checkpoint.pth",
    )

    plt.figure(figsize=(7, 5))
    plt.plot(training_losses, label="Training")
    plt.plot(validation_losses, label="Internal validation")
    plt.axvline(
        best_epoch - 1,
        linestyle="--",
        label=f"Best epoch = {best_epoch}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 loss")
    plt.title(
        f"Loss curve: nominal simulated scans = {scan_count}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        condition_directory / "loss_curve.png",
        dpi=200,
    )
    plt.close()

    test_realization_metrics = []
    test_realization_cultivar_metrics = []

    for realization_index in range(NUM_TEST_REALIZATIONS):
        test_rng = np.random.default_rng(
            condition_seed + 100 + realization_index
        )

        X_test_simulated = add_noise_copies(
            X_test_raw,
            noise_std=noise_std,
            copies=1,
            rng=test_rng,
        )

        test_loader = make_loader(
            X_test_simulated,
            y_test_raw,
            shuffle=False,
        )

        device = next(model.parameters()).device
        y_pred, y_true = predict(model, test_loader, device)

        spectrum_metrics = regression_metrics(y_true, y_pred)
        cultivar_table, cultivar_metrics = cultivar_level_metrics(
            metadata_test,
            y_true,
            y_pred,
        )

        spectrum_metrics["Realization"] = realization_index + 1
        cultivar_metrics["Realization"] = realization_index + 1

        test_realization_metrics.append(spectrum_metrics)
        test_realization_cultivar_metrics.append(cultivar_metrics)

        spectrum_prediction_table = metadata_test.copy()
        spectrum_prediction_table["Nominal_Scan_Count"] = scan_count
        spectrum_prediction_table["Realization"] = realization_index + 1
        spectrum_prediction_table["True_Cys"] = y_true
        spectrum_prediction_table["Predicted_Cys"] = y_pred
        spectrum_prediction_table["Residual"] = y_pred - y_true

        all_test_predictions.append(spectrum_prediction_table)

        cultivar_table["Nominal_Scan_Count"] = scan_count
        cultivar_table["Realization"] = realization_index + 1
        cultivar_table.to_csv(
            condition_directory
            / f"cultivar_predictions_realization_{realization_index + 1}.csv",
            index=False,
        )

    spectrum_metrics_df = pd.DataFrame(test_realization_metrics)
    cultivar_metrics_df = pd.DataFrame(
        test_realization_cultivar_metrics
    )

    spectrum_metrics_df.to_csv(
        condition_directory
        / "spectrum_metrics_by_test_realization.csv",
        index=False,
    )
    cultivar_metrics_df.to_csv(
        condition_directory
        / "cultivar_metrics_by_test_realization.csv",
        index=False,
    )

    summary_row = {
        "Number_of_Scans": scan_count,
        "Noise_SD": noise_std,
        "Best_Epoch": best_epoch,
        "Best_Validation_Loss": best_validation_loss,
        "RMSE": cultivar_metrics_df["RMSE"].mean(),
        "RMSE_SD": cultivar_metrics_df["RMSE"].std(ddof=1),
        "MAE": cultivar_metrics_df["MAE"].mean(),
        "MAE_SD": cultivar_metrics_df["MAE"].std(ddof=1),
        "R2": cultivar_metrics_df["R2"].mean(),
        "R2_SD": cultivar_metrics_df["R2"].std(ddof=1),
        "Bias": cultivar_metrics_df["Bias"].mean(),
        "Bias_SD": cultivar_metrics_df["Bias"].std(ddof=1),
        "Runtime_Seconds": time.time() - start_time,
    }
    summary_rows.append(summary_row)

    print("\nCultivar-level final test results")
    print(
        f"RMSE: {summary_row['RMSE']:.6f} "
        f"± {summary_row['RMSE_SD']:.6f}"
    )
    print(
        f"MAE:  {summary_row['MAE']:.6f} "
        f"± {summary_row['MAE_SD']:.6f}"
    )
    print(
        f"R²:   {summary_row['R2']:.6f} "
        f"± {summary_row['R2_SD']:.6f}"
    )
    print(
        f"Bias: {summary_row['Bias']:.6f} "
        f"± {summary_row['Bias_SD']:.6f}"
    )

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Save final manuscript-style table and plots
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_DIR / "scan_count_summary_with_uncertainty.csv",
    index=False,
)

manuscript_table = summary_df[
    ["Number_of_Scans", "RMSE", "MAE", "R2"]
].copy()

manuscript_table.columns = [
    "Number of Scans",
    "RMSE",
    "MAE",
    "R²",
]

manuscript_table.to_csv(
    OUTPUT_DIR / "manuscript_style_scan_count_table.csv",
    index=False,
)

pd.concat(
    all_test_predictions,
    ignore_index=True,
).to_csv(
    OUTPUT_DIR / "all_final_test_predictions.csv",
    index=False,
)

plot_df = summary_df.sort_values("Number_of_Scans")

plt.figure(figsize=(7, 5))
plt.errorbar(
    plot_df["Number_of_Scans"],
    plot_df["RMSE"],
    yerr=plot_df["RMSE_SD"],
    marker="o",
    capsize=4,
)
plt.xlabel("Nominal simulated scan count")
plt.ylabel("Cultivar-level RMSE (g/100 g)")
plt.title("Controlled simulated scan-count analysis")
plt.xscale("log", base=2)
plt.xticks(
    NOMINAL_SCAN_COUNTS,
    labels=[str(value) for value in NOMINAL_SCAN_COUNTS],
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "rmse_vs_nominal_scan_count.png",
    dpi=200,
)
plt.close()

plt.figure(figsize=(7, 5))
plt.errorbar(
    plot_df["Number_of_Scans"],
    plot_df["R2"],
    yerr=plot_df["R2_SD"],
    marker="o",
    capsize=4,
)
plt.xlabel("Nominal simulated scan count")
plt.ylabel("Cultivar-level R²")
plt.title("Controlled simulated scan-count analysis")
plt.xscale("log", base=2)
plt.xticks(
    NOMINAL_SCAN_COUNTS,
    labels=[str(value) for value in NOMINAL_SCAN_COUNTS],
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "r2_vs_nominal_scan_count.png",
    dpi=200,
)
plt.close()

print("\nCompleted corrected simulated scan-count sweep.")
print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
print(
    "Main manuscript table: "
    "manuscript_style_scan_count_table.csv"
)
