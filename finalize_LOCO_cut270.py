#!/usr/bin/env python3
"""
Finalize the 270 cm^-1 low-shift ablation LOCO analysis.

Reads the 20 per-cultivar output folders produced by
unified_preprocessed_5fold_innercv_cut270.py and calculates the primary
LOCO metrics at the independent cultivar level (N = 20).

No model fitting or model selection occurs here.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_ORDER = ["LR", "PLSR", "RFR", "SVR", "CNN"]

OUTPUT_DIR = Path("./Unified_Preprocessed_5Fold_InnerCV_LOCO_Cut270")
FULL_SPECTRUM_DIR = Path("./Unified_Preprocessed_5Fold_InnerCV_LOCO_Comparison")


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and not np.isclose(np.var(y_true), 0.0) else np.nan,
        "Bias": float(np.mean(y_pred - y_true)),
        "N": int(len(y_true)),
    }


def main():
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f"Output directory not found: {OUTPUT_DIR.resolve()}")

    prediction_files = sorted(OUTPUT_DIR.glob("Holdout_*/spectrum_predictions.csv"))
    selection_files = sorted(OUTPUT_DIR.glob("Holdout_*/model_selection.csv"))

    if len(prediction_files) != 20:
        raise RuntimeError(
            f"Expected 20 spectrum_predictions.csv files, found {len(prediction_files)}.\n"
            "The finalization job should run only after all 20 outer folds succeed."
        )

    all_predictions = pd.concat(
        [pd.read_csv(path) for path in prediction_files],
        ignore_index=True,
    )

    if selection_files:
        all_selections = pd.concat(
            [pd.read_csv(path) for path in selection_files],
            ignore_index=True,
        )
        all_selections.to_csv(OUTPUT_DIR / "all_model_selection.csv", index=False)

    required = {"Model", "Fold", "cultivar", "True_Cys", "Predicted_Cys"}
    missing = required.difference(all_predictions.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in prediction files: {sorted(missing)}")

    all_predictions.to_csv(OUTPUT_DIR / "all_spectrum_predictions.csv", index=False)

    # One independent prediction for each held-out cultivar and model.
    cultivar_predictions = (
        all_predictions
        .groupby(["Model", "Fold", "cultivar"], as_index=False)
        .agg(
            True_Cys=("True_Cys", "first"),
            Mean_Predicted_Cys=("Predicted_Cys", "mean"),
            SD_Predicted_Cys=("Predicted_Cys", "std"),
            N_Spectra=("Predicted_Cys", "size"),
        )
    )
    cultivar_predictions["SD_Predicted_Cys"] = cultivar_predictions["SD_Predicted_Cys"].fillna(0.0)
    cultivar_predictions["Residual"] = (
        cultivar_predictions["Mean_Predicted_Cys"] - cultivar_predictions["True_Cys"]
    )
    cultivar_predictions.to_csv(
        OUTPUT_DIR / "all_cultivar_mean_predictions.csv",
        index=False,
    )

    # Verify that every model contributes exactly 20 independent cultivar predictions.
    counts = cultivar_predictions.groupby("Model")["cultivar"].nunique()
    for model in MODEL_ORDER:
        n = int(counts.get(model, 0))
        if n != 20:
            raise RuntimeError(
                f"{model}: expected 20 held-out cultivar predictions, found {n}."
            )

    rows = []
    for model in MODEL_ORDER:
        sub = cultivar_predictions[cultivar_predictions["Model"] == model]
        metrics = metric_dict(sub["True_Cys"], sub["Mean_Predicted_Cys"])
        rows.append({"Model": model, **metrics})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison_cultivar_level.csv",
        index=False,
    )
    # Duplicate under the same naming convention used in the original workflow.
    comparison.to_csv(
        OUTPUT_DIR / "model_comparison_unit_level.csv",
        index=False,
    )

    spectrum_rows = []
    for model in MODEL_ORDER:
        sub = all_predictions[all_predictions["Model"] == model]
        spectrum_rows.append({
            "Model": model,
            **metric_dict(sub["True_Cys"], sub["Predicted_Cys"]),
        })
    pd.DataFrame(spectrum_rows).to_csv(
        OUTPUT_DIR / "model_comparison_spectrum_level.csv",
        index=False,
    )

    # Cultivar-level prediction plot.
    fig, ax = plt.subplots(figsize=(7, 7))
    for model in MODEL_ORDER:
        sub = cultivar_predictions[cultivar_predictions["Model"] == model]
        ax.scatter(
            sub["True_Cys"],
            sub["Mean_Predicted_Cys"],
            alpha=0.75,
            label=model,
        )
    lo = float(min(
        cultivar_predictions["True_Cys"].min(),
        cultivar_predictions["Mean_Predicted_Cys"].min(),
    ))
    hi = float(max(
        cultivar_predictions["True_Cys"].max(),
        cultivar_predictions["Mean_Predicted_Cys"].max(),
    ))
    ax.plot([lo, hi], [lo, hi], linestyle="--", label="Identity")
    ax.set_xlabel("Reference cysteine (g/100 g)")
    ax.set_ylabel("Mean predicted cysteine (g/100 g)")
    ax.set_title("LOCO after removing Raman shifts <270 cm$^{-1}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "predicted_vs_reference_all_models_cultivar_level.png",
        dpi=200,
    )
    plt.close(fig)

    # MAE/RMSE comparison plot.
    ordered = comparison.set_index("Model").reindex(MODEL_ORDER).reset_index()
    x = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ordered["MAE"], width, label="MAE")
    ax.bar(x + width / 2, ordered["RMSE"], width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["Model"])
    ax.set_ylabel("Error (g/100 g)")
    ax.set_title("LOCO cultivar-level error after removing <270 cm$^{-1}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison_errors_cultivar_level.png", dpi=200)
    plt.close(fig)

    # Optional direct comparison with existing full-spectrum aggregate results.
    full_path = FULL_SPECTRUM_DIR / "model_comparison_unit_level.csv"
    if full_path.is_file():
        full = pd.read_csv(full_path)
        merged = full.merge(
            comparison,
            on="Model",
            suffixes=("_Full", "_Cut270"),
        )
        for metric in ["MAE", "RMSE", "R2", "Bias"]:
            if f"{metric}_Full" in merged and f"{metric}_Cut270" in merged:
                merged[f"Delta_{metric}_Cut270_minus_Full"] = (
                    merged[f"{metric}_Cut270"] - merged[f"{metric}_Full"]
                )
        merged.to_csv(
            OUTPUT_DIR / "cut270_vs_full_spectrum_comparison.csv",
            index=False,
        )

    with open(OUTPUT_DIR / "README_finalization.txt", "w", encoding="utf-8") as f:
        f.write(
            "270 cm^-1 low-shift ablation LOCO finalization\n"
            "================================================\n"
            "Primary statistical unit: held-out cultivar.\n"
            "One mean out-of-fold prediction is calculated for each of 20 cultivars.\n"
            "Primary metrics file: model_comparison_cultivar_level.csv\n"
            "Equivalent compatibility file: model_comparison_unit_level.csv\n"
            "Raman shifts below 270 cm^-1 were removed before the same fixed preprocessing and LOCO analysis.\n"
            "No model fitting or model selection is performed by this finalization script.\n"
        )

    print("\nFINAL CULTIVAR-LEVEL LOCO COMPARISON — CUT <270 cm^-1")
    print(comparison.to_string(index=False))
    print(f"\nN independent cultivars per model = 20")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")

    optional = OUTPUT_DIR / "cut270_vs_full_spectrum_comparison.csv"
    if optional.is_file():
        print(f"Direct full-spectrum comparison saved to: {optional.resolve()}")


if __name__ == "__main__":
    main()
