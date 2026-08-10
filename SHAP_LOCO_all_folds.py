#!/usr/bin/env python3
"""Aggregate SHAP across all 20 corrected/preprocessed LOCO CNN models.

Primary interpretation rule
---------------------------
For each outer LOCO fold:
  1. background spectra come only from the 19 development cultivars;
  2. explained spectra come only from that fold's held-out cultivar;
  3. the saved final CNN fitted on all 19 development cultivars is loaded;
  4. mean absolute SHAP is calculated within the fold.
The final global importance curve is the equal-weight mean of the 20 fold-level
importance curves. Thus each cultivar contributes one independent fold-level
importance profile, rather than allowing technical spectra from one cultivar to
receive disproportionate weight.
"""

from pathlib import Path
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from unified_preprocessed_5fold_innercv_final_array import (
    CULTIVAR_MEAN_CYS,
    Cys1DCNN,
    load_all_spectra,
    preprocess_for_model,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "Unified_Preprocessed_5Fold_InnerCV_LOCO_Comparison"
AXIS_FILE = SCRIPT_DIR / "wavenumbers_1496.npy"
OUT_DIR = RESULT_DIR / "SHAP_All_LOCO_Folds"
SEED = 42
N_BACKGROUND_PER_FOLD = 120
N_EXPLAIN_PER_FOLD = 120
TOP_K = 20

# Location of the within-cultivar SHAP output for optional final Figure 2 assembly.
WITHIN_SHAP_DIR = (
    SCRIPT_DIR.parent / "Finalized-80-20Within-populationSplit" /
    "Unified_Raw_80_20_Comparison" / "SHAP_Within_Cultivar"
)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_shap_array(values):
    if isinstance(values, list): values = values[0]
    if isinstance(values, torch.Tensor): values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[1] == 1: values = values[:, 0, :]
    elif values.ndim == 3 and values.shape[2] == 1: values = values[:, :, 0]
    elif values.ndim != 2: values = values.reshape(values.shape[0], -1)
    return values.astype(np.float32, copy=False)


def sample(idx, n, rng):
    idx = np.asarray(idx)
    return rng.choice(idx, size=min(int(n), len(idx)), replace=False)


def save_fold(axis, cultivar, X_exp, shap_values, preds, metadata_exp, fold_out):
    fold_out.mkdir(parents=True, exist_ok=True)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    np.save(fold_out / "shap_values.npy", shap_values)
    np.save(fold_out / "X_explained.npy", X_exp)
    np.save(fold_out / "predictions.npy", preds)
    metadata_exp.to_csv(fold_out / "explained_spectra_metadata.csv", index=False)
    pd.DataFrame({"wavenumber_cm^-1": axis, "mean_abs_shap": mean_abs}).to_csv(
        fold_out / "shap_global_importance.csv", index=False
    )
    return mean_abs


def make_combined_figure(loco_shap, loco_X, axis, out_path):
    """Create the requested side-by-side Figure 2 if within-cultivar arrays exist."""
    within_sv = WITHIN_SHAP_DIR / "shap_values.npy"
    within_X = WITHIN_SHAP_DIR / "X_explained.npy"
    within_axis = WITHIN_SHAP_DIR / "wavenumbers_used.npy"
    if not (within_sv.is_file() and within_X.is_file() and within_axis.is_file()):
        print("Within-cultivar SHAP arrays not found yet; skipping combined Figure 2.")
        return
    sv_left = np.load(within_sv)
    X_left = np.load(within_X)
    ax_left = np.load(within_axis)
    names_left = np.array([f"f_{int(round(v))} cm^-1" for v in ax_left])
    names_right = np.array([f"f_{int(round(v))} cm^-1" for v in axis])

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.sca(ax1)
    shap.summary_plot(sv_left, X_left, feature_names=names_left, max_display=TOP_K,
                      show=False, plot_size=None)
    ax1.set_title("1D-CNN - Within cultivar")

    ax2 = fig.add_subplot(1, 2, 2)
    plt.sca(ax2)
    shap.summary_plot(loco_shap, loco_X, feature_names=names_right, max_display=TOP_K,
                      show=False, plot_size=None)
    ax2.set_title("1D-CNN - LOCO (all folds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined Figure 2 saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-background", type=int, default=N_BACKGROUND_PER_FOLD)
    parser.add_argument("--n-explain", type=int, default=N_EXPLAIN_PER_FOLD)
    args = parser.parse_args()

    set_seed(SEED)
    if not AXIS_FILE.is_file():
        raise FileNotFoundError(f"Missing {AXIS_FILE}; copy wavenumbers_1496.npy beside this script.")
    axis = np.load(AXIS_FILE, allow_pickle=False).reshape(-1).astype(float)
    X_raw, _, metadata = load_all_spectra()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_fold_dir = OUT_DIR / "per_fold"
    per_fold_dir.mkdir(exist_ok=True)

    fold_importance = []
    all_shap = []
    all_X = []
    all_meta = []
    fold_rows = []

    for fold_number, cultivar in enumerate(CULTIVAR_MEAN_CYS.keys(), start=1):
        print(f"[{fold_number:02d}/20] {cultivar}", flush=True)
        fold_dir = RESULT_DIR / f"Holdout_{cultivar}"
        checkpoint = fold_dir / "final_cnn_all_19_cultivars.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

        is_test = metadata["cultivar"].eq(cultivar).to_numpy()
        dev_idx = np.flatnonzero(~is_test)
        test_idx = np.flatnonzero(is_test)
        # Different deterministic seed per fold, while preserving reproducibility.
        rng = np.random.default_rng(SEED + fold_number)
        bg_idx = sample(dev_idx, args.n_background, rng)
        exp_idx = sample(test_idx, args.n_explain, rng)

        X_bg = preprocess_for_model(X_raw[bg_idx], "CNN")
        X_exp = preprocess_for_model(X_raw[exp_idx], "CNN")

        model = Cys1DCNN().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        bg_t = torch.as_tensor(X_bg, dtype=torch.float32, device=device).unsqueeze(1)
        exp_t = torch.as_tensor(X_exp, dtype=torch.float32, device=device).unsqueeze(1)
        explainer = shap.GradientExplainer(model, bg_t)
        sv = normalize_shap_array(explainer.shap_values(exp_t))
        with torch.no_grad():
            preds = model(exp_t).detach().cpu().numpy().reshape(-1)

        meta_exp = metadata.iloc[exp_idx].reset_index(drop=True).copy()
        meta_exp["LOCO_fold"] = cultivar
        mean_abs = save_fold(axis, cultivar, X_exp, sv, preds, meta_exp, per_fold_dir / cultivar)
        fold_importance.append(mean_abs)
        all_shap.append(sv)
        all_X.append(X_exp)
        all_meta.append(meta_exp)
        fold_rows.append({
            "cultivar": cultivar,
            "checkpoint": str(checkpoint.resolve()),
            "n_background": len(X_bg),
            "n_explained": len(X_exp),
        })

        del explainer, model, bg_t, exp_t
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    fold_importance = np.vstack(fold_importance)  # (20, 1496)
    mean_importance = fold_importance.mean(axis=0)
    sd_importance = fold_importance.std(axis=0, ddof=1)
    all_shap = np.vstack(all_shap)
    all_X = np.vstack(all_X)
    all_meta = pd.concat(all_meta, ignore_index=True)

    top_idx = np.argsort(mean_importance)[::-1][:TOP_K]
    # Number of folds in which each feature appears among that fold's top 20.
    top_counts = np.zeros(len(axis), dtype=int)
    for row in fold_importance:
        top_counts[np.argsort(row)[::-1][:TOP_K]] += 1

    pd.DataFrame({
        "wavenumber_cm^-1": axis,
        "mean_abs_shap_across_folds": mean_importance,
        "sd_abs_shap_across_folds": sd_importance,
        "top20_fold_count": top_counts,
    }).to_csv(OUT_DIR / "shap_all_folds_importance.csv", index=False)

    pd.DataFrame({
        "rank": np.arange(1, TOP_K + 1),
        "feature_index": top_idx,
        "wavenumber_cm^-1": axis[top_idx],
        "mean_abs_shap_across_folds": mean_importance[top_idx],
        "sd_abs_shap_across_folds": sd_importance[top_idx],
        "top20_fold_count": top_counts[top_idx],
    }).to_csv(OUT_DIR / "shap_all_folds_top_features.csv", index=False)

    pd.DataFrame(fold_rows).to_csv(OUT_DIR / "fold_analysis_metadata.csv", index=False)
    all_meta.to_csv(OUT_DIR / "all_explained_spectra_metadata.csv", index=False)
    np.save(OUT_DIR / "all_fold_shap_values.npy", all_shap)
    np.save(OUT_DIR / "all_fold_X_explained.npy", all_X)
    np.save(OUT_DIR / "fold_mean_abs_shap.npy", fold_importance)
    np.save(OUT_DIR / "wavenumbers_used.npy", axis)

    feature_names = np.array([f"f_{int(round(v))} cm^-1" for v in axis])
    plt.figure(figsize=(8, 6))
    shap.summary_plot(all_shap, all_X, feature_names=feature_names, max_display=TOP_K, show=False)
    plt.title("1D-CNN - LOCO (all 20 folds)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_beeswarm_all_folds.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(axis, mean_importance)
    plt.fill_between(axis, np.maximum(0, mean_importance - sd_importance),
                     mean_importance + sd_importance, alpha=0.2)
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.ylabel("Mean |SHAP| across LOCO folds")
    plt.title("LOCO CNN: equal-fold SHAP importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_all_folds_mean_curve.png", dpi=300)
    plt.close()

    # Fold x Raman-region consistency heatmap for global top features.
    heat = fold_importance[:, top_idx]
    plt.figure(figsize=(max(8, TOP_K * 0.42), 7))
    plt.imshow(heat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Fold mean |SHAP|")
    plt.yticks(np.arange(20), list(CULTIVAR_MEAN_CYS.keys()), fontsize=7)
    plt.xticks(np.arange(TOP_K), [f"{axis[i]:.0f}" for i in top_idx], rotation=45, ha="right")
    plt.xlabel("Raman shift (cm$^{-1}$)")
    plt.title("LOCO SHAP consistency across held-out cultivars")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_all_folds_top20_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    make_combined_figure(
        all_shap, all_X, axis,
        OUT_DIR / "Figure2_SHAP_within_vs_LOCO_all_folds.png"
    )

    print(f"Done. Outputs: {OUT_DIR.resolve()}")
    print(f"Equal-fold global SHAP used {fold_importance.shape[0]} LOCO models.")
    print(f"Concatenated beeswarm contains {len(all_shap)} spectra with equal requested N per fold.")


if __name__ == "__main__":
    main()
