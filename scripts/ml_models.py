"""
ml_models.py
------------
XGBoost regression pipeline with SHAP feature importance and
bootstrap uncertainty quantification for VIIRS electrification prediction.

Author: Bouchra Daddaoui
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Tuple

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[dict] = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 30,
    eval_fraction: float = 0.1,
    random_state: int = 42,
) -> xgb.XGBRegressor:
    """
    Train an XGBoost regressor with early stopping.

    Parameters
    ----------
    X_train              : Training features (n_samples, n_features).
    y_train              : Training target (n_samples,).
    params               : Optional hyperparameter dict (overrides defaults).
    n_estimators         : Maximum number of boosting rounds.
    early_stopping_rounds: Stop if validation loss doesn't improve for this many rounds.
    eval_fraction        : Fraction of training data used as internal eval set.
    random_state         : Random seed for reproducibility.

    Returns
    -------
    Fitted XGBRegressor.
    """
    default_params = {
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": random_state,
    }
    if params:
        default_params.update(params)

    n_eval = max(1, int(len(X_train) * eval_fraction))
    X_ev, y_ev = X_train[-n_eval:], y_train[-n_eval:]
    X_tr, y_tr = X_train[:-n_eval], y_train[:-n_eval]

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        **default_params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_ev, y_ev)],
        verbose=False,
    )
    return model


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Stratified K-fold cross-validation for XGBoost.

    Parameters
    ----------
    X         : Feature matrix.
    y         : Target vector.
    n_splits  : Number of CV folds.
    random_state : Seed.

    Returns
    -------
    dict with keys: rmse_scores, r2_scores, mean_rmse, std_rmse, mean_r2, std_r2.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_scores, r2_scores = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = train_xgboost(X_tr, y_tr, random_state=random_state)
        preds = model.predict(X_val)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        r2_scores.append(r2_score(y_val, preds))

    return {
        "rmse_scores": rmse_scores,
        "r2_scores": r2_scores,
        "mean_rmse": float(np.mean(rmse_scores)),
        "std_rmse": float(np.std(rmse_scores)),
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
    }


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    feature_names: list[str],
) -> Tuple[np.ndarray, shap.TreeExplainer]:
    """
    Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model         : Fitted XGBRegressor.
    X             : Feature matrix to explain.
    feature_names : List of feature names.

    Returns
    -------
    (shap_values, explainer) — SHAP values (n_samples, n_features) and explainer.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values   : (n_samples, n_features) array.
    X             : Feature matrix.
    feature_names : Feature labels.
    max_display   : Maximum features to show.
    save_path     : Optional PNG output path.

    Returns
    -------
    fig : Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=(8, 6),
    )
    fig = plt.gcf()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    feature: str,
    interaction_feature: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    SHAP dependence plot for a single feature.

    Parameters
    ----------
    shap_values         : (n_samples, n_features) array.
    X                   : Feature matrix.
    feature_names       : Feature labels.
    feature             : Feature to plot on x-axis.
    interaction_feature : Feature for colour encoding.
    save_path           : Optional save path.

    Returns
    -------
    fig : Matplotlib Figure.
    """
    feat_idx = feature_names.index(feature)
    int_idx = feature_names.index(interaction_feature) if interaction_feature else None

    fig, ax = plt.subplots(figsize=(6, 4))
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X,
        feature_names=feature_names,
        interaction_index=int_idx,
        ax=ax,
        show=False,
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def spatial_shap_map(
    gdf,
    shap_values: np.ndarray,
    feature_names: list[str],
    feature: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Choropleth map of SHAP values for a given feature.

    Each polygon is coloured by its SHAP contribution for `feature`,
    showing spatially where a variable drives predictions up or down.

    Parameters
    ----------
    gdf           : GeoDataFrame aligned with shap_values rows.
    shap_values   : (n_samples, n_features) array.
    feature_names : Feature labels.
    feature       : Feature to map.
    title         : Plot title.
    save_path     : Optional save path.

    Returns
    -------
    fig : Matplotlib Figure.
    """
    feat_idx = feature_names.index(feature)
    gdf = gdf.copy()
    gdf["shap_val"] = shap_values[:, feat_idx]

    vmax = np.abs(gdf["shap_val"]).quantile(0.97)
    cmap = plt.cm.RdBu_r

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    gdf.plot(
        column="shap_val",
        cmap=cmap,
        vmin=-vmax, vmax=vmax,
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        legend_kwds={"label": f"SHAP value ({feature})", "shrink": 0.7},
        ax=ax,
    )
    ax.set_title(title or f"Spatial SHAP: {feature}", fontsize=12, fontweight="bold")
    ax.set_axis_off()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Bootstrap uncertainty quantification
# ---------------------------------------------------------------------------

def bootstrap_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_bootstrap: int = 200,
    ci: float = 0.90,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap ensemble for prediction uncertainty quantification.

    Trains `n_bootstrap` XGBoost models on resampled training data and
    returns per-sample confidence intervals on the test set.

    Parameters
    ----------
    X_train     : Training features.
    y_train     : Training targets.
    X_test      : Test features to predict.
    n_bootstrap : Number of bootstrap replicates.
    ci          : Confidence level (e.g. 0.90 → 90% CI).
    random_state: Base seed (incremented per replicate).

    Returns
    -------
    dict with keys:
        mean_pred  : Mean prediction (n_test,)
        lower      : Lower CI bound (n_test,)
        upper      : Upper CI bound (n_test,)
        std_pred   : Prediction standard deviation (n_test,)
        all_preds  : All bootstrap predictions (n_bootstrap, n_test)
    """
    rng = np.random.default_rng(random_state)
    n = len(X_train)
    all_preds = np.zeros((n_bootstrap, len(X_test)))

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        model = train_xgboost(X_train[idx], y_train[idx], random_state=int(rng.integers(0, 1e6)))
        all_preds[b] = model.predict(X_test)

    alpha = (1.0 - ci) / 2.0
    return {
        "mean_pred": all_preds.mean(axis=0),
        "lower":     np.quantile(all_preds, alpha, axis=0),
        "upper":     np.quantile(all_preds, 1 - alpha, axis=0),
        "std_pred":  all_preds.std(axis=0),
        "all_preds": all_preds,
    }


def plot_uncertainty_map(
    gdf,
    uncertainty: np.ndarray,
    title: str = "Prediction Uncertainty (90% CI width)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Choropleth map of bootstrap prediction uncertainty (CI width).

    Parameters
    ----------
    gdf         : GeoDataFrame aligned with uncertainty array.
    uncertainty : Per-sample CI width (upper − lower).
    title       : Plot title.
    save_path   : Optional save path.

    Returns
    -------
    fig : Matplotlib Figure.
    """
    gdf = gdf.copy()
    gdf["uncertainty"] = uncertainty

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    gdf.plot(
        column="uncertainty",
        cmap="YlOrRd",
        linewidth=0.2,
        edgecolor="grey",
        legend=True,
        legend_kwds={"label": "CI width (NTL units)", "shrink": 0.7},
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_axis_off()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
