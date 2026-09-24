"""
train_weights.py — fit Stage 2's learned weights from the labelled URLs.

    python train_weights.py            # fit the rules-only model
    python train_weights.py --llm      # ALSO fit a model that includes Claude's
                                        # judgment as a feature (calls the API
                                        # once per labelled URL — costs money)

Fits a Lasso regression (L1-regularized linear model) mapping
credibility.extract_features() onto the expected scores in evaluate.py's
LABELLED_URLS, using leave-one-out cross-validation to pick the
regularization strength. LOOCV is the standard choice here rather than a
train/test split or k-fold: with only 24 examples, holding out even a 20%
test split leaves ~5 points to evaluate on, which is too noisy to trust —
LOOCV uses every point as validation exactly once while still fitting on
the other 23 each time.

Lasso's L1 penalty also performs feature selection: features that don't
carry real signal get shrunk to exactly zero. The printed report below is
the "which features actually matter" result the README points at
(Session 06) — put this table (or a version of it) in your Deliverable 2
report.

Writes learned_weights.json next to this file. credibility.py loads it
automatically the next time it's imported (see _load_learned_weights() /
_learned_score()); delete the file to fall back to the original hand-picked
constants from Stage 1.

HONESTY NOTE FOR YOUR REPORT: this script prints TWO MAE numbers. The
in-sample one is computed on the same 24 points the model was fit on and
will look artificially good — do not report it as your headline number.
The leave-one-out MAE refits the model 24 times, each time predicting one
point from a model that never saw it, and is the fairer number to put in
your before/after table. Even that isn't a true held-out test set (the
regularization strength alpha was still chosen using all 24 points), which
is a limitation worth stating plainly rather than hiding — the labelled set
is both your only training data and your only evaluation data here.

REQUIRES scikit-learn (not needed by credibility.py or the app itself —
only this training script):

    pip install scikit-learn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from credibility import FEATURE_NAMES, extract_features, llm_opinion
from evaluate import LABELLED_URLS

WEIGHTS_PATH = Path(__file__).with_name("learned_weights.json")


def _build_dataset(include_llm: bool) -> tuple[List[List[float]], List[float]]:
    """
    One row per labelled URL, in FEATURE_NAMES order (plus llm_score last, if
    requested). Uses extract_features(use_network=True) so the page-fetch and
    OpenAlex lookups run for real — this reuses credibility.py's own on-disk
    cache, so if you've already run `python evaluate.py`, most of this is
    already cached and fast.
    """
    X: List[List[float]] = []
    y: List[float] = []
    for url, expected, _rationale in LABELLED_URLS:
        features = extract_features(url, use_network=True)
        row = [features[name] for name in FEATURE_NAMES]
        if include_llm:
            llm = llm_opinion(url)
            row.append(llm.value if llm is not None else 0.0)
        X.append(row)
        y.append(expected)
    return X, y


def fit_and_report(include_llm: bool) -> Dict[str, object]:
    feature_names = list(FEATURE_NAMES) + (["llm_score"] if include_llm else [])
    X, y = _build_dataset(include_llm)

    model = LassoCV(cv=LeaveOneOut(), max_iter=10000, random_state=0)
    model.fit(X, y)

    predictions = model.predict(X)
    in_sample_mae = sum(abs(p - actual) for p, actual in zip(predictions, y)) / len(y)

    # HONEST number: refit a plain Lasso at the chosen alpha inside each of
    # the 24 leave-one-out folds, predicting each point from a model that
    # never saw it. This is still not a held-out test set in the usual sense
    # (the *alpha* was chosen using all 24 points via LassoCV above), but it
    # is a much fairer estimate of generalization than the in-sample number,
    # and it's what should anchor the "before/after" claim in your report.
    loo_predictions = cross_val_predict(
        Lasso(alpha=model.alpha_, max_iter=10000, random_state=0),
        X, y, cv=LeaveOneOut(),
    )
    loo_mae = sum(abs(p - actual) for p, actual in zip(loo_predictions, y)) / len(y)

    weights = {
        name: float(coef)
        for name, coef in zip(feature_names, model.coef_)
        if abs(coef) > 1e-6
    }
    dropped = [name for name in feature_names if name not in weights]

    label = "rules + LLM" if include_llm else "rules only"
    print(f"\n{'=' * 70}")
    print(f"  Model: {label}")
    print(f"  alpha (regularization strength) chosen by LOOCV: {model.alpha_:.4f}")
    print(f"  In-sample MAE:          {in_sample_mae:.3f}  (fit-on-same-data — optimistic)")
    print(f"  Leave-one-out MAE:      {loo_mae:.3f}  (each point predicted by a model that never saw it — report THIS one)")
    print(f"{'=' * 70}")
    print(f"\n  Features KEPT by lasso ({len(weights)}/{len(feature_names)}):")
    for name, coef in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {name:<22} {coef:+.4f}")
    if dropped:
        print("\n  Features DROPPED by lasso (coefficient shrunk to ~0):")
        for name in dropped:
            print(f"    {name}")

    return {"intercept": float(model.intercept_), "weights": weights}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit Stage 2 learned weights for credibility.py")
    parser.add_argument("--llm", action="store_true", help="also fit a rules+LLM model (calls the Anthropic API)")
    args = parser.parse_args()

    all_weights: Dict[str, object] = {}
    if WEIGHTS_PATH.exists():
        try:
            all_weights = json.loads(WEIGHTS_PATH.read_text())
        except (OSError, ValueError):
            all_weights = {}

    all_weights["rules"] = fit_and_report(include_llm=False)
    if args.llm:
        all_weights["rules_llm"] = fit_and_report(include_llm=True)

    WEIGHTS_PATH.write_text(json.dumps(all_weights, indent=2))
    print(f"\nSaved weights to {WEIGHTS_PATH}")
    print("Run `python evaluate.py` again to see the updated MAE / band accuracy.\n")
