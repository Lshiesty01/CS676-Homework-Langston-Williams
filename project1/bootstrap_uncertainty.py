"""
bootstrap_uncertainty.py — fit a bootstrap ensemble for confidence intervals
(weakness #8).

    python bootstrap_uncertainty.py

Resamples the 24 labelled URLs WITH REPLACEMENT 500 times, fits a Lasso
(at a fixed regularization strength, chosen once via LOOCV on the full
dataset — the same method train_weights.py uses) on each resample, and saves
all 500 fitted models to bootstrap_weights.json.

credibility.score_with_uncertainty(url) then evaluates a URL's feature
vector against all 500 saved models and reports a percentile interval
around the point estimate. Intuition: if a URL's features look like plenty
of the training data (its domain is literally in DOMAIN_SCORES), the 500
resampled-and-refit models mostly agree, giving a narrow interval. If its
features are unlike anything the 24-example training set covers, the models
disagree more, giving a wider one. Nothing is looked up or hand-tuned per
URL — the width is an emergent property of how much the training data
actually constrains that region of feature space.

WHY A FIXED ALPHA, NOT A FRESH LassoCV INSIDE EVERY RESAMPLE: re-selecting
the regularization strength inside every one of 500 resamples would be its
own nested cross-validation (500 LOOCV searches instead of 1), expensive
and answering a subtly different question (uncertainty in the hyperparameter
search itself, not in the fitted predictions). Fixing alpha once via the
full dataset and bootstrapping only the model FIT is the standard, cheaper
approach when the interval you want is on the model's predictions.

CAVEAT FOR YOUR REPORT: this interval reflects uncertainty in the fitted
model given resamples of THIS 24-row training set. It is not the same thing
as "how often is this score actually right" (that's weakness #7's
question), and it can't widen for a source unlike anything the training set
represents in kind (not just in feature values) — see the note in
credibility.py's KNOWN WEAKNESSES #8.

REQUIRES scikit-learn (only this script, not credibility.py itself):

    pip install scikit-learn
"""

from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import LeaveOneOut

import credibility
from credibility import FEATURE_NAMES, extract_features
from evaluate import LABELLED_URLS

WEIGHTS_PATH = Path(__file__).with_name("bootstrap_weights.json")
N_BOOTSTRAP = 500
RANDOM_SEED = 0


def _design_matrix() -> Tuple[List[List[float]], List[float]]:
    X: List[List[float]] = []
    y: List[float] = []
    for url, expected, _rationale in LABELLED_URLS:
        features = extract_features(url, use_network=True)
        X.append([features[name] for name in FEATURE_NAMES])
        y.append(expected)
    return X, y


def main() -> None:
    X, y = _design_matrix()
    n = len(y)

    print("Choosing alpha once via LOOCV on the full 24-row dataset "
          "(same method as train_weights.py)...")
    base = LassoCV(cv=LeaveOneOut(), max_iter=10000, random_state=0)
    base.fit(X, y)
    alpha = float(base.alpha_)
    print(f"  alpha = {alpha:.4f}")

    rng = random.Random(RANDOM_SEED)
    models: List[Dict[str, object]] = []
    kept_counts: Dict[str, int] = {name: 0 for name in FEATURE_NAMES}

    print(f"\nFitting {N_BOOTSTRAP} bootstrap resamples...")
    for _ in range(N_BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]   # sample WITH replacement
        Xb = [X[i] for i in idx]
        yb = [y[i] for i in idx]
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(Xb, yb)
        weights = {name: float(c) for name, c in zip(FEATURE_NAMES, model.coef_)}
        for name, c in weights.items():
            if abs(c) > 1e-6:
                kept_counts[name] += 1
        models.append({"intercept": float(model.intercept_), "weights": weights})

    print(f"\nFeature selection stability across {N_BOOTSTRAP} resamples")
    print("(fraction of resamples that kept each feature nonzero — a feature")
    print(" kept in, say, 30% of resamples has a sign/magnitude you shouldn't")
    print(" trust, even if train_weights.py's single fit happened to keep it):")
    for name, count in sorted(kept_counts.items(), key=lambda kv: -kv[1]):
        if count == 0:
            continue
        print(f"    {name:<22} {count / N_BOOTSTRAP:>6.1%}")

    WEIGHTS_PATH.write_text(json.dumps({"alpha": alpha, "models": models}, indent=2))
    print(f"\nSaved {N_BOOTSTRAP} bootstrap models to {WEIGHTS_PATH}")

    # credibility.py already imported bootstrap_weights.json's (empty, at the
    # time) contents at module-load time above; reload so the demo below
    # picks up the file we just wrote.
    importlib.reload(credibility)

    print("\nDemonstration — interval width should track training-data familiarity:")
    demo_urls = [
        "https://www.nature.com/articles/example",           # exact DOMAIN_SCORES match
        "https://www.a-domain-nobody-has-heard-of-xyz123.com/page",  # unlike the training set
    ]
    for demo_url in demo_urls:
        result = credibility.score_with_uncertainty(demo_url, use_llm=False)
        width = result["high"] - result["low"]
        print(f"  {demo_url}")
        print(f"    point={result['score']:.2f}  90% CI=[{result['low']:.2f}, {result['high']:.2f}]  width={width:.2f}")


if __name__ == "__main__":
    main()
