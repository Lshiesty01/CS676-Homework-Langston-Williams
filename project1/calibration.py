"""
calibration.py — measure and (optionally) fix calibration (weakness #7).

    python calibration.py

"Score is not calibrated" means: among all the URLs the scorer gives 0.7,
roughly 70% of them should actually be credible. Nothing in credibility.py
checks whether that's true — a 0.7 could mean "reliably 70% right" or "I made
some numbers add up to 0.7", and there was no way to tell the difference. This
script measures the gap, using the reframing the README itself uses ("a 0.7
should mean right about 70% of the time"):

  1. Turn each labelled URL's *continuous* expected score into a BINARY label:
     "credible" if expected >= CREDIBLE_THRESHOLD, else "not credible". This
     is the only way to ask a calibration question at all — calibration is a
     property of a predicted PROBABILITY of a binary event, and evaluate.py's
     labels are graded (0.05 .. 0.95), not binary.
  2. Report the Brier score and a reliability table for the scores
     credibility.score_url() currently produces, treating them as if they
     were P(credible) — this is the "how wrong is it" measurement.
  3. Fit Platt scaling (a 1-parameter-pair logistic recalibration,
     sigmoid(A * raw_score + B)) to correct it, evaluated honestly via
     leave-one-out cross-validation — same LOOCV discipline as
     train_weights.py, and low-risk here since Platt scaling only has 2
     parameters, unlike Stage 2's 21-feature regression.
  4. Report a trade-off you should discuss in your write-up: calibrating for
     the BINARY "is this credible at all" question can move MAE against the
     ORIGINAL CONTINUOUS labels (evaluate.py's metric) in either direction,
     because the two are different objectives. This script prints both so
     you can see whether that happened, not just assert it didn't.

CAVEAT WORTH PUTTING IN YOUR REPORT, IN BOLD: n=24 total, and CREDIBLE_THRESHOLD
splits it into roughly 15/9. A reliability table over 5 bins puts on the order
of 3-5 points per bin. Every number below is real, but "real" and "precise"
are different things at this sample size — a bin with 3 points showing 100%
empirical accuracy is not strong evidence of anything. Say so.

Writes calibration.json next to this file. credibility.py's
calibrated_probability() loads it automatically; deleting the file makes
that function fall back to returning the raw, uncalibrated score.

REQUIRES scikit-learn (only this script, not credibility.py itself):

    pip install scikit-learn
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from credibility import score_url
from evaluate import LABELLED_URLS

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")

# Where the continuous label gets cut into a binary one. 0.5 is the natural
# midpoint of the [0, 1] scale and splits this set roughly 15/9 (credible /
# not). The HIGH-band cutoff (0.70) is an equally defensible choice with a
# different, more skewed split (10/14) — which threshold you pick changes
# the calibration numbers below, and that choice is worth defending in your
# report rather than treating 0.5 as an obviously-correct default.
CREDIBLE_THRESHOLD = 0.5

N_BINS = 5


def _binary_dataset() -> Tuple[List[float], List[int], List[float]]:
    """Raw scores, binary labels, and original continuous labels, aligned."""
    raw_scores: List[float] = []
    binary_labels: List[int] = []
    continuous_labels: List[float] = []
    for url, expected, _rationale in LABELLED_URLS:
        raw = score_url(url, use_llm=False)["score"]
        raw_scores.append(raw)
        binary_labels.append(1 if expected >= CREDIBLE_THRESHOLD else 0)
        continuous_labels.append(expected)
    return raw_scores, binary_labels, continuous_labels


def brier_score(predicted_probs: List[float], binary_labels: List[int]) -> float:
    """Mean squared error between a predicted probability and a 0/1 outcome."""
    return sum((p - y) ** 2 for p, y in zip(predicted_probs, binary_labels)) / len(binary_labels)


def mean_abs_error(predicted: List[float], actual: List[float]) -> float:
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(actual)


def reliability_table(predicted_probs: List[float], binary_labels: List[int]) -> float:
    """
    Print a reliability table: for each score bin, how many points fell in
    it, what the model predicted on average, and what fraction actually
    turned out to be "credible". A well-calibrated model has predicted ≈
    empirical in every row. Returns the Expected Calibration Error (ECE) —
    the bin-count-weighted average gap between predicted and empirical.
    """
    bins = [[] for _ in range(N_BINS)]
    edges = [i / N_BINS for i in range(N_BINS + 1)]
    for p, y in zip(predicted_probs, binary_labels):
        idx = min(int(p * N_BINS), N_BINS - 1)
        bins[idx].append((p, y))

    print(f"\n  {'bin':<12}{'n':>4}{'mean predicted':>16}{'empirical frac':>16}{'gap':>8}")
    ece = 0.0
    for i, bucket in enumerate(bins):
        lo, hi = edges[i], edges[i + 1]
        if not bucket:
            print(f"  [{lo:.1f}, {hi:.1f})  {'0':>4}{'—':>16}{'—':>16}{'—':>8}")
            continue
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        empirical = sum(y for _, y in bucket) / len(bucket)
        gap = abs(mean_pred - empirical)
        ece += gap * len(bucket) / len(predicted_probs)
        print(f"  [{lo:.1f}, {hi:.1f})  {len(bucket):>4}{mean_pred:>16.2f}{empirical:>16.2f}{gap:>8.2f}")
    return ece


def main() -> None:
    raw_scores, binary_labels, continuous_labels = _binary_dataset()
    n_credible = sum(binary_labels)

    print(f"\n{'=' * 70}")
    print(f"  Binary split at CREDIBLE_THRESHOLD={CREDIBLE_THRESHOLD}: "
          f"{n_credible} credible / {len(binary_labels) - n_credible} not credible "
          f"(n={len(binary_labels)})")
    print(f"{'=' * 70}")

    print("\nBEFORE calibration — treating score_url()'s raw score as P(credible):")
    print(f"  Brier score: {brier_score(raw_scores, binary_labels):.3f}  (0 = perfect, 0.25 = 'always predict 0.5')")
    uncalibrated_ece = reliability_table(raw_scores, binary_labels)
    print(f"\n  Expected Calibration Error (ECE): {uncalibrated_ece:.3f}")

    # Platt scaling: fit sigmoid(A * raw_score + B) -> P(credible). Only 2
    # parameters, so this is a much smaller ask of 24 data points than
    # Stage 2's 21-feature regression was.
    X = [[s] for s in raw_scores]
    platt = LogisticRegression()
    platt.fit(X, binary_labels)

    # Honest evaluation: leave-one-out, same discipline as train_weights.py.
    loo_probs = cross_val_predict(
        LogisticRegression(), X, binary_labels, cv=LeaveOneOut(), method="predict_proba",
    )[:, 1]

    print("\nAFTER calibration — Platt-scaled probability, leave-one-out honest:")
    print(f"  Brier score: {brier_score(loo_probs, binary_labels):.3f}")
    calibrated_ece = reliability_table(list(loo_probs), binary_labels)
    print(f"\n  Expected Calibration Error (ECE): {calibrated_ece:.3f}")

    print(f"\n{'=' * 70}")
    print("  TRADE-OFF CHECK — calibration is fit for the BINARY question,")
    print("  not the CONTINUOUS one evaluate.py measures. Both MAEs below use")
    print("  the same original continuous labels (0.05 .. 0.95):")
    print(f"{'=' * 70}")
    print(f"  MAE, raw score vs. continuous label:         {mean_abs_error(raw_scores, continuous_labels):.3f}")
    print(f"  MAE, calibrated probability vs. continuous:  {mean_abs_error(list(loo_probs), continuous_labels):.3f}")
    print("  (If the second number is worse, that's expected, not a mistake:")
    print("   Platt scaling pulls everything toward 0/1 to be a good binary")
    print("   probability, which can throw away the graded distinction your")
    print("   continuous labels encode, e.g. Reuters 0.85 vs. Nature 0.95.")
    print("   Report this honestly — it's a real limitation of bolting a")
    print("   binary calibration fix onto a continuous scoring task.)\n")

    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    CALIBRATION_PATH.write_text(json.dumps(
        {"credible_threshold": CREDIBLE_THRESHOLD, "platt_a": a, "platt_b": b}, indent=2,
    ))
    print(f"Saved Platt scaling parameters to {CALIBRATION_PATH}")
    print("credibility.calibrated_probability(url) will use them; score_url()'s")
    print("own output is left as-is — see the Stage 3 note in credibility.py for why.\n")


if __name__ == "__main__":
    main()
