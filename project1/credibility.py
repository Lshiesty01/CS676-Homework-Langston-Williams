"""
credibility.py — THIS IS THE FILE YOU IMPROVE.
===============================================================================

Everything else in this project is scaffolding. The chatbot works, the UI works,
the tracing works. What does NOT work well is the function below: `score_url()`.

-------------------------------------------------------------------------------
STAGE 1 CHANGES (weaknesses #1, #2, #3, #4)
-------------------------------------------------------------------------------
  #1  `page_content_signals()` fetches the actual page and looks for a byline,
      a publish date, outbound citations, and a corrections policy.
  #2  DOMAIN_SCORES is annotated against Wikipedia's Reliable sources/Perennial
      sources list (WP:RSP) categories instead of being an unsourced guess.
  #3  `scholarly_signals()` calls OpenAlex for any DOI and flags per-paper
      preprints via OpenAlex's own `type` field.
  #4  The same OpenAlex call checks `is_retracted` and applies a heavy penalty.

-------------------------------------------------------------------------------
STAGE 2 CHANGES (weaknesses #6, #9) — this revision
-------------------------------------------------------------------------------
Stage 1 still combined every signal with hand-picked constants (0.05 for a
byline, -0.50 for a retraction, RULE_WEIGHT = 0.6, ...). None of those numbers
were ever tested against the labelled data. This revision:

  #6  Replaces the additive `_combine_signals` arithmetic with a *learned*
      linear model. `extract_features()` turns every signal into a fixed-length
      numeric vector; `train_weights.py` (a new, separate script) fits a Lasso
      regression against evaluate.py's 24 labelled URLs, using leave-one-out
      cross-validation to pick the regularization strength. Lasso's L1 penalty
      also does feature selection — the training script prints which features
      survived and which got shrunk to zero, which is the "report which
      features actually matter" the README points at (Session 06).

  #9  RULE_WEIGHT / LLM_WEIGHT are still here as a *fallback* for before you've
      run training, but once learned_weights.json exists, the LLM's opinion is
      itself just another feature the model can learn to weight (or ignore) —
      see `train_weights.py --llm`.

  credibility.py itself does NOT depend on scikit-learn — only the training
  script does. At inference time this file just loads a small JSON file and
  does a dot product, so the app and the grader's environment stay light.

  Until you run `python train_weights.py`, score_url() behaves exactly like
  Stage 1 (same hand-picked constants) — this file works immediately after
  cloning either way, same fail-open principle as llm_opinion() below.

-------------------------------------------------------------------------------
STAGE 3 CHANGES (weakness #7) — this revision
-------------------------------------------------------------------------------
"The score is not calibrated" means a 0.7 doesn't reliably mean "right about
70% of the time". `calibration.py` (a new, separate script) measures this
honestly — Brier score and a reliability table, evaluated via leave-one-out —
and fits a Platt-scaling correction on top of whatever score_url() currently
returns.

DESIGN DECISION, STATED PLAINLY: score_url()'s own output is left unchanged.
Calibration is a property of a predicted PROBABILITY OF A BINARY EVENT ("is
this credible at all"), but score_url() returns a CONTINUOUS graded estimate
(evaluate.py's labels run 0.05 to 0.95, not 0/1). Overwriting score_url()'s
output with a Platt-scaled probability would fix the binary-calibration
question at the cost of the graded distinction your continuous labels encode
— calibration.py measures and prints exactly this trade-off rather than
picking a side for you. The calibrated probability is available separately
via `calibrated_probability(url)` below, for a caller that specifically wants
"P(this source is trustworthy)" rather than a graded score. Whether to make
that the default in your submitted app is a judgment call worth defending in
the report, not something this file decides for you.

-------------------------------------------------------------------------------
STAGE 4 CHANGES (weakness #8) — this revision
-------------------------------------------------------------------------------
"There is no uncertainty" — an unrecognized domain and a famous journal both
returned a bare point estimate, with nothing distinguishing "I'm confident"
from "I'm guessing". `bootstrap_uncertainty.py` (a new, separate script)
resamples the 24 labelled URLs with replacement 500 times, refits a Lasso at
a fixed alpha on each resample, and saves all 500 fitted models to
bootstrap_weights.json. `score_with_uncertainty(url)` below runs a URL's
feature vector through every one of those 500 models and reports a
percentile interval around the point estimate — a URL whose features look
like a lot of the training data (a domain that's literally in DOMAIN_SCORES)
gets a narrow interval because the 500 resampled models mostly agree; a URL
unlike anything in the tiny 24-example training set gets a wide one because
they don't. This isn't hand-tuned per URL — it falls out of the bootstrap
mechanically.

Same architecture as Stages 2 and 3: the expensive fitting happens once in a
separate script that needs scikit-learn; credibility.py just loads the
result and does arithmetic, so the app itself stays light.

-------------------------------------------------------------------------------
THE CONTRACT (do not change this)
-------------------------------------------------------------------------------
    score_url("https://arxiv.org/abs/1706.03762")

    -> {"score": 0.9, "explanation": "arxiv.org is a recognized preprint ..."}

    score:       float in [0.0, 1.0].  0 = not credible, 1 = highly credible.
    explanation: str. A human-readable reason for the score.

Your grader, the evaluation harness (`evaluate.py`), the test suite
(`test_credibility.py`), and the Streamlit app all depend on this exact shape.
If you change the keys or the types, everything downstream breaks.

-------------------------------------------------------------------------------
NEW DEPENDENCIES
-------------------------------------------------------------------------------
    pip install requests beautifulsoup4     # needed by credibility.py itself
    pip install scikit-learn                # needed only by train_weights.py

(add all three to requirements.txt / pyproject.toml so the grader's
environment has them too — scikit-learn only needs to be there if you expect
the grader to re-run training, which is optional; ship learned_weights.json
itself and they don't need to).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# The model used for the Layer 2 judgment. Claude Opus 5 is the most capable
# model; switch to "claude-haiku-4-5" if you are scoring many URLs and want to
# cut cost, or "claude-sonnet-5" for a middle option. Scoring quality will move
# with this choice, so note in your report which model your numbers came from.
JUDGE_MODEL = "claude-opus-5"

# Fallback blend, used only until learned_weights.json exists (see Stage 2
# note above). Once you've run train_weights.py, this constant stops mattering
# for the "rules" model and is only a stopgap for combining a learned rules
# score with a *not yet jointly trained* LLM opinion — train with --llm to
# retire it completely.
RULE_WEIGHT = 0.6
LLM_WEIGHT = 0.4

# Network behaviour for the page-fetch and OpenAlex calls. Both are wrapped in
# try/except so a slow or blocked network degrades the score instead of
# crashing it, matching the existing pattern in llm_opinion() below.
HTTP_TIMEOUT_SECONDS = (3, 5)   # (connect timeout, read timeout)
USER_AGENT = "cs676-credibility-scorer/1.0 (student project; contact via course)"

# OpenAlex asks (politely, not mandatory) for a contact email on every request
# in exchange for a faster, less-rate-limited "polite pool". Set this env var
# if you have one; the code works fine without it.
OPENALEX_MAILTO = os.getenv("OPENALEX_CONTACT_EMAIL", "")


# =============================================================================
# LAYER 1a — RULE-BASED SIGNALS (URL string only)
# =============================================================================
# Each table below is a starting point, not a finished dataset. Extending them
# is the fastest way to move the evaluation numbers, but note that a longer
# lookup table is not the same thing as a better *algorithm* — the report asks
# you to justify your approach, not just your word list.

# Exact-domain judgments. Highest-confidence signal we have.
#
# SOURCING (weakness #2): these are aligned to Wikipedia's Reliable
# sources/Perennial sources list (WP:RSP) categories — "generally reliable",
# "no consensus / additional considerations", "generally unreliable", and
# "deprecated" — rather than invented outright. WP:RSP itself changes over
# time and reflects one community's editorial judgment, not ground truth; if
# you extend this table, note in your report which WP:RSP category (or other
# citable source) backs each addition, and where you disagree with it.
DOMAIN_SCORES: Dict[str, float] = {
    # Peer-reviewed / archival
    "nature.com": 0.95,
    "science.org": 0.95,
    "nejm.org": 0.95,
    "thelancet.com": 0.95,
    "pubmed.ncbi.nlm.nih.gov": 0.92,
    "jamanetwork.com": 0.93,
    "pnas.org": 0.92,
    "arxiv.org": 0.75,          # preprint: NOT peer reviewed
    "biorxiv.org": 0.70,        # preprint: NOT peer reviewed
    "ssrn.com": 0.65,           # working papers: NOT peer reviewed
    # Government / international bodies — WP:RSP treats primary-source data
    # from these as generally reliable, though editorial framing can differ.
    "census.gov": 0.90,
    "who.int": 0.88,
    "imf.org": 0.85,
    "cdc.gov": 0.88,
    # Reference
    "wikipedia.org": 0.65,
    "britannica.com": 0.80,
    # Mainstream press — WP:RSP "generally reliable"
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "bbc.com": 0.82,
    "nytimes.com": 0.80,
    "wsj.com": 0.80,
    "propublica.org": 0.85,
    "npr.org": 0.82,
    "theguardian.com": 0.80,
    # Mainstream press — WP:RSP "no consensus / additional considerations"
    "forbes.com": 0.55,        # contributor network: quality varies sharply
    "seekingalpha.com": 0.40,  # contributor-submitted, light editorial review
    # Q&A / community sites — often correct, unreviewed and unattributed
    "stackoverflow.com": 0.50,
    "scikit-learn.org": 0.80,  # primary documentation for its own library
    # User-generated / self-published
    "medium.com": 0.35,
    "substack.com": 0.35,
    "blogspot.com": 0.25,
    "wordpress.com": 0.25,
    "reddit.com": 0.25,
    "quora.com": 0.20,
    "x.com": 0.15,
    "twitter.com": 0.15,
    # WP:RSP "generally unreliable" / "deprecated"
    "dailymail.co.uk": 0.15,
    "breitbart.com": 0.10,
    "infowars.com": 0.03,
    # Satire — factually false by design, which the LLM layer often misses
    "theonion.com": 0.05,
    "clickhole.com": 0.05,
    "babylonbee.com": 0.05,
}

# Fallback when the exact domain is unknown. Coarse and easy to fool.
TLD_SCORES: Dict[str, float] = {
    ".gov": 0.88,
    ".edu": 0.82,
    ".mil": 0.85,
    ".org": 0.60,
    ".com": 0.50,
    ".net": 0.48,
    ".io": 0.45,
    ".biz": 0.30,
    ".info": 0.30,
    ".xyz": 0.25,
}

# Substrings in the URL path that hint at self-published or low-edit content.
PATH_PENALTIES: Dict[str, float] = {
    "/blog/": -0.10,
    "/opinion/": -0.08,
    "/sponsored/": -0.20,
    "/press-release/": -0.15,
    "/advertorial/": -0.25,
    "/forum/": -0.12,
    "/comments/": -0.12,
}

# Neutral starting point for a URL we know nothing about.
NEUTRAL_SCORE = 0.5


@dataclass
class Signal:
    """One piece of evidence that moved the score, kept so we can explain it."""

    name: str      # short machine-readable label, e.g. "known_domain"
    value: float   # the score or delta this signal contributed (Stage 1 sense)
    reason: str    # human-readable sentence for the explanation field


def _normalize_domain(url: str) -> str:
    """
    Pull a bare lowercase domain out of a URL.

    Strips the scheme, any userinfo, the port, and a leading "www.". Returns an
    empty string when the URL has no host at all, which the caller treats as a
    malformed input.
    """
    host = (urlparse(url).netloc or "").lower()
    host = host.split("@")[-1]      # drop user:pass@
    host = host.split(":")[0]       # drop :port
    if host.startswith("www."):
        host = host[4:]
    return host


def _match_known_domain(domain: str) -> Optional[Tuple[str, float]]:
    """
    Look the domain up in DOMAIN_SCORES, allowing subdomains to match.

    "en.wikipedia.org" matches the "wikipedia.org" entry, and "arxiv.org"
    matches itself. We check the exact domain first so a more specific entry
    always wins over a more general one. (Weakness #11 — subdomains inherit
    the parent's score in full — is still open.)
    """
    if domain in DOMAIN_SCORES:
        return domain, DOMAIN_SCORES[domain]
    for known, score in DOMAIN_SCORES.items():
        if domain.endswith("." + known):
            return known, score
    return None


def _domain_tld_lookup(domain: str) -> Tuple[str, float, float, float]:
    """
    Shared by rule_based_signals() and extract_features() so the "which table
    matched" logic lives in exactly one place.

    :return: (label, domain_score, tld_score, unknown_flag) — exactly one of
             domain_score/tld_score is nonzero, or unknown_flag is 1.0.
    """
    match = _match_known_domain(domain)
    if match:
        known, score = match
        return known, score, 0.0, 0.0
    for tld, score in TLD_SCORES.items():
        if domain.endswith(tld):
            return tld, 0.0, score, 0.0
    return "", 0.0, 0.0, 1.0


_DOI_PATH_RE = re.compile(r"/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")


def _extract_doi(url: str) -> Optional[str]:
    """
    Pull a DOI out of a URL path, if one is present.

    DOIs always start "10." followed by a 4-9 digit registrant code, a slash,
    and a suffix. This matches both direct DOI links (doi.org/10.xxxx/yyyy)
    and publisher URLs that embed the DOI in the path (as JAMA, PNAS, etc. do).
    """
    match = _DOI_PATH_RE.search(urlparse(url).path or "")
    return match.group(1) if match else None


def rule_based_signals(url: str) -> List[Signal]:
    """
    Inspect the URL string and return every signal that fired.

    This runs with no network access and no API key, which is what makes the
    app usable straight after `git clone`. Path-penalty signals are named
    `path_<fragment>` (not just "path") so extract_features() below can map
    each one to its own feature slot instead of collapsing them together.
    """
    signals: List[Signal] = []
    parsed = urlparse(url)
    domain = _normalize_domain(url)

    label, domain_score, tld_score, unknown = _domain_tld_lookup(domain)
    if domain_score:
        signals.append(Signal("known_domain", domain_score, f"'{label}' is a domain we recognize"))
    elif tld_score:
        signals.append(Signal("tld", tld_score, f"'{label}' domains score {tld_score:.2f} by default"))
    else:
        signals.append(Signal("unknown", NEUTRAL_SCORE, "unrecognized domain and TLD"))

    # HTTPS. Weak evidence — a scam site can buy a certificate too.
    if parsed.scheme == "https":
        signals.append(Signal("https", 0.02, "served over HTTPS"))
    elif parsed.scheme == "http":
        signals.append(Signal("no_https", -0.05, "served over plain HTTP"))

    # Path keywords suggesting opinion, sponsorship, or user content.
    path = (parsed.path or "").lower()
    for fragment, delta in PATH_PENALTIES.items():
        if fragment in path:
            key = "path_" + fragment.strip("/").replace("-", "_")
            signals.append(Signal(key, delta, f"URL path contains '{fragment}'"))

    # A DOI in the path implies a registered scholarly work.
    if _extract_doi(url):
        signals.append(Signal("doi", 0.10, "URL contains a DOI, suggesting a registered publication"))

    return signals


# =============================================================================
# LAYER 1b — PAGE CONTENT + SCHOLARLY METADATA (weaknesses #1, #3, #4)
# =============================================================================
# The raw lookups (_page_features_raw, _openalex_raw) are cached on disk and
# shared by two consumers: the Signal-producing functions below (used for
# human-readable explanations and the Stage 1 fallback score) and
# extract_features() (used to build the Stage 2 learned-model input) — so a
# URL is only ever fetched/queried once, not once per consumer.

_CORRECTIONS_RE = re.compile(r"correction[s]?\s*(policy|guideline)?", re.IGNORECASE)
_REFERENCES_HEADING_RE = re.compile(r"references|bibliography|works cited|sources", re.IGNORECASE)
_OPENALEX_PREPRINT_TYPES = {"preprint", "posted-content"}

# On-disk cache for both the page-fetch and OpenAlex lookups — the slow,
# network-bound parts of this file. Best-effort: any read/write failure is
# swallowed, since a missing cache should degrade to "slower", never "broken".
_FEATURE_CACHE_PATH = Path(__file__).with_name(".feature_cache.json")


def _load_feature_cache() -> Dict[str, Dict[str, Any]]:
    try:
        cache = json.loads(_FEATURE_CACHE_PATH.read_text())
    except (OSError, ValueError):
        cache = {}
    cache.setdefault("page", {})
    cache.setdefault("scholarly", {})
    return cache


def _save_feature_cache() -> None:
    try:
        _FEATURE_CACHE_PATH.write_text(json.dumps(_FEATURE_CACHE, indent=2))
    except OSError:
        pass


_FEATURE_CACHE: Dict[str, Dict[str, Any]] = _load_feature_cache()


def _fetch_html(url: str) -> Optional[str]:
    """
    Fetch a URL and return its HTML text, or None on any failure.

    Every failure mode (timeout, DNS error, non-200 status, non-HTML content,
    TLS error, redirect loop) collapses to None rather than raising, matching
    the fail-closed pattern already used by llm_opinion().
    """
    try:
        response = requests.get(
            url,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        content_type = response.headers.get("Content-Type", "")
        if response.status_code != 200 or "html" not in content_type.lower():
            return None
        return response.text
    except requests.RequestException:
        return None


def _page_features_raw(url: str) -> Dict[str, bool]:
    """
    Fetch the page once and extract boolean editorial fingerprints: a byline,
    a publish date, outbound citations, and a corrections policy. Cached on
    disk keyed by URL; returns {} (not an error) whenever the fetch fails.
    """
    if url in _FEATURE_CACHE["page"]:
        return _FEATURE_CACHE["page"][url]

    html = _fetch_html(url)
    if html is None:
        features: Dict[str, bool] = {}
    else:
        soup = BeautifulSoup(html, "html.parser")

        has_author = bool(
            soup.find("meta", attrs={"name": "author"})
            or soup.find("meta", attrs={"property": "article:author"})
            or soup.find(attrs={"rel": "author"})
        )
        has_date = bool(
            soup.find("meta", attrs={"property": "article:published_time"})
            or soup.find("meta", attrs={"name": "date"})
            or soup.find("time", attrs={"datetime": True})
        )
        links = soup.find_all("a", href=True)
        has_citations = (
            any("doi.org" in a["href"] for a in links)
            or bool(soup.find(string=_REFERENCES_HEADING_RE))
        )
        has_corrections = bool(_CORRECTIONS_RE.search(soup.get_text(" ", strip=True)[:20000]))

        features = {
            "has_byline": has_author,
            "has_dateline": has_date,
            "has_citations": has_citations,
            "has_corrections": has_corrections,
        }

    _FEATURE_CACHE["page"][url] = features
    _save_feature_cache()
    return features


def _openalex_raw(url: str) -> Optional[Dict[str, Any]]:
    """
    Look up a DOI on OpenAlex (https://openalex.org/, free, no key) once, and
    return the fields we care about: work type, retraction status, citation
    count. Cached on disk keyed by DOI; returns None if there's no DOI or the
    lookup fails for any reason.
    """
    doi = _extract_doi(url)
    if not doi:
        return None
    if doi in _FEATURE_CACHE["scholarly"]:
        return _FEATURE_CACHE["scholarly"][doi]

    try:
        params = {"mailto": OPENALEX_MAILTO} if OPENALEX_MAILTO else {}
        response = requests.get(
            f"https://api.openalex.org/works/doi:{doi}",
            params=params,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        if response.status_code != 200:
            return None
        work = response.json()
    except (requests.RequestException, ValueError):
        return None

    record = {
        "type": work.get("type"),
        "is_retracted": work.get("is_retracted"),
        "cited_by_count": work.get("cited_by_count"),
    }
    _FEATURE_CACHE["scholarly"][doi] = record
    _save_feature_cache()
    return record


def page_content_signals(url: str) -> List[Signal]:
    """Translate _page_features_raw() into explanation-ready Signals."""
    features = _page_features_raw(url)
    if not features:
        return []

    signals: List[Signal] = []
    if features.get("has_byline"):
        signals.append(Signal("has_byline", 0.05, "page has an identifiable author byline"))
    if features.get("has_dateline"):
        signals.append(Signal("has_dateline", 0.03, "page has a visible publish/update date"))
    if features.get("has_citations"):
        signals.append(Signal("has_citations", 0.05, "page links out to cited sources"))
    if features.get("has_corrections"):
        signals.append(Signal("has_corrections", 0.04, "publisher states a corrections policy"))
    if not any(features.values()):
        signals.append(Signal(
            "thin_content", -0.05,
            "page shows no author, date, citations, or corrections policy",
        ))
    return signals


def scholarly_signals(url: str) -> List[Signal]:
    """Translate _openalex_raw() into explanation-ready Signals."""
    work = _openalex_raw(url)
    if not work:
        return []

    signals: List[Signal] = []
    work_type = str(work.get("type") or "").lower()
    if work_type in _OPENALEX_PREPRINT_TYPES:
        signals.append(Signal(
            "openalex_preprint", -0.15,
            f"OpenAlex classifies this work as '{work_type}' (not peer reviewed)",
        ))
    if work.get("is_retracted") is True:
        signals.append(Signal("retracted", -0.50, "OpenAlex indicates this work has been retracted"))

    citation_count = work.get("cited_by_count")
    if isinstance(citation_count, int) and citation_count > 0:
        boost = min(0.08, math.log10(citation_count + 1) * 0.02)
        signals.append(Signal("citation_count", boost, f"cited by {citation_count} other works per OpenAlex"))

    return signals


def _combine_signals(signals: List[Signal]) -> float:
    """
    Fold the signal list into a single number in [0, 1] using hand-picked
    weights. This is the Stage 1 aggregation, kept as the fallback for when
    learned_weights.json doesn't exist yet (see _learned_score() below).
    """
    if not signals:
        return NEUTRAL_SCORE
    base = signals[0].value
    adjustment = sum(s.value for s in signals[1:])
    return max(0.0, min(1.0, base + adjustment))


# =============================================================================
# STAGE 2 — LEARNED FEATURE VECTOR (weaknesses #6, #9)
# =============================================================================
# A fixed-length numeric feature per URL, independent of which signals fired.
# train_weights.py builds a design matrix out of these (one row per labelled
# URL) and fits a Lasso regression onto the expected scores. At inference
# time, _learned_score() below just computes the same vector and takes a dot
# product with the fitted weights — no sklearn needed here.

FEATURE_NAMES: List[str] = [
    "domain_score", "tld_score", "unknown_domain",
    "https", "no_https",
    "path_blog", "path_opinion", "path_sponsored", "path_press_release",
    "path_advertorial", "path_forum", "path_comments",
    "doi_present",
    "has_byline", "has_dateline", "has_citations", "has_corrections", "thin_content",
    "openalex_preprint", "retracted", "citation_count_log",
]

# Human-readable labels for the learned-model explanation (partial answer to
# weakness #10 — a feature name like "openalex_preprint" isn't itself a good
# explanation, but the sentence built from this label is closer to one).
FEATURE_LABELS: Dict[str, str] = {
    "domain_score": "the domain's known reputation",
    "tld_score": "the top-level domain's default reputation",
    "unknown_domain": "an unrecognized domain and TLD",
    "https": "being served over HTTPS",
    "no_https": "being served over plain HTTP",
    "path_blog": "a '/blog/' path segment",
    "path_opinion": "an '/opinion/' path segment",
    "path_sponsored": "a '/sponsored/' path segment",
    "path_press_release": "a '/press-release/' path segment",
    "path_advertorial": "an '/advertorial/' path segment",
    "path_forum": "a '/forum/' path segment",
    "path_comments": "a '/comments/' path segment",
    "doi_present": "a DOI in the URL",
    "has_byline": "an identifiable author byline",
    "has_dateline": "a visible publish/update date",
    "has_citations": "outbound citations on the page",
    "has_corrections": "a stated corrections policy",
    "thin_content": "no author, date, citations, or corrections policy",
    "openalex_preprint": "OpenAlex classifying this as a preprint",
    "retracted": "OpenAlex marking this work as retracted",
    "citation_count_log": "how often this work is cited (log-scaled)",
    "llm_score": "Claude's own credibility judgment",
}


def extract_features(url: str, use_network: bool = True) -> Dict[str, float]:
    """
    Build the fixed-length numeric feature vector used by the Stage 2 learned
    model (and by train_weights.py to build its design matrix).

    Always returns every key in FEATURE_NAMES, defaulting unfired ones to 0.0
    — every URL gets the same shape of vector, which a regression requires.
    citation_count_log is normalized to roughly [0, 1] (capped at a citation
    count of ~100,000) so it sits on a similar scale to the binary features,
    since this file deliberately skips a formal StandardScaler step.
    """
    features: Dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}

    domain = _normalize_domain(url)
    _label, domain_score, tld_score, unknown = _domain_tld_lookup(domain)
    features["domain_score"] = domain_score
    features["tld_score"] = tld_score
    features["unknown_domain"] = unknown

    parsed = urlparse(url)
    if parsed.scheme == "https":
        features["https"] = 1.0
    elif parsed.scheme == "http":
        features["no_https"] = 1.0

    path = (parsed.path or "").lower()
    for fragment in PATH_PENALTIES:
        key = "path_" + fragment.strip("/").replace("-", "_")
        if key in features and fragment in path:
            features[key] = 1.0

    if _extract_doi(url):
        features["doi_present"] = 1.0

    if use_network:
        page = _page_features_raw(url)
        features["has_byline"] = float(bool(page.get("has_byline")))
        features["has_dateline"] = float(bool(page.get("has_dateline")))
        features["has_citations"] = float(bool(page.get("has_citations")))
        features["has_corrections"] = float(bool(page.get("has_corrections")))
        if page and not any(page.values()):
            features["thin_content"] = 1.0

        work = _openalex_raw(url)
        if work:
            work_type = str(work.get("type") or "").lower()
            if work_type in _OPENALEX_PREPRINT_TYPES:
                features["openalex_preprint"] = 1.0
            if work.get("is_retracted") is True:
                features["retracted"] = 1.0
            citation_count = work.get("cited_by_count")
            if isinstance(citation_count, int) and citation_count > 0:
                features["citation_count_log"] = min(1.0, math.log10(citation_count + 1) / 5.0)

    return features


_LEARNED_WEIGHTS_PATH = Path(__file__).with_name("learned_weights.json")


def _load_learned_weights() -> Dict[str, Any]:
    try:
        return json.loads(_LEARNED_WEIGHTS_PATH.read_text())
    except (OSError, ValueError):
        return {}


_LEARNED_WEIGHTS: Dict[str, Any] = _load_learned_weights()


def _learned_score(features: Dict[str, float], llm: Optional[Signal]) -> Optional[Tuple[float, List[str]]]:
    """
    Apply Stage 2's fitted weights, if train_weights.py has been run.

    :return: None if no learned_weights.json is present yet (score_url then
             falls back to the Stage 1 hand-weighted path), otherwise
             (raw_score, explanation_reasons).
    """
    model_key = "rules_llm" if llm is not None and "rules_llm" in _LEARNED_WEIGHTS else "rules"
    model = _LEARNED_WEIGHTS.get(model_key)
    if not model:
        return None

    weights: Dict[str, float] = model["weights"]
    intercept: float = model["intercept"]

    vector = dict(features)
    if model_key == "rules_llm" and llm is not None:
        vector["llm_score"] = llm.value

    raw = intercept + sum(weights.get(name, 0.0) * vector.get(name, 0.0) for name in weights)

    # Top contributions by magnitude, for a human-readable explanation — a
    # full feature dump reads like a debug log, not an explanation for a
    # reader (weakness #10 is still open, but this is a step toward it).
    contributions = sorted(
        (
            (name, weights.get(name, 0.0) * vector.get(name, 0.0))
            for name in weights
            if vector.get(name, 0.0) != 0.0
        ),
        key=lambda item: -abs(item[1]),
    )
    reasons = [
        f"{FEATURE_LABELS.get(name, name)} (learned weight {weights[name]:+.2f})"
        for name, _delta in contributions[:3]
    ]
    if not reasons:
        reasons = ["no strong learned signals fired for this URL"]

    return raw, reasons


# =============================================================================
# LAYER 2 — LLM JUDGMENT
# =============================================================================

_JUDGE_SYSTEM = """You assess the credibility of web sources for a research assistant.

Given a URL, judge how much a careful reader should trust content published there.
Consider: the publisher's editorial standards and reputation, whether the content is
peer reviewed, whether it is self-published, and whether the outlet is satirical.

Score 0.0 (not credible at all) to 1.0 (highly credible). Be skeptical of
self-published platforms and satire. Judge the SOURCE, not the topic. If you do not
recognize the domain, say so and score near 0.5 rather than guessing confidently."""

# Constraining the response to this schema means we never have to parse prose or
# repair malformed JSON — the API guarantees the shape.
_JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Credibility from 0.0 to 1.0"},
        "reason": {"type": "string", "description": "One sentence justifying the score"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


# =============================================================================
# ⚠️  NEEDS YOUR OWN API KEY — AND SHIPPED UNVERIFIED
# =============================================================================
# REQUIRES A KEY. This function is the only part of credibility.py that calls
# Anthropic. Without ANTHROPIC_API_KEY set it returns None and the scorer falls
# back to rules only — no error, just a weaker score. Get a key at
# https://console.anthropic.com/ and put it in `.env` (copy `.env.example`).
# The key is yours and the calls are billed to you, which is exactly why the
# rules layer, the tests, and evaluate.py were all built to run without one.
# =============================================================================


def llm_opinion(url: str) -> Optional[Signal]:
    """
    Ask Claude to judge the URL. Returns None whenever the call cannot be made.

    Returning None rather than raising is deliberate: a missing API key, a
    network blip, or a safety refusal should degrade the score to rules-only
    instead of taking down the whole app. Effort is set to "low" because this
    is a small judgment and we may be scoring several URLs per question.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": f"Rate the credibility of this source: {url}"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        )

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            return None

        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
        score = max(0.0, min(1.0, float(data["score"])))
        return Signal("llm", score, str(data["reason"]))

    except Exception:
        # Any failure falls back to rules-only scoring rather than crashing.
        return None


# =============================================================================
# THE FUNCTION YOU ARE GRADED ON
# =============================================================================

# Scoring the same URL repeatedly in one session is common (a chat may cite the
# same paper on every turn), so results are memoized for the process lifetime.
_CACHE: Dict[Tuple[str, Optional[bool], Optional[bool]], Dict[str, Any]] = {}


def score_url(
    url: str,
    use_llm: Optional[bool] = None,
    use_network: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Score the credibility of a source URL.

    :param url:         The URL to evaluate.
    :param use_llm:     True forces the Claude judgment, False forces rules
                        only, None (default) uses the LLM when an API key is
                        available.
    :param use_network: True/None (default) fetches the page and queries
                        OpenAlex for DOI metadata; False scores on the URL
                        string alone (fast, offline iteration).
    :return:            {"score": float in [0,1], "explanation": str}
    """
    cache_key = (url, use_llm, use_network)
    if cache_key in _CACHE:
        return dict(_CACHE[cache_key])

    # Guard clause: anything that is not a usable http(s) URL scores 0.0 with an
    # explanation rather than raising, so one bad link cannot break a whole page.
    if not isinstance(url, str) or not url.strip():
        return {"score": 0.0, "explanation": "No URL was provided."}

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not _normalize_domain(url):
        return {"score": 0.0, "explanation": f"'{url}' is not a valid http(s) URL."}

    network_on = use_network is not False
    signals = rule_based_signals(url)
    if network_on:
        signals = signals + page_content_signals(url) + scholarly_signals(url)

    features = extract_features(url, use_network=network_on)
    llm = llm_opinion(url) if use_llm is not False else None

    learned = _learned_score(features, llm)
    if learned is not None:
        # STAGE 2 PATH — learned_weights.json exists; use the fitted model.
        raw_score, reasons = learned
        if llm is not None and "rules_llm" not in _LEARNED_WEIGHTS:
            # A rules-only model was trained but not a combined one yet:
            # blend the LLM opinion on top with the old constants until you
            # run `python train_weights.py --llm`.
            raw_score = RULE_WEIGHT * raw_score + LLM_WEIGHT * llm.value
            reasons.append(f"model judgment {llm.value:.2f} — {llm.reason}")
        final = round(max(0.0, min(1.0, raw_score)), 2)
        explanation = "Fitted from labelled examples (Stage 2): " + "; ".join(reasons) + "."
    else:
        # STAGE 1 FALLBACK PATH — no learned_weights.json yet.
        rule_score = _combine_signals(signals)
        parts = [s.reason for s in signals]
        if llm is not None:
            raw_score = RULE_WEIGHT * rule_score + LLM_WEIGHT * llm.value
            parts.append(f"model judgment {llm.value:.2f} — {llm.reason}")
        else:
            raw_score = rule_score
        final = round(max(0.0, min(1.0, raw_score)), 2)
        explanation = "; ".join(parts) + "."

    result = {"score": final, "explanation": explanation}
    _CACHE[cache_key] = dict(result)
    return result


def score_band(score: float) -> Tuple[str, str]:
    """
    Map a score onto a display band. Used by the app to colour the source chips.

    :return: (label, streamlit_colour) — e.g. ("HIGH", "green")
    """
    if score >= 0.70:
        return "HIGH", "green"
    if score >= 0.40:
        return "MEDIUM", "orange"
    return "LOW", "red"


# =============================================================================
# STAGE 3 — CALIBRATED PROBABILITY (weakness #7)
# =============================================================================
# calibration.py fits sigmoid(platt_a * raw_score + platt_b) against the
# binary "was this actually credible" question and saves the two parameters
# here. This section only ever reads that file; it never influences
# score_url()'s own output (see the Stage 3 design note at the top of this
# file for why that's a deliberate choice, not an oversight).

_CALIBRATION_PATH = Path(__file__).with_name("calibration.json")


def _load_calibration() -> Dict[str, float]:
    try:
        return json.loads(_CALIBRATION_PATH.read_text())
    except (OSError, ValueError):
        return {}


_CALIBRATION: Dict[str, float] = _load_calibration()


def calibrated_probability(url: str, use_llm: Optional[bool] = None, use_network: Optional[bool] = None) -> float:
    """
    Return P(this source is credible), Platt-scaled against the binary
    "was this actually credible" question calibration.py was fit on.

    Falls back to score_url()'s raw score, unchanged, if calibration.py
    hasn't been run yet — same fail-open pattern used throughout this file.
    This is a SEPARATE, more specific question than score_url()'s graded
    estimate ("how credible, on a spectrum") and the two will disagree,
    sometimes by a lot, for a source in the middle of the range — that's
    expected, not a bug; see the Stage 3 note above.
    """
    raw = score_url(url, use_llm=use_llm, use_network=use_network)["score"]
    if not _CALIBRATION:
        return raw
    a = _CALIBRATION.get("platt_a", 1.0)
    b = _CALIBRATION.get("platt_b", 0.0)
    return 1.0 / (1.0 + math.exp(-(a * raw + b)))


# =============================================================================
# STAGE 4 — BOOTSTRAPPED UNCERTAINTY (weakness #8)
# =============================================================================
# bootstrap_uncertainty.py fits 500 Lasso models, each on a different
# resample-with-replacement of the 24 labelled URLs, and saves all 500 to
# bootstrap_weights.json. This section only reads that file and evaluates
# each saved model on a URL's feature vector — no sklearn needed here.

_BOOTSTRAP_WEIGHTS_PATH = Path(__file__).with_name("bootstrap_weights.json")


def _load_bootstrap_models() -> List[Dict[str, Any]]:
    try:
        data = json.loads(_BOOTSTRAP_WEIGHTS_PATH.read_text())
        return data.get("models", [])
    except (OSError, ValueError):
        return []


_BOOTSTRAP_MODELS: List[Dict[str, Any]] = _load_bootstrap_models()


def _percentile(sorted_values: List[float], q: float) -> float:
    """Simple linear-interpolation percentile, avoiding a numpy dependency."""
    if not sorted_values:
        return 0.0
    idx = q * (len(sorted_values) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def score_with_uncertainty(
    url: str,
    use_llm: Optional[bool] = None,
    use_network: Optional[bool] = None,
    confidence: float = 0.90,
) -> Dict[str, Any]:
    """
    Return score_url()'s point estimate plus a bootstrapped confidence
    interval around it.

    :return: {"score": point estimate (same as score_url()'s "score"),
              "low": lower bound, "high": upper bound,
              "n_bootstrap": how many resampled models this used — 0 means
              bootstrap_uncertainty.py hasn't been run yet, and low/high
              collapse to the point estimate (fail-open, same pattern as
              calibrated_probability()).}

    The interval isn't hand-tuned per domain: it's wide when the 500
    resampled models disagree about this particular URL's feature vector
    (unfamiliar territory for the tiny 24-example training set) and narrow
    when they agree (a domain the training data covers well) — that's an
    emergent property of the bootstrap, not something looked up per URL.
    """
    point = score_url(url, use_llm=use_llm, use_network=use_network)["score"]

    if not _BOOTSTRAP_MODELS:
        return {"score": point, "low": point, "high": point, "n_bootstrap": 0}

    features = extract_features(url, use_network=(use_network is not False))
    predictions = []
    for model in _BOOTSTRAP_MODELS:
        weights: Dict[str, float] = model["weights"]
        intercept: float = model["intercept"]
        raw = intercept + sum(weights.get(name, 0.0) * features.get(name, 0.0) for name in weights)
        predictions.append(max(0.0, min(1.0, raw)))
    predictions.sort()

    tail = (1.0 - confidence) / 2.0
    low = round(_percentile(predictions, tail), 2)
    high = round(_percentile(predictions, 1.0 - tail), 2)
    return {"score": point, "low": low, "high": high, "n_bootstrap": len(_BOOTSTRAP_MODELS)}


# =============================================================================
# KNOWN WEAKNESSES — YOUR TASK LIST
# =============================================================================
#
#  1. [STAGE 1 — DONE] IT NEVER READS THE PAGE.
#  2. [STAGE 1 — PARTIAL] THE DOMAIN TABLE IS A HAND-WRITTEN GUESS — now
#     annotated against WP:RSP, still a hand-transcribed subset.
#  3. [STAGE 1 — DONE] IT CANNOT TELL A PREPRINT FROM A PEER-REVIEWED PAPER.
#  4. [STAGE 1 — DONE] IT HAS NEVER HEARD OF RETRACTION.
#
#  5. ANY .edu SCORES HIGHLY. Not addressed.
#
#  6. [STAGE 2 — DONE, WITH CAVEATS] THE AGGREGATION IS ARITHMETIC, NOT
#     STATISTICAL. extract_features() + train_weights.py replace the hand
#     picked deltas with a Lasso regression fitted on the 24 labelled URLs.
#     CAVEAT: n=24 with up to 20 features is a small-data regime; lasso's L1
#     penalty and leave-one-out CV are the standard mitigations, not a cure.
#     The in-sample MAE that train_weights.py prints is fit-on-same-data, not
#     a held-out test — say so plainly in the report.
#
#  7. [STAGE 3 — DONE, AS A SEPARATE FUNCTION] THE SCORE IS NOT CALIBRATED.
#     calibration.py measures Brier score / a reliability table (honestly,
#     via leave-one-out) and fits Platt scaling; calibrated_probability()
#     exposes the result. score_url()'s own output is deliberately left
#     unchanged — see the Stage 3 design note — because calibration answers
#     a binary question ("credible or not") and score_url() answers a
#     graded one; forcing one function to answer both was a real trade-off,
#     not a free win. Measured trade-off (run calibration.py to reproduce):
#     calibrating for the binary question moved MAE against the original
#     continuous labels in a direction worth reading about before you decide
#     whether to make it the app's default.
#
#  8. [STAGE 4 — DONE, AS A SEPARATE FUNCTION] THERE IS NO UNCERTAINTY.
#     bootstrap_uncertainty.py fits 500 Lasso models on resamples of the 24
#     labelled URLs; score_with_uncertainty() evaluates all 500 on a URL and
#     reports a percentile interval. Same design choice as #7: score_url()'s
#     own output is untouched (still just "score" and "explanation"), and
#     the interval lives in a separate function so the contract never moves.
#     CAVEAT: the interval only reflects uncertainty the *model* has about
#     ITS OWN fit given resamples of this same 24-row training set. It does
#     not know about label noise (weakness #7's neighbor — "the labels are
#     one instructor's judgment, not ground truth", per evaluate.py's own
#     docstring), and it cannot widen for a URL whose true credibility this
#     tiny, US-news-and-journals-heavy label set was never going to cover
#     well in the first place (a source in a language or domain this label
#     set has zero examples of, say).
#
#  9. [STAGE 2 — DONE for the rules-only model] THE TWO LAYERS ARE BLENDED
#     WITH A CONSTANT. Once learned_weights.json has a "rules_llm" entry
#     (`python train_weights.py --llm`), the LLM's opinion is just another
#     learned feature. Until then, RULE_WEIGHT/LLM_WEIGHT still blend the
#     learned rules score with a not-yet-integrated LLM opinion — see the
#     "rules_llm" branch in score_url().
#
# 10. THE EXPLANATION IS A LIST OF FRAGMENTS JOINED BY SEMICOLONS (Stage 1
#     path) or the top-3 learned contributions (Stage 2 path). Better than
#     before, but still not prose written *for a reader*. Still open.
#
# 11. SUBDOMAINS INHERIT THE PARENT'S REPUTATION IN FULL. Still true.
#
# 12. PENALTIES STACK WITHOUT A FLOOR in the Stage 1 fallback path. The
#     Stage 2 learned path doesn't have this problem in the same way — lasso
#     can down-weight or zero out a feature that's redundant with another —
#     but it can still sum multiple learned contributions past what's
#     sensible for a single URL; nothing clips per-feature contributions.
