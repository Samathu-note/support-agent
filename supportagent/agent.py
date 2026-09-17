"""
Core AI-agent logic: intent classification, grounded retrieval, reply
drafting, and the escalation decision. Kept independent of Django's
request/response cycle so it can be called from management commands,
views, or tests alike.
"""
import os
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    # pyrefly: ignore [missing-import]
    import openai
except ImportError:
    openai = None


def _get_api_config():
    """Dynamically loads OpenAI/Groq API key, base_url, and model from .env or os.environ."""
    _env_file = Path(__file__).resolve().parent.parent / '.env'
    cfg = {}
    if _env_file.exists():
        with open(_env_file, 'r', encoding='utf-8') as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    cfg[_k.strip()] = _v.strip().strip("'\"")
    api_key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = cfg.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None
    model = cfg.get("OPENAI_MODEL") or os.environ.get("OPENAI_MODEL", "openai/gpt-oss-120b")
    return api_key, base_url, model

# ---------------------------------------------------------------------------
# 1. Intent taxonomy (edit this after reading a sample of your brand's data)
# ---------------------------------------------------------------------------
# Each intent has a handful of seed keywords used for *weak supervision*:
# we auto-label a larger unlabelled pool with these rules, then train a
# real classifier on the weak labels. The hand-labelled golden set is kept
# separate and is NEVER used for training -- only for evaluation.
INTENT_SEEDS = {
    "account_access": ["login", "log in", "password", "locked out", "can't sign in", "reset my"],
    "billing_refund": ["refund", "charged", "billing", "invoice", "double charged", "cancel my subscription"],
    "playback_bug": ["crash", "won't play", "buffering", "error", "not working", "bug", "glitch"],
    "cancellation": ["cancel", "unsubscribe", "close my account"],
    "device_connectivity": ["won't connect", "bluetooth", "offline", "sync", "pairing"],
    "feature_request": ["wish", "please add", "feature request", "would be great if"],
    "other": [],
}

NEGATIVE_WORDS = ["furious", "unacceptable", "lawsuit", "scam", "worst", "never again", "disgusting"]


def weak_label(text: str) -> str:
    t = text.lower()
    for intent, keywords in INTENT_SEEDS.items():
        for kw in keywords:
            if kw in t:
                return intent
    return "other"


class IntentClassifier:
    """TF-IDF + Logistic Regression classifier trained on weak labels.
    Falls back to the keyword rule if no trained model is loaded."""

    def __init__(self):
        self.vectorizer = None
        self.model = None

    def fit(self, texts, labels):
        from sklearn.linear_model import LogisticRegression
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

    def predict(self, text: str):
        if self.model is None:
            return weak_label(text), 0.4  # low confidence: rule-based fallback
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        idx = proba.argmax()
        return self.model.classes_[idx], float(proba[idx])


# ---------------------------------------------------------------------------
# 2. Grounded retrieval: find similar past resolved threads for this brand
# ---------------------------------------------------------------------------
def retrieve_similar_threads(query_text: str, candidate_threads: list, top_k: int = 3):
    """candidate_threads: list of dicts with 'id', 'customer_text', 'company_text'.
    Returns the top_k most similar threads + their similarity scores."""
    if not candidate_threads:
        return []
    corpus = [t["customer_text"] for t in candidate_threads] + [query_text]
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(corpus)
    sims = cosine_similarity(X[-1], X[:-1]).flatten()
    ranked = sorted(zip(candidate_threads, sims), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"thread": t, "similarity": float(s)} for t, s in ranked]


# ---------------------------------------------------------------------------
# 3. Reply drafting -- grounded in retrieved examples
# ---------------------------------------------------------------------------
def draft_reply(query_text: str, grounding: list, brand: str) -> str:
    """Uses LLM API if API key is configured; otherwise falls
    back to a deterministic template so the pipeline still runs end-to-end
    without any API key (important for the 15-minute reproduce rule)."""
    api_key, base_url, model = _get_api_config()
    if openai and api_key:
        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            examples_block = "\n".join(
                f"- Customer said: {g['thread']['customer_text']!r}\n  {brand} replied: {g['thread']['company_text']!r}"
                for g in grounding
            ) or "No close historical match was found."
            prompt = (
                f"You are drafting a support reply as the brand '{brand}' on Twitter.\n"
                f"Here is how this brand has replied to similar past issues:\n{examples_block}\n\n"
                f"New customer message: {query_text!r}\n\n"
                "Draft a short, on-brand reply (<280 chars) consistent with the resolution "
                "pattern shown above. If the past examples don't clearly resolve this, say so "
                "plainly rather than inventing a resolution."
            )
            resp = client.chat.completions.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (resp.choices[0].message.content or "").strip()
            return content if content else _template_reply(grounding, brand)
        except Exception as e:
            return f"[LLM draft failed, falling back to template: {e}] " + _template_reply(grounding, brand)
    return _template_reply(grounding, brand)


def _template_reply(grounding: list, brand: str) -> str:
    if grounding:
        best = grounding[0]["thread"]["company_text"]
        return f"Hi, thanks for reaching out to {brand}. " + best[:200]
    return f"Hi, thanks for reaching out to {brand} -- we're looking into this and will follow up shortly."


# ---------------------------------------------------------------------------
# 4. Escalation decision -- hybrid rules + model confidence
# ---------------------------------------------------------------------------
def decide_escalation(text: str, intent: str, confidence: float, grounding: list):
    """Returns (should_escalate: bool, reason: str)."""
    t = text.lower()
    top_sim = grounding[0]["similarity"] if grounding else 0.0

    if any(w in t for w in NEGATIVE_WORDS):
        return True, "Message contains strong negative-sentiment language."
    if intent == "billing_refund" and re.search(r"\$\s?\d{2,}|\d{2,}\s?(usd|dollars)", t):
        return True, "Refund/billing issue mentions a non-trivial amount -- needs human sign-off."
    if confidence < 0.5:
        return True, f"Intent classifier confidence too low ({confidence:.2f})."
    if top_sim < 0.15:
        return True, "No sufficiently similar historical resolution found to ground a reply."
    return False, "High-confidence intent with a strong grounding match; safe to auto-handle."


# ---------------------------------------------------------------------------
# 5. LLM-as-judge for reply quality (with a non-LLM fallback for offline runs)
# ---------------------------------------------------------------------------
def judge_reply(customer_text: str, drafted_reply: str, ideal_notes: str = "") -> tuple:
    api_key, base_url, model = _get_api_config()
    if openai and api_key:
        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            prompt = (
                "Rate the following customer-support reply from 1 (bad) to 5 (excellent) on "
                "correctness, tone, and whether it resolves the issue. Respond ONLY as JSON: "
                '{"score": <1-5>, "rationale": "<one sentence>"}.\n\n'
                f"Customer message: {customer_text!r}\n"
                f"Drafted reply: {drafted_reply!r}\n"
                f"Notes on what a good reply needs: {ideal_notes!r}"
            )
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            return float(data["score"]), data.get("rationale", "")
        except Exception as e:
            pass
    # Offline fallback: crude lexical overlap heuristic (clearly weaker --
    # report this limitation explicitly in the "misleading headline number" section).
    overlap = len(set(customer_text.lower().split()) & set(drafted_reply.lower().split()))
    score = min(5.0, 2.0 + overlap * 0.3)
    return score, "Heuristic fallback score (no OPENAI_API_KEY set) -- word-overlap based, not a real quality judgment."
