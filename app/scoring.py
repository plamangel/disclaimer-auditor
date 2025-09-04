from typing import Dict, Any, List, Tuple, Sequence
from sentence_transformers import SentenceTransformer, util
from .utils import split_sentences
import numpy as np
import re

# -----------------------------
# Text normalization & fuzzy helpers
# -----------------------------

_WORD_RE = re.compile(r"[^a-z0-9 ]+")

def _norm_text(s: str) -> str:
    return _WORD_RE.sub(" ", s.lower()).strip()

def _token_set(s: str) -> set:
    return set(w for w in _norm_text(s).split() if len(w) > 2)

def fuzzy_overlap(a: str, b: str, min_overlap: int = 2) -> bool:
    """
    Cheap fuzzy check: returns True if at least `min_overlap` tokens overlap.
    """
    A, B = _token_set(a), _token_set(b)
    return sum(1 for w in A if w in B) >= min_overlap

def matches_anchors(sentence: str, anchors: Sequence[str]) -> bool:
    """
    Returns True if the sentence matches any of the anchors (substring or fuzzy token overlap).
    If anchors is empty/None, always True.
    """
    if not anchors:
        return True
    s_norm = _norm_text(sentence)
    for a in anchors:
        a_norm = _norm_text(a)
        if a_norm and a_norm in s_norm:
            return True
        if fuzzy_overlap(sentence, a, min_overlap=2):
            return True
    return False

def select_candidates(sentences: List[str], anchors: Sequence[str], max_candidates: int = 128) -> List[str]:
    """
    Filters sentences by anchors; falls back to all sentences if none matched.
    """
    if not sentences:
        return []
    cand = [s for s in sentences if matches_anchors(s, anchors or [])]
    if not cand:
        cand = sentences  # fallback to avoid missing everything
    return cand[:max_candidates]


# -----------------------------
# Keyword matching (more permissive / fuzzy)
# -----------------------------

def contains_keywords(text: str, keywords: List[str]) -> bool:
    """
    Returns True if any keyword is a substring OR has >=2 token overlap with the text.
    (Backwards-compatible signature; more tolerant internals.)
    """
    if not keywords:
        return False
    t_norm = _norm_text(text)
    t_tokens = _token_set(text)
    for k in keywords:
        k_norm = _norm_text(k)
        if k_norm and k_norm in t_norm:
            return True
        k_tokens = _token_set(k)
        if sum(1 for w in k_tokens if w in t_tokens) >= 2:
            return True
    return False


# -----------------------------
# Similarity helpers (unchanged API)
# -----------------------------

def top_matching_sentences(agent_sents: List[str], examples: List[str], model: SentenceTransformer, top_k: int = 3):
    if not agent_sents or not examples:
        return []
    sent_emb = model.encode(agent_sents, normalize_embeddings=True)
    ex_emb = model.encode(examples, normalize_embeddings=True)
    sims = util.cos_sim(sent_emb, ex_emb).cpu().numpy()  # shape [num_sents, num_examples]
    # take max similarity per sentence
    max_per_sent = sims.max(axis=1)
    idxs = np.argsort(-max_per_sent)[:top_k]
    return [(agent_sents[i], float(max_per_sent[i])) for i in idxs]

def max_similarity(agent_sents: List[str], canon_examples: List[str], model: SentenceTransformer) -> float:
    if not agent_sents or not canon_examples:
        return 0.0
    sent_emb = model.encode(agent_sents, normalize_embeddings=True)
    ex_emb = model.encode(canon_examples, normalize_embeddings=True)
    sims = util.cos_sim(sent_emb, ex_emb).cpu().numpy()
    return float(sims.max())


# -----------------------------
# Scoring (anchors + YAML weights support)
# -----------------------------

def score_requirement(req: Dict[str, Any], agent_text: str, extractor_flags: Dict[str, str], policy_scoring: Dict[str, Any], model: SentenceTransformer):
    # 0) load per-dimension weights from policy (defaults preserve previous behavior)
    weights_cfg = (policy_scoring or {}).get("weights", {})
    w_kw = float(weights_cfg.get("kw", 0.3))
    w_sim = float(weights_cfg.get("sim", 0.4))
    w_llm = float(weights_cfg.get("llm", 0.3))

    # 1) semantic candidates via ANCHORS (used across all dimensions)
    agent_sents = split_sentences(agent_text)
    anchors = (req or {}).get("anchors", [])
    candidates = select_candidates(agent_sents, anchors, max_candidates=128)

    # 1.1) keywords with polarity (positive/negative) control, evaluated on candidates
    neg_kw = req.get("negative_keywords", [])
    require_neg = bool(req.get("require_negative_polarity", False))

    # positive hits on candidates
    has_pos = any(contains_keywords(s, req.get("keywords", [])) for s in candidates)
    # negative blockers: check candidates first (strict), plus a light transcript-wide guard
    has_neg = any(contains_keywords(s, neg_kw) for s in candidates) or contains_keywords(agent_text, neg_kw)

    # enforce polarity: only count kw if positive and NOT negative
    kw_ok = (has_pos and not has_neg) if not require_neg else (has_pos and not has_neg)

    kw_score_unit = 1.0 if kw_ok else 0.0
    kw_score = w_kw * kw_score_unit  # keep contribution scaled by w_kw

    # 2) semantic similarity; drop contradictory lines from similarity/evidence when negative_keywords exist
    if neg_kw:
        filtered_candidates = [s for s in candidates if not contains_keywords(s, neg_kw)]
    else:
        filtered_candidates = candidates

    sim = max_similarity(filtered_candidates, req.get("canonical_examples", []), model)
    sim_unit = 0.0
    if sim >= policy_scoring.get("similarity_threshold_strong", 0.75):
        sim_unit = 1.0  # strong hit
    elif sim >= policy_scoring.get("similarity_threshold_ok", 0.60):
        sim_unit = 0.5  # ok hit
    sim_score = w_sim * sim_unit

    # 3) LLM extractor (true/unclear/false) with weight from YAML
    flag = (extractor_flags or {}).get(req["id"], "unclear")
    llm_unit = {"true": 1.0, "unclear": 0.3333333333}.get(flag, 0.0)  # scale to [0..1]
    llm_score = w_llm * llm_unit

    # per-requirement weight multiplier (unchanged)
    weight = float(req.get("weight", 0.15))
    total = (kw_score + sim_score + llm_score) * weight

    # evidence (use filtered candidates to avoid off-topic/contradictory picks)
    top_sents = top_matching_sentences(filtered_candidates, req.get("canonical_examples", []), model, top_k=3)
    evidence = [{"quote": s, "start_ms": None, "end_ms": None} for s, _ in top_sents[:3]]

    debug = {
        "kw": kw_score_unit,      # unit score before weights (for transparency)
        "sim": sim,               # raw cosine max
        "sim_score": sim_score,   # weighted contribution
        "llm": llm_score,         # weighted contribution
        "top_sentences": evidence
    }
    return total, debug, evidence

def aggregate_score(agent_text: str, extractor_flags: Dict[str, str], policy: Dict[str, Any], model: SentenceTransformer):
    total = 0.0
    breakdown = {}
    evidence_map = {}
    for req in policy["requirements"]:
        s, dbg, ev = score_requirement(req, agent_text, extractor_flags, policy.get("scoring", {}), model)
        total += s
        breakdown[req["id"]] = dbg
        evidence_map[req["id"]] = ev

    sc = policy.get("scoring", {})
    pass_th = float(sc.get("pass_threshold", 0.90))
    review_th = float(sc.get("review_threshold", 0.70))
    verdict = "PASS" if total >= pass_th else ("NEEDS_REVIEW" if total >= review_th else "FAIL")
    return total, verdict, breakdown, evidence_map