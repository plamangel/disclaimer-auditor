from typing import Dict, Any, List, Sequence
from sentence_transformers import SentenceTransformer, util
from .utils import split_sentences, split_sentences_with_times
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

def select_candidates(sentences: List[str], anchors: Sequence[str], max_candidates: int = 128) -> List[int]:
    """
    Returns indices of sentences that match anchors; falls back to all indices if none matched.
    """
    if not sentences:
        return []
    idxs = [i for i, s in enumerate(sentences) if matches_anchors(s, anchors or [])]
    if not idxs:
        idxs = list(range(len(sentences)))  # fallback to avoid missing everything
    return idxs[:max_candidates]


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


# -------- Preencoded similarity helpers --------

def max_similarity_preencoded(sent_emb: np.ndarray, idx_list: List[int], ex_emb: np.ndarray) -> float:
    """Compute max cosine similarity using pre-normalized embeddings.
    sent_emb: [num_sents, dim], ex_emb: [num_examples, dim]
    """
    if sent_emb is None or ex_emb is None:
        return 0.0
    if not isinstance(sent_emb, np.ndarray) or not isinstance(ex_emb, np.ndarray):
        return 0.0
    if ex_emb.size == 0 or len(idx_list) == 0:
        return 0.0
    cand = sent_emb[idx_list]  # [k, dim]
    sims = cand @ ex_emb.T     # cosine if both are normalized
    return float(np.max(sims))


def top_by_similarity_preencoded(sentences: List[str], sent_emb: np.ndarray, idx_list: List[int], ex_emb: np.ndarray, top_k: int = 3):
    if sent_emb is None or ex_emb is None:
        return []
    if not isinstance(sent_emb, np.ndarray) or not isinstance(ex_emb, np.ndarray):
        return []
    if ex_emb.size == 0 or len(idx_list) == 0:
        return []
    cand = sent_emb[idx_list]
    sims = cand @ ex_emb.T  # [k, m]
    max_per = np.max(sims, axis=1)  # [k]
    order = np.argsort(-max_per)[:top_k]
    return [(sentences[idx_list[i]], float(max_per[i])) for i in order]


# -----------------------------
# Scoring (anchors + YAML weights support)
# -----------------------------

def score_requirement(req: Dict[str, Any], agent_text: str, extractor_flags: Dict[str, str], policy_scoring: Dict[str, Any], model: SentenceTransformer):
    # 0) load per-dimension weights from policy (defaults preserve previous behavior)
    weights_cfg = (policy_scoring or {}).get("weights", {})
    w_kw = float(weights_cfg.get("kw", 0.3))
    w_sim = float(weights_cfg.get("sim", 0.4))
    w_llm = float(weights_cfg.get("llm", 0.3))

    # 1) sentences & optional times bundle
    sent_time_bundle = policy_scoring.get("_sentence_times_bundle") if isinstance(policy_scoring, dict) else None
    if sent_time_bundle and sent_time_bundle.get("sentences") and sent_time_bundle.get("times"):
        sentences = sent_time_bundle["sentences"]
        sentence_times = sent_time_bundle["times"]
    else:
        sentences = split_sentences(agent_text)
        sentence_times = [(None, None) for _ in sentences]

    pre = policy_scoring.get("_pre_encoded") if isinstance(policy_scoring, dict) else None
    sent_emb = pre.get("sent_emb") if isinstance(pre, dict) else None

    # 1.1) anchor-based candidate indices
    anchors = (req or {}).get("anchors", [])
    idx_candidates = select_candidates(sentences, anchors, max_candidates=128)
    candidates = [sentences[i] for i in idx_candidates]

    # 1.2) keyword polarity on candidates
    neg_kw = req.get("negative_keywords", [])
    require_neg = bool(req.get("require_negative_polarity", False))

    has_pos = any(contains_keywords(sentences[i], req.get("keywords", [])) for i in idx_candidates)
    has_neg = any(contains_keywords(sentences[i], neg_kw) for i in idx_candidates) or contains_keywords(agent_text, neg_kw)
    kw_ok = (has_pos and not has_neg) if not require_neg else (has_pos and not has_neg)

    kw_score_unit = 1.0 if kw_ok else 0.0
    kw_score = w_kw * kw_score_unit

    # Example embeddings (precomputed at policy load). Convert to ndarray for fast math.
    ex_emb_list = req.get("_ex_emb") or []
    ex_emb = np.array(ex_emb_list, dtype=float) if ex_emb_list else np.zeros((0, 0), dtype=float)

    # 2) similarity on filtered candidates (remove negatives for evidence/sim)
    def not_negative(i: int) -> bool:
        return not contains_keywords(sentences[i], neg_kw) if neg_kw else True

    filtered_idx = [i for i in idx_candidates if not_negative(i)]
    if not filtered_idx:
        filtered_idx = idx_candidates

    filtered_candidates = [sentences[i] for i in filtered_idx]

    if isinstance(sent_emb, np.ndarray) and ex_emb.size > 0:
        sim = max_similarity_preencoded(sent_emb, filtered_idx, ex_emb)
    else:
        # Fallback to on-the-fly encoding if preencoded not available
        sim = max_similarity([sentences[i] for i in filtered_idx], req.get("canonical_examples", []), model)

    sim_unit = 0.0
    if sim >= policy_scoring.get("similarity_threshold_strong", 0.75):
        sim_unit = 1.0
    elif sim >= policy_scoring.get("similarity_threshold_ok", 0.60):
        sim_unit = 0.5
    sim_score = w_sim * sim_unit

    # 3) LLM extractor
    flag = (extractor_flags or {}).get(req["id"], "unclear")
    llm_unit = {"true": 1.0, "unclear": 0.3333333333}.get(flag, 0.0)
    llm_score = w_llm * llm_unit

    # 4) total with per-requirement weight
    weight = float(req.get("weight", 0.15))
    total = (kw_score + sim_score + llm_score) * weight

    # 5) evidence with timestamps from filtered_idx
    if isinstance(sent_emb, np.ndarray) and ex_emb.size > 0:
        top_pairs = top_by_similarity_preencoded(sentences, sent_emb, filtered_idx, ex_emb, top_k=3)
    else:
        top_pairs = top_matching_sentences(filtered_candidates, req.get("canonical_examples", []), model, top_k=3)

    # map sentence -> a source index in filtered_idx
    sent_to_indices = {}
    for j, s in enumerate(filtered_candidates):
        sent_to_indices.setdefault(s, []).append(filtered_idx[j])

    evidence = []
    for s, _score in top_pairs[:3]:
        src_idx_list = sent_to_indices.get(s)
        if src_idx_list:
            src_idx = src_idx_list[0]
            st, en = sentence_times[src_idx] if 0 <= src_idx < len(sentence_times) else (None, None)
        else:
            st, en = (None, None)
        evidence.append({"quote": s, "start_ms": st, "end_ms": en})

    debug = {
        "kw": kw_score_unit,
        "sim": sim,
        "sim_score": sim_score,
        "llm": llm_score,
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