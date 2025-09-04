import os
from app.policy_loader import load_policy
from app.scoring import aggregate_score
from sentence_transformers import SentenceTransformer

def test_policy_load():
    policy = load_policy("policy/policy_disclaimer.yml")
    assert "requirements" in policy and len(policy["requirements"]) >= 3

def test_scoring_basic():
    policy = load_policy("policy/policy_disclaimer.yml")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    text = "You won’t be able to accrue miles for this ticket. Do not contact the airline. Keep our contact in the booking."
    flags = {r["id"]: "unclear" for r in policy["requirements"]}
    score, verdict, breakdown, evidence = aggregate_score(text, flags, policy, model)
    assert score >= 0.2
    assert verdict in {"PASS","FAIL","NEEDS_REVIEW"}
