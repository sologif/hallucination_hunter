import spacy
from sentence_transformers import CrossEncoder
from fastembed import TextEmbedding
import torch
import numpy as np
import gc

# Global model holders
_nlp = None
_nli_model = None
_embedder = None

def get_models():
    global _nlp, _nli_model, _embedder
    if _nlp is None:
        # Load only the senter to save ~50MB RAM
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer", "textcat"])
        _nlp.add_pipe("sentencizer")
    if _nli_model is None:
        import os
        model_path = './models/nli_finetuned'
        if os.path.exists(os.path.join(model_path, 'config.json')):
            try:
                _nli_model = CrossEncoder(model_path)
            except Exception:
                _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
        else:
            # Use 'small' instead of 'base' to save memory on Streamlit Cloud
            _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    if _embedder is None:
        # FastEmbed uses ONNX runtime, which is much lighter than PyTorch
        _embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _nlp, _nli_model, _embedder

# The standard mapping for cross-encoder/nli-deberta-v3-small
# 0: Contradiction, 1: Entailment, 2: Neutral
LABEL_MAPPING = {0: "Contradiction", 1: "Entailment", 2: "Neutral"}

def extract_claims(text: str):
    nlp, _, _ = get_models()
    try:
        doc = nlp(text)
        junk_keywords = ["cookie", "privacy policy", "all rights reserved", "subscribe now", "sign up", "click here", "javascript", "browser"]
        
        claims = []
        for sent in doc.sents:
            txt = sent.text.strip()
            if len(txt) < 15:
                continue
            if any(k in txt.lower() for k in junk_keywords):
                continue
            claims.append(txt)
            
        if not claims:
            claims = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
        return claims
    except Exception:
        return [line.strip() for line in text.split('\n') if len(line.strip()) > 5]

def analyze_hallucination(source_text: str, generated_text: str):
    nlp, nli_model, embedder = get_models()
    source_sentences = extract_claims(source_text)
    generated_claims = extract_claims(generated_text)
    
    if not source_sentences or not generated_claims:
        return {"verdict": "ERROR", "confidence_score": 0.0, "details": "Empty source or generated text.", "claims": []}
    
    # 1. Embed source sentences and claims
    source_embeddings = np.array(list(embedder.embed(source_sentences)))
    claim_embeddings = np.array(list(embedder.embed(generated_claims)))
    
    source_embeddings = source_embeddings / np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    claim_embeddings = claim_embeddings / np.linalg.norm(claim_embeddings, axis=1, keepdims=True)
    cosine_scores = np.dot(claim_embeddings, source_embeddings.T)
    
    results = []
    num_claims = len(generated_claims)
    
    SIMILARITY_THRESHOLD = 0.25
    TOP_K = min(3, len(source_sentences))
    
    nli_pairs = []
    claim_to_pair_indices = []
    pair_metadata = []
    
    for i, claim in enumerate(generated_claims):
        top_indices = np.argsort(cosine_scores[i])[-TOP_K:][::-1]
        current_claim_pairs = []
        has_relevant_source = False
        for idx in top_indices:
            similarity = cosine_scores[i][idx]
            if similarity >= SIMILARITY_THRESHOLD:
                current_claim_pairs.append(len(nli_pairs))
                nli_pairs.append((source_sentences[idx], claim))
                pair_metadata.append((idx, similarity))
                has_relevant_source = True
        
        if not has_relevant_source:
            current_claim_pairs.append(len(nli_pairs))
            nli_pairs.append(None)
            pair_metadata.append((top_indices[0], cosine_scores[i][top_indices[0]]))
        claim_to_pair_indices.append(current_claim_pairs)

    # 3. Batched NLI Prediction
    valid_indices = [idx for idx, p in enumerate(nli_pairs) if p is not None]
    valid_pairs = [nli_pairs[idx] for idx in valid_indices]
    
    pair_results = [None] * len(nli_pairs)
    if valid_pairs:
        all_nli_logits = nli_model.predict(valid_pairs, batch_size=16)
        for i, idx in enumerate(valid_indices):
            logits = all_nli_logits[i]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            label_idx = np.argmax(probs)
            
            if len(probs) == 2:
                mapping = {0: "Entailment", 1: "Contradiction"}
                label = mapping.get(label_idx, "Unknown")
                ent_prob = probs[0]
                is_hallucinated = (label == "Contradiction")
            else:
                label = LABEL_MAPPING.get(label_idx, "Unknown")
                ent_prob = probs[1]
                is_hallucinated = (label in ["Contradiction", "Neutral"])
                
            pair_results[idx] = {
                "label": label,
                "prob": ent_prob,
                "is_hallucinated": is_hallucinated
            }

    # 4. Aggregate by claim
    contradiction_count = 0
    neutral_count = 0
    entailment_count = 0
    unsupported_count = 0
    
    for i, claim in enumerate(generated_claims):
        best_res = None
        highest_entailment = -1.0
        
        for pair_idx in claim_to_pair_indices[i]:
            source_idx, sim = pair_metadata[pair_idx]
            nli_res = pair_results[pair_idx]
            
            if nli_res is None:
                current_res = {
                    "claim": claim,
                    "best_source_sentence": source_sentences[source_idx],
                    "similarity_score": float(round(float(sim), 3)),
                    "nli_label": "Unsupported",
                    "entailment_prob": 0.0,
                    "is_hallucinated": True
                }
            else:
                current_res = {
                    "claim": claim,
                    "best_source_sentence": source_sentences[source_idx],
                    "similarity_score": float(round(float(sim), 3)),
                    "nli_label": nli_res["label"],
                    "entailment_prob": float(round(float(nli_res["prob"]), 3)),
                    "is_hallucinated": nli_res["is_hallucinated"]
                }
            
            # Prioritize Entailment
            if not current_res["is_hallucinated"] and current_res["nli_label"] == "Entailment":
                if current_res["entailment_prob"] > highest_entailment:
                    highest_entailment = current_res["entailment_prob"]
                    best_res = current_res
            elif best_res is None or (best_res["is_hallucinated"] and not current_res["is_hallucinated"]):
                best_res = current_res
        
        if best_res["nli_label"] == "Entailment":
            entailment_count += 1
        elif best_res["nli_label"] == "Contradiction":
            contradiction_count += 1
        elif best_res["nli_label"] == "Unsupported":
            unsupported_count += 1
        else:
            neutral_count += 1
            
        results.append(best_res)

    gc.collect()
    
    # 5. Nuanced Verdict Logic
    # HALLUCINATED: Hard contradictions found (> 10% of claims)
    # WARNING: No contradictions, but many unsupported/neutral claims (> 30%)
    # FAITHFUL: Mostly entailed claims.
    
    contradiction_rate = contradiction_count / num_claims
    unverified_rate = (neutral_count + unsupported_count) / num_claims
    
    if contradiction_rate > 0.1:
        overall_verdict = "HALLUCINATED"
    elif unverified_rate > 0.3:
        overall_verdict = "WARNING"
    else:
        overall_verdict = "FAITHFUL"
        
    verified_claims_ratio = round((entailment_count / num_claims) * 100, 2)
    
    return {
        "verdict": overall_verdict,
        "confidence_score": float(verified_claims_ratio),
        "verified_claims": entailment_count,
        "total_claims": num_claims,
        "claims": results,
        "source_sentences": source_sentences,
        "generated_claims": generated_claims
    }
