import spacy
from sentence_transformers import CrossEncoder
from fastembed import TextEmbedding
import torch
import numpy as np
import gc
import re

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
        junk_keywords = [
            "cookie", "privacy policy", "all rights reserved", "subscribe now", 
            "sign up", "click here", "javascript", "browser", "login", "register",
            "terms of use", "contact us", "site map", "navigation", "read more",
            "warranty", "representation", "liability", "disclaimer", "copyright"
        ]
        
        claims = []
        for sent in doc.sents:
            txt = sent.text.strip()
            # Length filter: too short or suspiciously long (often scraped junk)
            if len(txt) < 20 or len(txt) > 500:
                continue
                
            # Pattern filter: sentences starting with common UI navigation words
            lower_txt = txt.lower()
            if any(lower_txt.startswith(k) for k in ["home", "about", "contact", "search", "menu"]):
                continue
                
            if any(k in lower_txt for k in junk_keywords):
                continue
                
            # Filter out sentences that are just lists of links (low verb/noun ratio)
            # (Simple heuristic: if it has very few spaces compared to its length)
            if len(txt) > 50 and txt.count(' ') < len(txt) / 10:
                continue
                
            claims.append(txt)
            
        if not claims:
            claims = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
        return claims
    except Exception:
        return [line.strip() for line in text.split('\n') if len(line.strip()) > 5]

def get_keyword_overlap(text1: str, text2: str):
    """
    Calculate keyword overlap between two texts.
    Returns a score between 0 and 1.
    """
    def tokenize(text):
        # Lowercase, remove non-alphanumeric, split, and filter short words
        words = re.findall(r'\w+', text.lower())
        return set(w for w in words if len(w) > 2)
    
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    # Jaccard-like similarity but biased towards the smaller set (query/claim)
    return len(intersection) / len(words1) if words1 else 0.0

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
        # Combine semantic similarity with keyword overlap
        combined_scores = []
        for idx, src_sent in enumerate(source_sentences):
            sem_sim = cosine_scores[i][idx]
            key_overlap = get_keyword_overlap(claim, src_sent)
            # Weighted score: Semantic is great for meaning, but keyword is a hard requirement for factual grounding
            # If keyword overlap is 0, we penalize the semantic score significantly
            hybrid_score = (sem_sim * 0.7) + (key_overlap * 0.3)
            if key_overlap == 0:
                hybrid_score *= 0.5
            combined_scores.append(hybrid_score)
        
        combined_scores = np.array(combined_scores)
        top_indices = np.argsort(combined_scores)[-TOP_K:][::-1]
        
        current_claim_pairs = []
        has_relevant_source = False
        for idx in top_indices:
            score = combined_scores[idx]
            # Use a slightly more relaxed threshold for hybrid score but still strict
            if score >= 0.25: 
                current_claim_pairs.append(len(nli_pairs))
                nli_pairs.append((source_sentences[idx], claim))
                pair_metadata.append((idx, float(cosine_scores[i][idx]))) # Still store raw cosine for UI
                has_relevant_source = True
        
        if not has_relevant_source:
            current_claim_pairs.append(len(nli_pairs))
            nli_pairs.append(None)
            # Store the best semantic match for the "Unsupported" case
            best_sem_idx = np.argmax(cosine_scores[i])
            pair_metadata.append((best_sem_idx, float(cosine_scores[i][best_sem_idx])))
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
                is_hallucinated = (label == "Contradiction")
                
            pair_results[idx] = {
                "label": label,
                "prob": ent_prob,
                "is_hallucinated": is_hallucinated
            }

    # 4. Aggregate by claim and build alignment matrix
    alignment_matrix = []
    
    for i, claim in enumerate(generated_claims):
        claim_row = []
        best_res = None
        highest_entailment = -1.0
        
        # Build the full row for this claim against ALL source sentences
        for j, src_sent in enumerate(source_sentences):
            # Find the pair index in nli_pairs
            pair_idx = -1
            for p_idx in claim_to_pair_indices[i]:
                if pair_metadata[p_idx][0] == j:
                    pair_idx = p_idx
                    break
            
            if pair_idx != -1 and pair_results[pair_idx] is not None:
                nli_res = pair_results[pair_idx]
                cell_data = {
                    "label": nli_res["label"],
                    "prob": float(round(float(nli_res["prob"]), 3)),
                    "is_hallucinated": nli_res["is_hallucinated"]
                }
            else:
                # If not in top-K or no NLI result, treat as Neutral/Unsupported
                # (We still want a score for the heatmap)
                cell_data = {
                    "label": "Neutral",
                    "prob": 0.0,
                    "is_hallucinated": False
                }
            claim_row.append(cell_data)

            # Logic for selecting the "Best Source" (Highest Entailment)
            if pair_idx != -1 and pair_results[pair_idx] is not None:
                current_res = {
                    "claim": claim,
                    "best_source_sentence": source_sentences[j],
                    "similarity_score": float(round(float(cosine_scores[i][j]), 3)),
                    "nli_label": nli_res["label"],
                    "entailment_prob": float(round(float(nli_res["prob"]), 3)),
                    "is_hallucinated": nli_res["is_hallucinated"]
                }
                
                if current_res["nli_label"] == "Entailment":
                    if current_res["entailment_prob"] > highest_entailment:
                        highest_entailment = current_res["entailment_prob"]
                        best_res = current_res
                elif best_res is None or (best_res["is_hallucinated"] and not current_res["is_hallucinated"]):
                    best_res = current_res
        
        # Fallback if no good match found in top-K
        if best_res is None:
            best_idx = np.argmax(cosine_scores[i])
            best_res = {
                "claim": claim,
                "best_source_sentence": source_sentences[best_idx],
                "similarity_score": float(round(float(cosine_scores[i][best_idx]), 3)),
                "nli_label": "Unsupported",
                "entailment_prob": 0.0,
                "is_hallucinated": True
            }
            
        if best_res["nli_label"] == "Entailment":
            entailment_count += 1
        elif best_res["nli_label"] == "Contradiction":
            contradiction_count += 1
        elif best_res["nli_label"] == "Unsupported":
            unsupported_count += 1
        else:
            neutral_count += 1
            
        results.append(best_res)
        alignment_matrix.append(claim_row)

    gc.collect()
    
    # 5. Nuanced Verdict Logic
    if contradiction_count > 0:
        overall_verdict = "HALLUCINATED"
    elif neutral_count > 0 or unsupported_count > 0:
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
        "generated_claims": generated_claims,
        "alignment_matrix": alignment_matrix
    }

