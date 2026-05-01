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
        claims = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
        if not claims:
            # Fallback to simple line split if SpaCy finds no sentences
            claims = [line.strip() for line in text.split('\n') if len(line.strip()) > 5]
        return claims
    except Exception as e:
        # Emergency fallback for unexpected SpaCy errors
        return [line.strip() for line in text.split('\n') if len(line.strip()) > 5]

def analyze_hallucination(source_text: str, generated_text: str):
    nlp, nli_model, embedder = get_models()
    source_sentences = extract_claims(source_text)
    generated_claims = extract_claims(generated_text)
    
    if not source_sentences or not generated_claims:
        return {"verdict": "ERROR", "confidence_score": 0.0, "details": "Empty source or generated text.", "claims": []}
    
    # 1. Embed source sentences and claims using FastEmbed (yields numpy arrays)
    source_embeddings = np.array(list(embedder.embed(source_sentences)))
    claim_embeddings = np.array(list(embedder.embed(generated_claims)))
    
    # 2. Compute similarity matrix (Cosine Similarity = Dot Product of normalized vectors)
    # Normalize vectors for cosine similarity
    source_embeddings = source_embeddings / np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    claim_embeddings = claim_embeddings / np.linalg.norm(claim_embeddings, axis=1, keepdims=True)
    cosine_scores = np.dot(claim_embeddings, source_embeddings.T)
    
    results = []
    verified_claims_count = 0
    num_claims = len(generated_claims)
    
    # Prepare pairs for batched NLI inference to save overhead
    nli_pairs = []
    best_source_indices = []
    
    for i, claim in enumerate(generated_claims):
        best_source_idx = np.argmax(cosine_scores[i])
        best_source_indices.append(best_source_idx)
        similarity_score = cosine_scores[i][best_source_idx]
        
        if similarity_score >= 0.35:
            nli_pairs.append((source_sentences[best_source_idx], claim))
        else:
            nli_pairs.append(None) # Skip NLI for irrelevant claims

    # 3. Batched NLI Prediction
    valid_pairs = [p for p in nli_pairs if p is not None]
    if valid_pairs:
        all_nli_logits = nli_model.predict(valid_pairs, batch_size=8)
    else:
        all_nli_logits = []

    logits_idx = 0
    for i, claim in enumerate(generated_claims):
        best_source_idx = best_source_indices[i]
        best_source_sentence = source_sentences[best_source_idx]
        similarity_score = cosine_scores[i][best_source_idx]
        
        if nli_pairs[i] is None:
            pred_label = "Unsupported"
            entailment_prob = 0.0
            is_hallucinated = True
        else:
            nli_logits = all_nli_logits[logits_idx]
            logits_idx += 1
            
            # softmax
            nli_probs = np.exp(nli_logits) / np.sum(np.exp(nli_logits))
            pred_label_idx = np.argmax(nli_probs)
            
            if len(nli_probs) == 2:
                FINETUNED_MAPPING = {0: "Entailment", 1: "Contradiction"}
                pred_label = FINETUNED_MAPPING.get(pred_label_idx, "Unknown")
                entailment_prob = nli_probs[0]
                is_hallucinated = (pred_label == "Contradiction")
            else:
                pred_label = LABEL_MAPPING.get(pred_label_idx, "Unknown")
                entailment_prob = nli_probs[1]
                is_hallucinated = pred_label in ["Contradiction", "Neutral"]
            
        if not is_hallucinated:
            verified_claims_count += 1
            
        results.append({
            "claim": claim,
            "best_source_sentence": best_source_sentence,
            "similarity_score": float(round(float(similarity_score), 3)),
            "nli_label": pred_label,
            "entailment_prob": float(round(float(entailment_prob), 3)),
            "is_hallucinated": bool(is_hallucinated)
        })
    
    # Force garbage collection to free up memory after large analysis
    gc.collect()
        
    verified_claims_ratio = round((verified_claims_count / num_claims) * 100, 2)
    
    # Calculate hallucination rate
    hallucination_rate = (num_claims - verified_claims_count) / num_claims
    
    # Switch to a threshold ratio (e.g., > 30% of claims hallucinated) to avoid over-flagging
    # and improve Balanced Accuracy on benchmarking datasets.
    overall_verdict = "HALLUCINATED" if hallucination_rate > 0.3 else "FAITHFUL"
    
    return {
        "verdict": overall_verdict,
        "confidence_score": float(verified_claims_ratio),
        "verified_claims": verified_claims_count,
        "total_claims": num_claims,
        "claims": results,
        "cosine_scores": cosine_scores.tolist(), # Convert to list for JSON/UI serialization
        "source_sentences": source_sentences,
        "generated_claims": generated_claims
    }
