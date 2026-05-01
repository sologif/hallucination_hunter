import spacy
import en_core_web_sm
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch
import numpy as np

# Load models globally to avoid reloading
nlp = en_core_web_sm.load()

import os
if os.path.isdir('./models/nli_finetuned'):
    nli_model = CrossEncoder('./models/nli_finetuned')
else:
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# The standard mapping for cross-encoder/nli-deberta-v3-small
# 0: Contradiction, 1: Entailment, 2: Neutral
LABEL_MAPPING = {0: "Contradiction", 1: "Entailment", 2: "Neutral"}

def extract_claims(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]

def analyze_hallucination(source_text: str, generated_text: str):
    source_sentences = extract_claims(source_text)
    generated_claims = extract_claims(generated_text)
    
    if not source_sentences or not generated_claims:
        return {"verdict": "ERROR", "confidence_score": 0.0, "details": "Empty source or generated text.", "claims": []}
    
    # 1. Embed source sentences and claims
    source_embeddings = embedder.encode(source_sentences, convert_to_tensor=True)
    claim_embeddings = embedder.encode(generated_claims, convert_to_tensor=True)
    
    # 2. Compute similarity to find best matching source sentence for each claim
    cosine_scores = util.cos_sim(claim_embeddings, source_embeddings)
    
    results = []
    total_entailment_prob = 0.0
    num_claims = len(generated_claims)
    has_hallucination = False
    
    for i, claim in enumerate(generated_claims):
        # Find best source sentence
        best_source_idx = torch.argmax(cosine_scores[i]).item()
        best_source_sentence = source_sentences[best_source_idx]
        similarity_score = cosine_scores[i][best_source_idx].item()
        
        # 3. Strict Similarity Threshold Check
        if similarity_score < 0.35:
            # The closest source sentence is too irrelevant. Claim is unsupported.
            pred_label = "Unsupported"
            entailment_prob = 0.0
            is_hallucinated = True
        else:
            # 4. NLI (Premise=Source, Hypothesis=Claim)
            nli_logits = nli_model.predict([(best_source_sentence, claim)])[0]
            # apply softmax to get probabilities
            nli_probs = np.exp(nli_logits) / np.sum(np.exp(nli_logits))
            
            pred_label_idx = np.argmax(nli_probs)
            
            # Detect if we are using the fine-tuned 2-label model
            if len(nli_probs) == 2:
                # Fine-tuned: 0: Faithful, 1: Hallucinated
                FINETUNED_MAPPING = {0: "Entailment", 1: "Contradiction"}
                pred_label = FINETUNED_MAPPING.get(pred_label_idx, "Unknown")
                entailment_prob = nli_probs[0]
                is_hallucinated = (pred_label == "Contradiction")
            else:
                # Original: 0: Contradiction, 1: Entailment, 2: Neutral
                pred_label = LABEL_MAPPING.get(pred_label_idx, "Unknown")
                entailment_prob = nli_probs[1]
                # Treat Neutral as hallucinated because we want strict grounding
                is_hallucinated = pred_label in ["Contradiction", "Neutral"]
            
            total_entailment_prob += entailment_prob
            
        if is_hallucinated:
            has_hallucination = True
            
        results.append({
            "claim": claim,
            "best_source_sentence": best_source_sentence,
            "similarity_score": float(round(similarity_score, 3)),
            "nli_label": pred_label,
            "entailment_prob": float(round(float(entailment_prob), 3)),
            "is_hallucinated": bool(is_hallucinated)
        })
        
    avg_confidence = round((total_entailment_prob / num_claims) * 100, 2)
    overall_verdict = "HALLUCINATED" if has_hallucination else "FAITHFUL"
    
    return {
        "verdict": overall_verdict,
        "confidence_score": float(avg_confidence),
        "claims": results
    }
