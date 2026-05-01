# hallucination_hunter
AI-powered hallucination detection system using RAG + NLI to verify LLM-generated responses with claim-level reasoning and confidence scoring.

### 🚀 Authentication
The system supports streamlined access for enterprise users:
- **Login with Google**: Integrated SSO for secure access.
- **Login as Guest**: Instant access to the verification engine for rapid testing.

### 🧠 Model & Fine-tuning
The system uses a **DeBERTa-v3 Small** cross-encoder as its core NLI engine.
- **Fine-tuned on**: [HaluEval](https://github.com/RUCAIBox/HaluEval) (Summarization, QA, and Dialogue subsets).
- **Purpose**: Enhanced detection of factual contradictions and hallucinations in AI-generated text.
- **Training Script**: `scripts/train_nli_finetune.py`
- **Dataset Source**: Hugging Face `pminervini/HaluEval`

