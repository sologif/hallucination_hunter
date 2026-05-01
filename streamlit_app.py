import streamlit as st
from engine import analyze_hallucination, get_models
from rag.vector_db import db as vector_db
from rag.generator import generate_answer
from rag.web_search import search_web
import html
import os
import json
import pandas as pd
import plotly.express as px
from datasets import load_dataset
import numpy as np
import gc

# Optimization: Cache models and resources to stay within Streamlit Cloud memory limits
@st.cache_resource
def load_resources():
    models = get_models()
    return models, vector_db

# Trigger resource loading
resources, db_instance = load_resources()

# Handle Hugging Face Token for higher rate limits
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
elif "hf_token" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["hf_token"]

st.set_page_config(
    page_title="Hallucination Hunter",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Custom CSS to force Streamlit to look like the premium HTML/CSS design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;600&display=swap');
    
    :root {
        --bg-dark: #0a0f1c;
        --panel-bg: rgba(18, 24, 43, 0.6);
        --panel-border: rgba(255, 255, 255, 0.08);
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --accent-primary: #00f2fe;
        --accent-secondary: #4facfe;
        --success: #10b981;
        --success-bg: rgba(16, 185, 129, 0.1);
        --danger: #ef4444;
        --danger-bg: rgba(239, 68, 68, 0.1);
        --warning: #f59e0b;
        --warning-bg: rgba(245, 158, 11, 0.1);
        --shadow-glow: 0 0 40px rgba(79, 172, 254, 0.15);
    }
    
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #111a30 0%, var(--bg-dark) 70%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide top bar */
    header {visibility: hidden;}
    
    /* Header Styles */
    .main-title {
        font-family: 'Outfit', sans-serif;
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 10px;
        color: white;
        background: linear-gradient(135deg, #fff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 2rem;
    }
    .main-title span {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.2rem;
        max-width: 700px;
        margin: 0 auto 40px auto;
        line-height: 1.6;
    }
    
    /* Inputs Styling (Text Area & Text Inputs) */
    div[data-testid="stTextInput"] > div > div {
        background-color: rgba(0, 0, 0, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px;
    }
    div[data-testid="stTextInput"] label p {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    
    div[data-testid="stTextArea"] {
        background: var(--panel-bg);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        box-shadow: var(--shadow-glow);
        padding: 2.5rem;
        margin-bottom: 0;
    }
    div[data-testid="stTextArea"] label p {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 15px;
    }
    div[data-testid="stTextArea"] textarea {
        background-color: rgba(0, 0, 0, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px;
        padding: 15px;
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent-secondary) !important;
        box-shadow: 0 0 0 4px rgba(79, 172, 254, 0.1) !important;
        background-color: rgba(0, 0, 0, 0.3) !important;
    }

    /* Primary Button Styling */
    div[data-testid="stButton"] {
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, var(--accent-secondary), var(--accent-primary));
        color: white;
        border: none;
        border-radius: 14px;
        padding: 1rem 3rem;
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 10px 20px -10px var(--accent-secondary);
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 25px -10px var(--accent-secondary);
        color: white;
        border: none;
    }
    
    /* Tabs Styling */
    div[data-testid="stTabs"] button {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* HTML Elements injected via markdown */
    .glass-panel {
        background: var(--panel-bg);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        box-shadow: var(--shadow-glow);
        padding: 2.5rem;
        margin-bottom: 2rem;
    }
    
    /* Verdict Card */
    .verdict-card {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        position: relative;
        overflow: hidden;
        border-radius: 24px;
        border: 1px solid var(--panel-border);
        margin-bottom: 2rem;
        margin-top: 2rem;
    }
    .verdict-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
    }
    .verdict-card.faithful::before { background: var(--success); }
    .verdict-card.hallucinated::before { background: var(--danger); }
    .verdict-card.warning::before { background: var(--warning); }
    
    .verdict-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .verdict-faithful { color: var(--success); text-shadow: 0 0 30px rgba(16, 185, 129, 0.4); }
    .verdict-hallucinated { color: var(--danger); text-shadow: 0 0 30px rgba(239, 68, 68, 0.4); }
    .verdict-warning { color: var(--warning); text-shadow: 0 0 30px rgba(245, 158, 11, 0.4); }
    
    .score-container {
        font-size: 1.25rem;
        color: var(--text-secondary);
    }
    .score {
        font-weight: 800;
        color: var(--text-primary);
        font-size: 1.5rem;
    }
    
    /* Content Cards */
    .content-card h3 {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: white;
    }
    .content-card h3::before {
        content: '';
        display: inline-block;
        width: 8px;
        height: 24px;
        background: var(--accent-primary);
        border-radius: 4px;
    }
    .answer-text {
        font-size: 1.15rem;
        line-height: 1.8;
        color: #f8fafc;
    }

    /* Highlights for Sentences */
    .highlight-faithful {
        background-color: rgba(16, 185, 129, 0.2);
        border-bottom: 2px solid var(--success);
        padding: 0 4px;
        border-radius: 4px;
        cursor: help;
        transition: all 0.3s ease;
    }
    .highlight-faithful:hover { background-color: rgba(16, 185, 129, 0.3); }

    .highlight-contradiction {
        background-color: rgba(239, 68, 68, 0.25);
        border-bottom: 2px solid var(--danger);
        padding: 0 4px;
        border-radius: 4px;
        cursor: help;
        transition: all 0.3s ease;
    }
    .highlight-contradiction:hover { background-color: rgba(239, 68, 68, 0.4); }

    .highlight-neutral {
        background-color: rgba(245, 158, 11, 0.2);
        border-bottom: 2px solid var(--warning);
        padding: 0 4px;
        border-radius: 4px;
        cursor: help;
        transition: all 0.3s ease;
    }
    .highlight-neutral:hover { background-color: rgba(245, 158, 11, 0.3); }

    .highlight-unsupported {
        background-color: rgba(148, 163, 184, 0.15);
        border-bottom: 2px dashed var(--text-secondary);
        padding: 0 4px;
        border-radius: 4px;
        cursor: help;
        transition: all 0.3s ease;
    }
    .highlight-unsupported:hover { background-color: rgba(148, 163, 184, 0.25); }
    
    /* Sources */
    .source-list { list-style: none; padding: 0; margin: 0; }
    .source-item {
        padding: 1rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        margin-bottom: 1rem;
        position: relative;
        padding-left: 2rem;
        color: #cbd5e1;
        line-height: 1.6;
    }
    .source-item::before {
        content: '•';
        position: absolute;
        left: 0.8rem;
        color: var(--accent-secondary);
        font-size: 1.5rem;
        top: 0.8rem;
        line-height: 1;
    }
    .source-item strong { color: white; display: inline-block; margin-bottom: 0.2rem; }

    /* Claims */
    .analysis-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        margin-top: 3rem;
    }
    .claim-card {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid var(--panel-border);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .claim-card.faithful { border-left: 6px solid var(--success); }
    .claim-card.hallucinated { border-left: 6px solid var(--danger); }
    .claim-card.neutral { border-left: 6px solid var(--warning); }
    .claim-card.unsupported { border-left: 6px solid var(--text-secondary); }
    
    .claim-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.5rem;
    }
    .claim-text {
        font-size: 1.25rem;
        font-weight: 500;
        line-height: 1.5;
        flex: 1;
        padding-right: 1.5rem;
        color: white;
    }
    .badge {
        padding: 0.4rem 1rem;
        border-radius: 99px;
        font-size: 0.85rem;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        white-space: nowrap;
    }
    .badge-faithful { background: var(--success-bg); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.2); }
    .badge-hallucinated { background: var(--danger-bg); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.2); }
    .badge-neutral { background: var(--warning-bg); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.2); }
    .badge-unsupported { background: rgba(148, 163, 184, 0.1); color: var(--text-secondary); border: 1px solid rgba(148, 163, 184, 0.2); }
    
    .source-match {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .source-match strong {
        color: var(--text-secondary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: block;
        margin-bottom: 0.5rem;
    }
    .source-match p {
        margin: 0;
        font-style: italic;
        color: #cbd5e1;
    }
    
    .metrics-bar {
        display: flex;
        gap: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .metric { display: flex; flex-direction: column; }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        color: white;
    }
    
    /* Remove default streamlit container padding for cleaner look */
    div[data-testid="stVerticalBlock"] {
        gap: 0;
    }
    </style>
""", unsafe_allow_html=True)

if not st.session_state['logged_in']:
    # Login Page Layout
    st.markdown('<div class="main-title" style="margin-top:5rem;">Hallucination <span>Hunter</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="login-header" style="margin-top:2rem;"><h2>Welcome Back</h2><p>Sign in to access Enterprise-grade AI verification.</p></div>', unsafe_allow_html=True)
    
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    
    if st.button("Login"):
        if username == "admin" and password == "Domaiyn labs":
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Invalid username or password")
            
else:
    # Main Application
    st.markdown('<div class="main-title">Hallucination <span>Hunter</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enterprise-grade AI verification. Grounded against factual sources to eliminate hallucinations using Natural Language Inference.</div>', unsafe_allow_html=True)
    
    # Sidebar for HaluEval Playground and settings
    with st.sidebar:
        st.markdown('<div class="main-title" style="font-size:1.5rem;">HaluEval <span>Playground</span></div>', unsafe_allow_html=True)
        st.write("Automatically load benchmark samples to test the model.")
        
        if st.button("🎲 Load Random HaluEval Sample"):
            try:
                # Use datasets library to load sample (Cloud compatible)
                import random
                ds = load_dataset("pminervini/HaluEval", "summarization", split="data", streaming=True)
                # Proper shuffle and larger skip range
                ds = ds.shuffle(seed=random.randint(0, 1000000), buffer_size=5000)
                skip = random.randint(0, 2000)
                sample = None
                for i, s in enumerate(ds):
                    if i == skip:
                        sample = s
                        break
                
                if sample:
                    # Update session state keys directly to ensure UI reflects changes
                    st.session_state["verify_input"] = sample["hallucinated_summary"]
                    st.session_state["hidden_ground_truth"] = sample["document"]
                    st.success("Loaded HaluEval Sample!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {e}")
        
        st.markdown("---")
        if st.button("🚪 Log out"):
            st.session_state['logged_in'] = False
            st.rerun()

    # Add a tiny logout button at top right (redundant but keeping for UI consistency)
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        pass # Using sidebar now

    def render_results(verification_result, answer_text, sources_list):
        verdict = verification_result["verdict"]
        score = verification_result.get("confidence_score", 0.0)
        
        if verdict == "ERROR":
            st.error(f"Analysis Error: {verification_result.get('details', 'Unknown error')}")
            return

        # Overall Verdict
        if verdict == "FAITHFUL":
            verdict_class = "faithful"
        elif verdict == "WARNING":
            verdict_class = "warning"
        else:
            verdict_class = "hallucinated"
        
        # Re-frame confidence score: If hallucinated, show risk.
        display_score = score
        score_label = "Verified"
        if verdict == "HALLUCINATED":
            display_score = 100 - score
            score_label = "Hallucination Risk"

        st.markdown(f"""
        <div class="verdict-card {verdict_class}">
            <div class="verdict-title verdict-{verdict_class}">{verdict}</div>
            <div class="score-container">
                <span class="score">{verification_result.get('verified_claims', 0)}/{verification_result.get('total_claims', 0)}</span> claims supported
            </div>
            <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">{display_score:.1f}% {score_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Highlight Sentences within the generated answer
        highlighted_text = html.escape(answer_text)
        for claim in verification_result["claims"]:
            nli_label = claim["nli_label"]
            escaped_claim = html.escape(claim["claim"])
            
            if nli_label == "Entailment":
                span_class = "highlight-faithful"
            elif nli_label == "Contradiction":
                span_class = "highlight-contradiction"
            elif nli_label == "Unsupported":
                span_class = "highlight-unsupported"
            else:
                span_class = "highlight-neutral"
                
            # Replace the exact claim string with a highlighted span
            highlighted_span = f'<span class="{span_class}" title="{nli_label}">{escaped_claim}</span>'
            highlighted_text = highlighted_text.replace(escaped_claim, highlighted_span)
        
        # Generated Answer Card with Highlights
        st.markdown(f"""
        <div class="glass-panel content-card">
            <h3>Evaluated Text <span style="font-size:0.9rem; color:#94a3b8; font-weight:400; margin-left:1rem;">(Hover over highlighted sentences for analysis)</span></h3>
            <div class="answer-text">{highlighted_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sources Card
        sources_html = ""
        for i, s in enumerate(sources_list):
            title = s.get("title", f"Source {i+1}")
            url = s.get("url", "")
            text = s.get("text", "")
            source_type = s.get("source", "local").upper()
            
            # Badge for source type
            type_color = "var(--accent-primary)" if source_type == "WEB" else "var(--success)"
            type_badge = f'<span style="background:{type_color}22; color:{type_color}; border:1px solid {type_color}44; padding:2px 8px; border-radius:4px; font-size:0.7rem; margin-right:10px; font-weight:bold;">{source_type}</span>'
            
            link_html = f'<a href="{url}" target="_blank" style="color:var(--accent-primary); text-decoration:none; font-size:0.8rem; margin-left:10px;">[Visit Source]</a>' if url and url != "#" else ""
            sources_html += f'<li class="source-item">{type_badge}<strong>{html.escape(title)}:</strong>{link_html}<br>{html.escape(text)}</li>'
        if not sources_html:
            sources_html = '<li class="source-item">No relevant sources found in the database.</li>'
            
        st.markdown(f"""
        <div class="glass-panel content-card">
            <h3>Retrieved Ground Truth Sources</h3>
            <ul class="source-list">
                {sources_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Detailed Analysis
        st.markdown('<div class="analysis-title">Sentence-Level Validation</div>', unsafe_allow_html=True)
        
        claims_html = ""
        for claim in verification_result["claims"]:
            nli_label = claim["nli_label"]
            similarity = claim["similarity_score"]
            
            if nli_label == "Entailment":
                status_class = "faithful"
                badge_class = "badge-faithful"
                badge_text = "Verified"
            elif nli_label == "Contradiction":
                status_class = "hallucinated"
                badge_class = "badge-hallucinated"
                badge_text = "Hallucinated"
            elif nli_label == "Unsupported":
                status_class = "unsupported"
                badge_class = "badge-unsupported"
                badge_text = "Unsupported"
            else:
                status_class = "neutral"
                badge_class = "badge-neutral"
                badge_text = "Unverified"
                
            entailment_prob = claim["entailment_prob"] * 100
            
            claims_html += f"""
            <div class="claim-card {status_class}">
                <div class="claim-header">
                    <div class="claim-text">"{html.escape(claim['claim'])}"</div>
                    <div class="badge {badge_class}">{badge_text}</div>
                </div>
                <div class="source-match">
                    <strong>Closest Ground Truth Match</strong>
                    <p>"{html.escape(claim['best_source_sentence']) if similarity >= 0.2 else 'No direct semantic match found in the provided sources.'}"</p>
                </div>
                <div class="metrics-bar">
                    <div class="metric">
                        <span class="metric-label">Entailment Prob</span>
                        <span class="metric-value">{entailment_prob:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Similarity</span>
                        <span class="metric-value">{similarity:.2f}</span>
                    </div>
                </div>
            </div>
            """
        st.markdown(claims_html, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Verify Pasted Text", "HaluEval Benchmark"])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Directly paste an AI-generated answer here. You can also provide the ground truth text to verify against.")
        
        # New: Direct button for those who can't see the sidebar
        if st.button("🎲 Load Random HaluEval Sample", key="tab_sample_loader"):
            try:
                import random
                ds = load_dataset("pminervini/HaluEval", "summarization", split="data", streaming=True)
                ds = ds.shuffle(seed=random.randint(0, 1000000), buffer_size=5000)
                skip = random.randint(0, 2000)
                sample = None
                for i, s in enumerate(ds):
                    if i == skip:
                        sample = s
                        break
                if sample:
                    # Update session state keys directly to ensure UI reflects changes
                    st.session_state["verify_input"] = sample["hallucinated_summary"]
                    st.session_state["hidden_ground_truth"] = sample["document"]
                    st.success("Loaded HaluEval Sample!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading sample: {e}")

        # Handle inputs from session state (HaluEval Sample)
        pasted_text = st.text_area("Paste text to verify", height=150, key="verify_input")
        
        col1, col2 = st.columns(2)
        with col1:
            use_web_verify = st.toggle("Enable Web Search Verification", value=True, help="If enabled, we will search the live web for ground truth if no manual source is provided.")
        with col2:
            use_hidden_truth = False
            if "hidden_ground_truth" in st.session_state:
                use_hidden_truth = st.toggle("Use Loaded HaluEval Ground Truth", value=True, key="use_hidden_truth")

        custom_ground_truth = ""
        if not use_web_verify and not use_hidden_truth:
            custom_ground_truth = st.text_area("Custom Ground Truth (Optional)", placeholder="Paste the source text/knowledge here to verify against.", height=150, key="custom_source")

        if st.button("Verify Pasted Text"):
            with st.spinner("Validating claims..."):
                source_passage = ""
                sources = []
                
                if use_hidden_truth and "hidden_ground_truth" in st.session_state:
                    source_passage = st.session_state.hidden_ground_truth
                    sources = [{"title": "HaluEval Hidden Document", "text": source_passage, "url": "#", "source": "HaluEval"}]
                elif custom_ground_truth.strip():
                    source_passage = custom_ground_truth
                    sources = [{"title": "Manual Source", "text": custom_ground_truth, "url": "#", "source": "manual"}]
                else:
                    # Perform Local DB Search
                    local_results = db_instance.search(pasted_text, limit=3)
                    local_sources = [{"title": "Local Database Match", "text": r["text"], "url": "#", "source": "local"} for r in local_results]
                    
                    # Auto-fallback or manual web search
                    max_local_score = max([r["score"] for r in local_results]) if local_results else 0
                    should_use_web = use_web_verify or (max_local_score < 1.0)
                    
                    if should_use_web:
                        st.write("🔍 Searching live web for grounded evidence...")
                        search_query = pasted_text[:150] if len(pasted_text) > 150 else pasted_text
                        web_sources = search_web(search_query, limit=5)
                        
                        # Fallback for Web: Try even shorter query if empty
                        if not web_sources and len(search_query) > 60:
                            web_sources = search_web(search_query[:60], limit=5)
                        
                        sources = local_sources + web_sources
                    else:
                        sources = local_sources
                    
                    source_passage = " ".join([s["text"] for s in sources])
                
                if not sources:
                    st.error("🕵️ No sources found. Try simplifying the text or providing a manual Ground Truth.")
                else:
                    verification_result = analyze_hallucination(source_passage, pasted_text)
                    render_results(verification_result, pasted_text, sources)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="analysis-title">HaluEval Benchmarking Dashboard</div>', unsafe_allow_html=True)
        st.write("Evaluate the engine's performance against the industry-standard HaluEval dataset.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            dataset_option = st.selectbox("Select Benchmark Dataset", 
                                        ["Summarization (HaluEval)", "QA (HaluEval)", "Dialogue (HaluEval)", "TRUE Benchmark (google-research/true)"], 
                                        index=0)
        with col_b:
            sample_count = st.slider("Sample Size", 20, 200, 50)
            
        # Mapping for HaluEval subsets and TRUE
        mappings = {
            "Summarization (HaluEval)": {"subset": "summarization", "knowledge": "document", "faithful": "right_summary", "hallucinated": "hallucinated_summary", "source": "pminervini/HaluEval"},
            "QA (HaluEval)": {"subset": "qa", "knowledge": "knowledge", "faithful": "right_answer", "hallucinated": "hallucinated_answer", "source": "pminervini/HaluEval"},
            "Dialogue (HaluEval)": {"subset": "dialogue", "knowledge": "knowledge", "faithful": "right_response", "hallucinated": "hallucinated_response", "source": "pminervini/HaluEval"},
            "TRUE Benchmark (google-research/true)": {"subset": "qags_cnndm", "knowledge": "premise", "faithful": "hypothesis", "hallucinated": None, "source": "google-research/true"}
        }
            
        if st.button("Run Benchmark Run"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Explicitly clear cache/memory before run
            gc.collect()
            
            results_data = []
            y_true = []
            y_pred = []
            
            m = mappings[dataset_option]
            
            try:
                status_text.text(f"Loading {dataset_option} dataset from Hugging Face...")
                dataset = load_dataset(m["source"], m["subset"], split="data" if "HaluEval" in m["source"] else "train", streaming=True)
                
                # Shuffle the dataset to avoid bias
                import random
                dataset = dataset.shuffle(seed=random.randint(0, 1000000), buffer_size=5000)
                
                # Take only the requested sample count
                data = []
                for i, entry in enumerate(dataset):
                    if i >= sample_count:
                        break
                    data.append(entry)
                
                for i, item in enumerate(data):
                    status_text.text(f"Evaluating sample {i+1}/{len(data)}...")
                    progress_bar.progress((i + 1) / len(data))
                    
                    knowledge = item.get(m["knowledge"], "")
                    
                    if "TRUE" in dataset_option:
                        # TRUE format: premise, hypothesis, label (1=Entailment, 0=Not)
                        hypothesis = item.get(m["faithful"], "")
                        label = item.get("label", 1) # 1 in TRUE is Entailment (Faithful)
                        
                        res = analyze_hallucination(knowledge, hypothesis)
                        # Map labels to binary: 0=Faithful, 1=Hallucinated
                        ground_truth = 0 if label == 1 else 1
                        prediction = 1 if res["verdict"] == "HALLUCINATED" else 0
                        
                        y_true.append(ground_truth)
                        y_pred.append(prediction)
                        
                        type_str = "Faithful" if ground_truth == 0 else "Hallucinated"
                        results_data.append({"Type": type_str, "Prediction": res["verdict"], "Correct": ground_truth == prediction})
                    else:
                        # HaluEval format: document, right_summary, hallucinated_summary
                        faithful_ans = item.get(m["faithful"], "")
                        hallucinated_ans = item.get(m["hallucinated"], "")
                        
                        if knowledge:
                            # Test Faithful
                            res_f = analyze_hallucination(knowledge, faithful_ans)
                            y_true.append(0) # 0 = Faithful
                            y_pred.append(0 if res_f["verdict"] == "FAITHFUL" else 1)
                            results_data.append({"Type": "Faithful", "Prediction": res_f["verdict"], "Correct": res_f["verdict"] == "FAITHFUL"})
                            
                            # Test Hallucinated
                            res_h = analyze_hallucination(knowledge, hallucinated_ans)
                            y_true.append(1) # 1 = Hallucinated
                            y_pred.append(1 if res_h["verdict"] == "HALLUCINATED" else 0)
                            results_data.append({"Type": "Hallucinated", "Prediction": res_h["verdict"], "Correct": res_h["verdict"] == "HALLUCINATED"})

                # Metrics Calculation
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
                tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
                
                accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                balanced_acc = (recall + specificity) / 2
                
                # Display Metrics
                st.markdown("### 📊 Performance Metrics")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{accuracy:.1%}")
                m2.metric("Balanced Acc", f"{balanced_acc:.1%}", help="Primary metric: Average of Sensitivity and Specificity")
                m3.metric("Precision (Claims)", f"{precision:.1%}", help="% of flagged hallucinations that were actually incorrect")
                m4.metric("Caught", tp, help="True Positives")
                m5.metric("Missed", fn, help="False Negatives")
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    df = pd.DataFrame(results_data)
                    fig = px.pie(df, names='Correct', title='Overall Prediction Accuracy', color='Correct',
                               color_discrete_map={True: '#10b981', False: '#ef4444'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Type Comparison
                    fig2 = px.bar(df, x='Type', color='Correct', barmode='group', title='Accuracy by Sample Type',
                                color_discrete_map={True: '#10b981', False: '#ef4444'})
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.write("Detailed Breakdown:")
                st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error evaluating dataset: {e}")
                import traceback
                st.code(traceback.format_exc())
