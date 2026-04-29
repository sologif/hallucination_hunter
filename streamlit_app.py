import streamlit as st
from engine import analyze_hallucination
from rag.vector_db import db as vector_db
from rag.generator import generate_answer
import html

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
    
    username = st.text_input("Username", placeholder="Enter your username (any)")
    password = st.text_input("Password", type="password", placeholder="Enter your password (any)")
    
    if st.button("Login"):
        if username and password:
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Please enter both username and password")
            
else:
    # Main Application
    st.markdown('<div class="main-title">Hallucination <span>Hunter</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enterprise-grade AI verification. Grounded against factual sources to eliminate hallucinations using Natural Language Inference.</div>', unsafe_allow_html=True)
    
    # Add a tiny logout button at top right
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        if st.button("Log out"):
            st.session_state['logged_in'] = False
            st.rerun()

    def render_results(verification_result, answer_text, sources_list):
        verdict = verification_result["verdict"]
        score = verification_result["confidence_score"]
        
        # Overall Verdict
        verdict_class = "faithful" if verdict == "FAITHFUL" else "hallucinated"
        
        st.markdown(f"""
        <div class="verdict-card {verdict_class}">
            <div class="score-container">Confidence Score</div>
            <div class="verdict-title verdict-{verdict_class}">{verdict}</div>
            <div class="score">{score:.2f}%</div>
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
        sources_html = "".join([f'<li class="source-item"><strong>Source {i+1}:</strong><br>{html.escape(s["text"])}</li>' for i, s in enumerate(sources_list)])
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
            
            if nli_label == "Entailment":
                status_class = "faithful"
                badge_class = "badge-faithful"
            elif nli_label == "Contradiction":
                status_class = "hallucinated"
                badge_class = "badge-hallucinated"
            elif nli_label == "Unsupported":
                status_class = "unsupported"
                badge_class = "badge-unsupported"
            else:
                status_class = "neutral"
                badge_class = "badge-neutral"
                
            entailment_prob = claim["entailment_prob"] * 100
            similarity = claim["similarity_score"]
            
            claims_html += f"""
            <div class="claim-card {status_class}">
                <div class="claim-header">
                    <div class="claim-text">"{html.escape(claim['claim'])}"</div>
                    <div class="badge {badge_class}">{nli_label}</div>
                </div>
                <div class="source-match">
                    <strong>Closest Ground Truth Match</strong>
                    <p>"{html.escape(claim['best_source_sentence'])}"</p>
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

    tab1, tab2 = st.tabs(["Ask AI", "Verify Pasted Text"])
    
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        query = st.text_area("Enter your prompt or question for the AI", value="What's the airspeed velocity of a European swallow carrying a coconut according to the 2019 Cambridge Ornithology Review?", height=120, key="ask_input")
        
        if st.button("Generate & Analyze"):
            with st.spinner("Hunting for Hallucinations..."):
                sources = vector_db.search(query, limit=3)
                answer = generate_answer(query, sources)
                
                if answer.startswith("ERROR:"):
                    st.error(answer)
                else:
                    source_passage = " ".join([s["text"] for s in sources])
                    verification_result = analyze_hallucination(source_passage, answer)
                    render_results(verification_result, answer, sources)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Directly paste an AI-generated answer here. We will evaluate the pasted text against the Vector Database.")
        pasted_text = st.text_area("Paste text to verify against ground truth", value="The 2019 Cambridge Ornithology Review proved that European swallows can easily carry 2-pound coconuts for distances up to 50 miles.", height=120, key="verify_input")
        
        if st.button("Verify Pasted Text"):
            with st.spinner("Validating claims..."):
                sources = vector_db.search(pasted_text, limit=3)
                source_passage = " ".join([s["text"] for s in sources])
                verification_result = analyze_hallucination(source_passage, pasted_text)
                render_results(verification_result, pasted_text, sources)
