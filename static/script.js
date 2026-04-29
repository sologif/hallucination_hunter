document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const queryInput = document.getElementById('queryInput');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');
    const overallVerdict = document.getElementById('overallVerdict');
    const confidenceScore = document.getElementById('confidenceScore');
    const claimsContainer = document.getElementById('claimsContainer');
    const generatedAnswerText = document.getElementById('generatedAnswerText');
    const sourcesList = document.getElementById('sourcesList');
    const verdictCard = document.querySelector('.verdict-card');

    // Default query showcasing the capabilities
    if (!queryInput.value) {
        queryInput.value = "What's the airspeed velocity of a European swallow carrying a coconut according to the 2019 Cambridge Ornithology Review?";
    }

    // Auto-resize textarea
    queryInput.addEventListener('input', function() {
        this.style.height = '120px';
        this.style.height = (this.scrollHeight) + 'px';
    });

    analyzeBtn.addEventListener('click', async () => {
        const query = queryInput.value.trim();

        if (!query) {
            alert('Please provide a prompt.');
            return;
        }

        // UI Transition to Loading State
        analyzeBtn.disabled = true;
        loader.classList.remove('hidden');
        resultsSection.classList.remove('visible');
        setTimeout(() => resultsSection.classList.add('hidden'), 400); // Wait for fade out

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.verification && data.verification.verdict === "ERROR") {
                alert("Error during processing: " + data.verification.details);
                if (data.answer) {
                    typeWriter(generatedAnswerText, data.answer);
                    showResultsSection();
                }
                return;
            }

            renderResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Make sure the FastAPI backend is running and dependencies are installed.');
        } finally {
            analyzeBtn.disabled = false;
            loader.classList.add('hidden');
        }
    });

    function showResultsSection() {
        resultsSection.classList.remove('hidden');
        // Small delay to allow display:block to apply before animating opacity
        setTimeout(() => {
            resultsSection.classList.add('visible');
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 50);
    }

    function typeWriter(element, text, speed = 10) {
        element.innerHTML = '';
        let i = 0;
        function type() {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        type();
    }

    function renderResults(data) {
        // 1. Typewriter effect for the generated answer
        typeWriter(generatedAnswerText, data.answer);

        // 2. Render Sources with stagger effect
        sourcesList.innerHTML = '';
        data.sources.forEach((source, index) => {
            const li = document.createElement('li');
            li.className = 'source-item';
            li.innerHTML = `<strong>Source ${index + 1}:</strong> ${escapeHtml(source.text)}`;
            sourcesList.appendChild(li);
        });

        // 3. Render Verification Verdict
        const verData = data.verification;
        overallVerdict.textContent = verData.verdict;
        
        // Update styling based on verdict
        verdictCard.classList.remove('faithful', 'hallucinated');
        if (verData.verdict === 'FAITHFUL') {
            overallVerdict.className = 'verdict-faithful';
            verdictCard.classList.add('faithful');
        } else {
            overallVerdict.className = 'verdict-hallucinated';
            verdictCard.classList.add('hallucinated');
        }
        
        // Render Score Counter Animation
        animateValue(confidenceScore, 0, verData.confidence_score, 1000);
        
        // 4. Render Sentence-Level Claims
        claimsContainer.innerHTML = '';
        verData.claims.forEach((claim, i) => {
            const card = document.createElement('div');
            
            let statusClass = 'neutral';
            let badgeClass = 'badge-neutral';
            
            if (claim.nli_label === 'Entailment') {
                statusClass = 'faithful';
                badgeClass = 'badge-faithful';
            } else if (claim.nli_label === 'Contradiction') {
                statusClass = 'hallucinated';
                badgeClass = 'badge-hallucinated';
            }

            card.className = `claim-card ${statusClass}`;
            card.style.animationDelay = `${i * 0.15}s`; // Staggered entrance
            
            card.innerHTML = `
                <div class="claim-header">
                    <div class="claim-text">"${escapeHtml(claim.claim)}"</div>
                    <div class="badge ${badgeClass}">${claim.nli_label}</div>
                </div>
                <div class="source-match">
                    <strong>Closest Ground Truth Match</strong>
                    <p>"${escapeHtml(claim.best_source_sentence)}"</p>
                </div>
                <div class="metrics-bar">
                    <div class="metric">
                        <span class="metric-label">Entailment Prob</span>
                        <span class="metric-value">${(claim.entailment_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Similarity</span>
                        <span class="metric-value">${claim.similarity_score.toFixed(2)}</span>
                    </div>
                </div>
            `;
            claimsContainer.appendChild(card);
        });

        showResultsSection();
    }

    // Number counting animation
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start) + '%';
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end.toFixed(1) + '%';
            }
        };
        window.requestAnimationFrame(step);
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
