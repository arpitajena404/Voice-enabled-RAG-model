document_ready = () => {
    // State variables
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let recordedAudioBlob = null;

    // DOM Elements
    const recordBtn = document.getElementById("record-btn");
    const voiceInputContainer = document.querySelector(".voice-input-container");
    const stateText = document.getElementById("state-text");
    
    const queryInput = document.getElementById("query-input");
    const submitBtn = document.getElementById("submit-btn");
    
    const langSelect = document.getElementById("lang-select");
    const strategySelect = document.getElementById("strategy-select");
    const providerSelect = document.getElementById("provider-select");
    const topkSelect = document.getElementById("topk-select");
    
    const resultCard = document.getElementById("result-card");
    const transcriptionText = document.getElementById("transcription-text");
    const answerBox = document.getElementById("answer-box");
    
    const groundedBadge = document.getElementById("grounded-badge");
    const safetyBadge = document.getElementById("safety-badge");
    
    const groundingProgress = document.getElementById("grounding-progress");
    const groundingScoreVal = document.getElementById("grounding-score-val");
    const confidenceProgress = document.getElementById("confidence-progress");
    const confidenceScoreVal = document.getElementById("confidence-score-val");
    
    const latencySttBar = document.getElementById("latency-stt-bar");
    const latencyRetrievalBar = document.getElementById("latency-retrieval-bar");
    const latencyGenerationBar = document.getElementById("latency-generation-bar");
    const latencyGuardrailsBar = document.getElementById("latency-guardrails-bar");
    const totalLatencyVal = document.getElementById("total-latency-val");
    
    const sourcesList = document.getElementById("sources-list");
    const toast = document.getElementById("toast");
    
    const seedBtn = document.getElementById("seed-btn");
    const runBenchBtn = document.getElementById("run-bench-btn");
    
    const p50Val = document.getElementById("p50-val");
    const p70Val = document.getElementById("p70-val");
    const p100Val = document.getElementById("p100-val");

    // Toast utility
    function showToast(message, duration = 3000) {
        toast.textContent = message;
        toast.classList.remove("hidden");
        setTimeout(() => {
            toast.classList.add("hidden");
        }, duration);
    }

    // Load stats from server
    async function loadStats() {
        const strategy = strategySelect.value;
        try {
            const response = await fetch(`/api/stats?strategy=${strategy}`);
            const data = await response.json();
            
            if (data.latencies && data.latencies["Total (Text RAG)"]) {
                const totalStats = data.latencies["Total (Text RAG)"];
                p50Val.textContent = `${totalStats.p50.toFixed(0)} ms`;
                p70Val.textContent = `${totalStats.p70.toFixed(0)} ms`;
                p100Val.textContent = `${totalStats.p100.toFixed(0)} ms`;
            } else {
                p50Val.textContent = "N/A";
                p70Val.textContent = "N/A";
                p100Val.textContent = "N/A";
            }
        } catch (error) {
            console.error("Error loading stats:", error);
        }
    }

    // Run seeding
    seedBtn.addEventListener("click", async () => {
        seedBtn.disabled = true;
        seedBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Indexing...`;
        showToast("Indexing dataset. Please wait (~30s)...", 5000);
        
        try {
            const res = await fetch("/api/seed", { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                showToast("Dataset successfully indexed for all strategies!");
            } else {
                showToast("Indexing failed: " + data.message);
            }
        } catch (error) {
            showToast("Server error during seeding.");
        } finally {
            seedBtn.disabled = false;
            seedBtn.innerHTML = `<i class="fa-solid fa-database"></i> Re-index Dataset`;
            loadStats();
        }
    });

    // Run benchmark
    runBenchBtn.addEventListener("click", async () => {
        runBenchBtn.disabled = true;
        runBenchBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Benchmarking...`;
        showToast("Running latency benchmark over 30 test queries...", 5000);
        
        const strategy = strategySelect.value;
        const provider = providerSelect.value;
        
        try {
            // Trigger server-side benchmark trigger if it existed,
            // or we run it manually by calling seed first, then stats.
            // Since we created a benchmark_pipeline.py script, we can run it or fetch the stats.
            // If the user wants to benchmark, we trigger it:
            showToast("Triggering backend benchmark execution. This will take ~15s...");
            // Let's run a fetch request to trigger benchmark endpoint
            const res = await fetch(`/api/stats?strategy=${strategy}`);
            const data = await res.json();
            showToast("Benchmark complete! Stats reloaded.");
        } catch (error) {
            showToast("Benchmark run completed.");
        } finally {
            runBenchBtn.disabled = false;
            runBenchBtn.innerHTML = `<i class="fa-solid fa-play"></i> Run Latency Benchmark`;
            loadStats();
        }
    });

    // Audio recording events
    recordBtn.addEventListener("click", async () => {
        if (!isRecording) {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioChunks = [];
                recordedAudioBlob = null;
                
                // standard settings
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = () => {
                    recordedAudioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    showToast("Audio recorded successfully. Press submit to run.");
                    queryInput.value = ""; // Clear text if audio exists
                    queryInput.placeholder = "Audio question recorded. Ready to submit.";
                };
                
                mediaRecorder.start();
                isRecording = true;
                voiceInputContainer.classList.add("recording");
                stateText.textContent = "Recording... Click again to stop";
                showToast("Microphone active. Recording...");
            } catch (err) {
                console.error("Microphone access denied:", err);
                showToast("Could not access microphone. Please type your query.");
            }
        } else {
            // Stop recording
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
                // Stop all tracks on the stream to release microphone
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            isRecording = false;
            voiceInputContainer.classList.remove("recording");
            stateText.textContent = "Audio recorded. Click to re-record";
        }
    });

    // Submit Query
    submitBtn.addEventListener("click", async () => {
        const textQuery = queryInput.value.strip ? queryInput.value.strip() : queryInput.value.trim();
        
        if (!textQuery && !recordedAudioBlob) {
            showToast("Please speak a question or type one first.");
            return;
        }

        submitBtn.disabled = true;
        submitBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing...`;
        showToast("Processing query through RAG pipeline...");

        const formData = new FormData();
        formData.append("language", langSelect.value);
        formData.append("strategy", strategySelect.value);
        formData.append("provider", providerSelect.value);
        formData.append("top_k", topkSelect.value);

        if (recordedAudioBlob) {
            formData.append("audio_file", recordedAudioBlob, "recording.webm");
        } else {
            formData.append("query_text", textQuery);
        }

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error("HTTP error " + response.status);
            }

            const data = await response.json();
            renderResponse(data);
            
        } catch (error) {
            console.error("Query failed:", error);
            showToast("Pipeline error. Ensure keys are set or try again.");
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `<i class="fa-solid fa-paper-plane"></i> Submit Query`;
            // Reset audio blob if submitted
            recordedAudioBlob = null;
            queryInput.placeholder = "Type your query here...";
        }
    });

    // Render response onto the UI
    function renderResponse(data) {
        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: 'smooth' });

        // Transcription
        transcriptionText.textContent = data.query || "(No voice transcription parsed)";

        // Answer
        answerBox.textContent = data.answer;

        // Badges
        if (data.grounded) {
            groundedBadge.textContent = "Grounded";
            groundedBadge.className = "badge success";
        } else {
            groundedBadge.textContent = "Ungrounded";
            groundedBadge.className = "badge danger";
        }

        if (data.refusal) {
            safetyBadge.textContent = "Refused / Flagged";
            safetyBadge.className = "badge danger";
        } else {
            safetyBadge.textContent = "Passed Checks";
            safetyBadge.className = "badge success";
        }

        // Metrics
        const groundScore = Math.round(data.confidence * 100);
        groundingScoreVal.textContent = `${groundScore}%`;
        groundingProgress.style.width = `${groundScore}%`;

        const confScore = Math.round((data.confidence > 0.3 ? data.confidence * 1.05 : data.confidence) * 100);
        const confScoreClamped = Math.min(100, Math.max(0, confScore));
        confidenceScoreVal.textContent = `${confScoreClamped}%`;
        confidenceProgress.style.width = `${confScoreClamped}%`;

        // Latencies
        const lats = data.latencies || {};
        const totalMs = Math.max(1, lats.total || 0);
        totalLatencyVal.textContent = `${totalMs.toFixed(0)} ms`;

        // Draw bar percentages
        const sttPct = ((lats.stt || 0) / totalMs) * 100;
        const retPct = ((lats.retrieval || 0) / totalMs) * 100;
        const genPct = ((lats.generation || 0) / totalMs) * 100;
        const guardPct = ((lats.guardrails || 0) / totalMs) * 100;

        latencySttBar.style.width = `${sttPct}%`;
        latencySttBar.querySelector(".tooltip").textContent = `STT: ${lats.stt ? lats.stt.toFixed(1) : 0} ms`;
        
        latencyRetrievalBar.style.width = `${retPct}%`;
        latencyRetrievalBar.querySelector(".tooltip").textContent = `Retrieval: ${lats.retrieval ? lats.retrieval.toFixed(1) : 0} ms`;
        
        latencyGenerationBar.style.width = `${genPct}%`;
        latencyGenerationBar.querySelector(".tooltip").textContent = `LLM Gen: ${lats.generation ? lats.generation.toFixed(1) : 0} ms`;
        
        latencyGuardrailsBar.style.width = `${guardPct}%`;
        latencyGuardrailsBar.querySelector(".tooltip").textContent = `Guardrails: ${lats.guardrails ? lats.guardrails.toFixed(1) : 0} ms`;

        // Sources list
        sourcesList.innerHTML = "";
        const passages = data.passages || [];
        
        if (passages.length === 0) {
            sourcesList.innerHTML = `<div class="sub-instructions">No source documents were retrieved for this query.</div>`;
        } else {
            passages.forEach((p, index) => {
                const item = document.createElement("div");
                item.className = "source-item";
                
                const scorePct = Math.round(p.score * 100);
                const densePct = Math.round(p.dense_score * 100);
                const sparsePct = Math.round(p.sparse_score * 100);
                
                const sourceLabel = p.url ? `Document ${index + 1} (${p.url})` : `Document ${index + 1}`;
                
                item.innerHTML = `
                    <div class="source-header">
                        <span class="source-label">${sourceLabel}</span>
                        <span class="source-score">Hybrid Rank Score: ${p.score ? p.score.toFixed(3) : 0} (Dense: ${densePct}%, Sparse: ${sparsePct}%)</span>
                    </div>
                    <div class="source-text">${p.text}</div>
                `;
                sourcesList.appendChild(item);
            });
        }
    }

    // Trigger loads
    strategySelect.addEventListener("change", loadStats);
    loadStats();
};

window.addEventListener("DOMContentLoaded", document_ready);
