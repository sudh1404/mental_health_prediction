
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowRight,
  BrainCircuit,
  Check,
  ChevronRight,
  RotateCcw,
  Sparkles,
  WandSparkles,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const examples = [
  {
    label: "Academic pressure",
    text: "I have so many assignments and deadlines. I feel overwhelmed and cannot concentrate.",
  },
  {
    label: "A peaceful day",
    text: "Today was a lovely day. I went for a walk, met my friends, and felt relaxed.",
  },
  {
    label: "Workload",
    text: "There is so much work piling up and I am struggling to keep up with everything.",
  },
];

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyzeText() {
    if (!text.trim() || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!response.ok) {
        throw new Error(`Prediction request failed (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        "Unable to connect to the analysis service. Check that FastAPI is running on port 8000."
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  function resetAnalysis() {
    setText("");
    setResult(null);
    setError("");
  }

  const stressPercentage = result
    ? result.roberta.stress_probability * 100
    : 0;

  return (
    <div className="site-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar">
        <a className="brand" href="#home">
          <span className="brand-icon">
            <BrainCircuit size={22} />
          </span>
          <span>social<span className="brand-accent">sense</span></span>
        </a>

        <div className="topbar-right">
          <span className="status-dot" />
          <span>AI-powered text intelligence</span>
        </div>
      </header>

      <main id="home">
        <section className="hero">
          <motion.div
            className="hero-copy"
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
          >
            <div className="eyebrow">
              <Sparkles size={14} />
              A NEW PERSPECTIVE ON LANGUAGE
            </div>

            <h1>
              Every word has
              <br />
              <span>a deeper story.</span>
            </h1>

            <p className="hero-description">
              Explore stress-related language patterns with three NLP models.
              Turn everyday text into understandable, interactive insights.
            </p>

            <a className="hero-button" href="#analyze">
              Explore your text <ArrowRight size={18} />
            </a>

            <div className="hero-footnote">
              <Check size={15} />
              Research demonstration · Not a clinical diagnosis
            </div>
          </motion.div>

          <motion.div
            className="orb-scene"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1 }}
            aria-hidden="true"
          >
            <div className="orb-ring orb-ring-one" />
            <div className="orb-ring orb-ring-two" />
            <div className="orb">
              <div className="orb-shine" />
            </div>
            <div className="orb-caption">LANGUAGE · PATTERNS · INSIGHT</div>
          </motion.div>
        </section>

        <section className="analysis-section" id="analyze">
          <div className="section-heading">
            <div>
              <div className="eyebrow">
                <WandSparkles size={14} />
                INTERACTIVE ANALYSIS
              </div>
              <h2>What's behind the words?</h2>
              <p>Enter a post or choose an example to begin exploring.</p>
            </div>
            <span className="section-number">01 / ANALYZE</span>
          </div>

          <div className="analysis-card">
            <div className="card-topline">
              <span className="card-label">
                <span className="input-indicator" />
                TEXT INPUT
              </span>
              <span className="character-count">{text.length} characters</span>
            </div>

            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              maxLength={10000}
              placeholder="What's on your mind? Paste a social media post here..."
              aria-label="Text to analyze"
            />

            <div className="examples">
              <span>TRY AN EXAMPLE</span>
              <div className="example-buttons">
                {examples.map((example) => (
                  <button
                    key={example.label}
                    className="example-chip"
                    onClick={() => {
                      setText(example.text);
                      setResult(null);
                      setError("");
                    }}
                  >
                    {example.label} <ChevronRight size={13} />
                  </button>
                ))}
              </div>
            </div>

            <div className="card-actions">
              <button className="reset-button" onClick={resetAnalysis}>
                <RotateCcw size={16} /> Clear
              </button>

              <button
                className="analyze-button"
                onClick={analyzeText}
                disabled={!text.trim() || loading}
              >
                {loading ? "Analyzing language..." : "Analyze text"}
                {!loading && <ArrowRight size={18} />}
              </button>
            </div>
          </div>

          {error && <div className="error-message">{error}</div>}

          <AnimatePresence mode="wait">
            {result && (
              <motion.section
                className="results-section"
                key={result.text}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="section-heading">
                  <div>
                    <div className="eyebrow">
                      <Sparkles size={14} />
                      ANALYSIS COMPLETE
                    </div>
                    <h2>Your text insights</h2>
                    <p>See how three different models interpreted the post.</p>
                  </div>
                  <span className="section-number">02 / INSIGHTS</span>
                </div>

                <div className="results-grid">
                  <div className="primary-result">
                    <span className="result-kicker">ROBERTA ANALYSIS</span>
                    <h3>{result.roberta.prediction}</h3>
                    <p>Predicted text category</p>

                    <div className="probability-heading">
                      <span>Stress-related language probability</span>
                      <strong>{stressPercentage.toFixed(1)}%</strong>
                    </div>

                    <div className="probability-track">
                      <motion.div
                        className="probability-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${stressPercentage}%` }}
                        transition={{ duration: 1, ease: "easeOut" }}
                      />
                    </div>

                    <div className="probability-labels">
                      <span>No Stress</span>
                      <span>Stress</span>
                    </div>
                  </div>

                  <div className="model-results">
                    <div className="model-card">
                      <span>LOGISTIC REGRESSION</span>
                      <strong>{result.logistic_regression.prediction}</strong>
                      <p>TF-IDF baseline</p>
                    </div>

                    <div className="model-card">
                      <span>LINEAR SVM</span>
                      <strong>{result.linear_svm.prediction}</strong>
                      <p>TF-IDF baseline</p>
                    </div>
                  </div>
                </div>

                <p className="result-disclaimer">
                  These predictions identify patterns in text. They do not
                  determine a person's mental-health status or provide a diagnosis.
                </p>
              </motion.section>
            )}
          </AnimatePresence>
        </section>
      </main>

      <footer>
        <span>socialsense · Explore language with perspective.</span>
        <span>Built with NLP and responsible AI principles.</span>
      </footer>
    </div>
  );
}

export default App;
