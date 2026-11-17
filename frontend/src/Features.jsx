
import { useState } from "react";
import {
  Bot, Search, Heart, Wind, BookOpen, Phone, Send,
  ChevronDown, ChevronUp, Play, RotateCcw, MessageCircle,
  ShieldCheck, ExternalLink
} from "lucide-react";
import "./Features.css";

const API_URL = "http://127.0.0.1:8000";

const questions = [
  {
    q: "What is stress?",
    a: "Stress is your body's response to pressure or demands. Some stress is normal, but ongoing stress can affect sleep, concentration, mood, and daily life."
  },
  {
    q: "How can I manage academic pressure?",
    a: "Break assignments into smaller tasks, choose one achievable next step, take regular breaks, and talk to a teacher or someone you trust if the workload feels unmanageable."
  },
  {
    q: "What can I do when I feel overwhelmed?",
    a: "Pause and take a few comfortable breaths. Name what is worrying you, identify one thing you can do now, and consider asking someone for support."
  },
  {
    q: "How can I improve my sleep routine?",
    a: "Try keeping a consistent sleep and wake time, reducing caffeine late in the day, and setting aside a little time to unwind before bed."
  },
  {
    q: "When should I seek professional help?",
    a: "Consider talking to a qualified mental-health professional if distress persists, worsens, affects daily activities, or feels difficult to manage alone."
  },
  {
    q: "Can a text prediction diagnose stress or a mental-health condition?",
    a: "No. SocialSense identifies patterns in text for educational purposes. Its predictions are not a diagnosis and cannot determine a person's mental-health status."
  }
];

const resources = [
  {
    name: "Tele-MANAS",
    detail: "India's national tele-mental-health support service",
    contact: "14416",
    action: "Call 14416",
    url: "https://telemanas.mohfw.gov.in/"
  },
  {
    name: "Emergency assistance in India",
    detail: "For immediate danger or an urgent emergency",
    contact: "112",
    action: "Call 112",
    url: "https://112.gov.in/"
  }
];

function Companion() {
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function sendMessage(event) {
    event.preventDefault();
    const message = draft.trim();
    if (!message || loading) return;

    const previousMessages = messages;
    setMessages([...previousMessages, { role: "user", content: message }]);
    setDraft("");
    setError("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          history: previousMessages.slice(-10)
        })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "The companion is temporarily unavailable.");
      }

      setMessages(current => [
        ...current,
        { role: "assistant", content: data.reply }
      ]);
    } catch (err) {
      setError(err.message || "Couldn't connect to the companion. Check that the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="feature-section" id="companion">
      <div className="feature-heading">
        <span className="feature-kicker">01 / TALK IT OUT</span>
        <h2><Bot size={25} /> SocialSense AI Companion</h2>
        <p>A supportive space to put your thoughts into words, without judgment.</p>
      </div>
      <div className="feature-card companion-card">
        <div className="companion-top">
          <span className="feature-icon"><MessageCircle size={21} /></span>
          <div>
            <strong>SocialSense Companion</strong>
            <p>AI wellbeing support · Not a therapist</p>
          </div>
        </div>
        <div className="chat-messages" aria-live="polite">
          {messages.length === 0 && (
            <div className="chat-welcome">
              <Heart size={27} />
              <h3>What's on your mind?</h3>
              <p>You can talk about college stress, difficult days, or anything you'd like help thinking through.</p>
              <button type="button" className="suggestion-chip"
                onClick={() => setDraft("I've been feeling stressed about college lately.")}>
                I've been stressed about college
              </button>
            </div>
          )}
          {messages.map((item, index) => (
            <div key={index} className={`chat-bubble ${item.role === "user" ? "user-bubble" : "ai-bubble"}`}>
              <span>{item.role === "user" ? "You" : "Companion"}</span>
              <p>{item.content}</p>
            </div>
          ))}
          {loading && <div className="chat-bubble ai-bubble"><span>Companion</span><p>Thinking of a supportive response…</p></div>}
        </div>
        {error && <p className="feature-error">{error}</p>}
        <form className="chat-form" onSubmit={sendMessage}>
          <input value={draft} onChange={e => setDraft(e.target.value)}
            maxLength={4000} placeholder="Write what's on your mind…"
            aria-label="Message to SocialSense Companion" />
          <button type="submit" disabled={!draft.trim() || loading} aria-label="Send message">
            <Send size={18} />
          </button>
        </form>
        <p className="feature-note"><ShieldCheck size={14} /> Avoid sharing passwords or sensitive personal details. This is AI support, not emergency care.</p>
      </div>
    </section>
  );
}

function HelpLibrary() {
  const [search, setSearch] = useState("");
  const [opened, setOpened] = useState(0);
  const filtered = questions.filter(item =>
    `${item.q} ${item.a}`.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <section className="feature-section" id="help">
      <div className="feature-heading">
        <span className="feature-kicker">02 / LEARN & UNDERSTAND</span>
        <h2><BookOpen size={25} /> Help &amp; Hope</h2>
        <p>Explore practical answers to common questions about stress and wellbeing.</p>
      </div>
      <div className="feature-card">
        <label className="library-search">
          <Search size={18} />
          <input value={search} onChange={e => { setSearch(e.target.value); setOpened(-1); }}
            placeholder="Search stress, sleep, academic pressure…" />
        </label>
        <div className="faq-list">
          {filtered.map((item, index) => (
            <div className="faq-item" key={item.q}>
              <button className="faq-question" onClick={() => setOpened(opened === index ? -1 : index)}>
                <span>{item.q}</span>
                {opened === index ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
              {opened === index && <p className="faq-answer">{item.a}</p>}
            </div>
          ))}
          {filtered.length === 0 && <p className="empty-state">No matching questions. Try a different keyword.</p>}
        </div>
      </div>
    </section>
  );
}

function CalmCorner() {
  const [breathing, setBreathing] = useState(false);
  const [groundingStep, setGroundingStep] = useState(0);
  const [journal, setJournal] = useState("");
  const [saved, setSaved] = useState(false);
  const steps = [
    "Notice 5 things you can see around you.",
    "Notice 4 things you can physically feel.",
    "Notice 3 things you can hear.",
    "Notice 2 things you can smell.",
    "Notice 1 thing you can taste, or one slow breath."
  ];

  return (
    <section className="feature-section" id="calm">
      <div className="feature-heading">
        <span className="feature-kicker">03 / TAKE A PAUSE</span>
        <h2><Wind size={25} /> Calm Corner</h2>
        <p>Try a small reset. Choose whichever activity feels comfortable for you.</p>
      </div>
      <div className="calm-grid">
        <div className="feature-card calm-card">
          <span className="feature-icon"><Wind size={22} /></span>
          <h3>Gentle breathing</h3>
          <p>Follow a comfortable rhythm. Never force or hold your breath; stop if you feel dizzy or uncomfortable.</p>
          <div className={`breathing-orb ${breathing ? "breathing-active" : ""}`}>
            <span>{breathing ? "Breathe\nslowly" : "Pause"}</span>
          </div>
          <button className="feature-button" onClick={() => setBreathing(!breathing)}>
            {breathing ? <RotateCcw size={16} /> : <Play size={16} />}
            {breathing ? "Finish exercise" : "Begin breathing"}
          </button>
          {breathing && <p className="exercise-hint">Breathe in gently, then breathe out slowly at your own pace.</p>}
        </div>
        <div className="feature-card calm-card">
          <span className="feature-icon"><Heart size={22} /></span>
          <h3>5–4–3–2–1 grounding</h3>
          <p>Bring your attention back to the present, one sense at a time.</p>
          <div className="grounding-step">
            <span>STEP {groundingStep + 1} OF 5</span>
            <p>{steps[groundingStep]}</p>
          </div>
          <button className="feature-button" onClick={() => setGroundingStep((groundingStep + 1) % steps.length)}>
            {groundingStep === 4 ? "Start again" : "Next step"} <ChevronDown size={16} />
          </button>
        </div>
        <div className="feature-card calm-card journal-card">
          <span className="feature-icon"><BookOpen size={22} /></span>
          <h3>Mindful journal</h3>
          <p>What is one thing on your mind, and one small thing that might help today?</p>
          <textarea value={journal} onChange={e => { setJournal(e.target.value); setSaved(false); }}
            placeholder="Write a few thoughts here…" maxLength={3000} />
          <button className="feature-button" onClick={() => setSaved(true)} disabled={!journal.trim()}>
            {saved ? <ShieldCheck size={16} /> : <Heart size={16} />}
            {saved ? "Saved in this session" : "Keep note in this session"}
          </button>
          <p className="feature-note">Your note is only held in this page's current session; it is not sent to the backend or permanently stored.</p>
        </div>
      </div>
    </section>
  );
}

function SupportResources() {
  return (
    <section className="feature-section" id="resources">
      <div className="feature-heading">
        <span className="feature-kicker">04 / YOU ARE NOT ALONE</span>
        <h2><Phone size={25} /> Reach Out</h2>
        <p>Support options for moments when you need to speak with a person.</p>
      </div>
      <div className="resource-grid">
        {resources.map(resource => (
          <div className="feature-card resource-card" key={resource.name}>
            <span className="feature-icon"><Heart size={22} /></span>
            <h3>{resource.name}</h3>
            <p>{resource.detail}</p>
            <strong className="resource-number">{resource.contact}</strong>
            <a className="feature-button resource-link" href={resource.url} target="_blank" rel="noreferrer">
              {resource.action} <ExternalLink size={15} />
            </a>
          </div>
        ))}
        <div className="feature-card resource-card">
          <span className="feature-icon"><MessageCircle size={22} /></span>
          <h3>Someone you trust</h3>
          <p>Consider reaching out to a friend, family member, faculty mentor, campus counsellor, or qualified mental-health professional.</p>
          <p className="resource-reminder">You do not have to handle everything alone.</p>
        </div>
      </div>
      <p className="resource-disclaimer">If you or someone else is in immediate danger, contact emergency services. Helpline availability can change; use official resources for current information.</p>
    </section>
  );
}

export default function Features() {
  return (
    <div className="extra-features">
      <nav className="feature-nav" aria-label="Wellbeing features">
        <a href="#companion"><Bot size={16} /> AI Companion</a>
        <a href="#help"><BookOpen size={16} /> Help &amp; Hope</a>
        <a href="#calm"><Wind size={16} /> Calm Corner</a>
        <a href="#resources"><Phone size={16} /> Reach Out</a>
      </nav>
      <Companion />
      <HelpLibrary />
      <CalmCorner />
      <SupportResources />
    </div>
  );
}
