import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const initialState = [0.4, 0.5, 0.4, 0.6, 0.5, 0.5, 0.1];

function App() {
  const [message, setMessage] = useState("");
  const [reply, setReply] = useState("");
  const [loading, setLoading] = useState(false);

  async function askAdvisor(event) {
    event.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/advice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: initialState, user_type: "neutral" }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Advisor request failed");
      setReply(data.response);
    } catch (error) {
      setReply(`Unable to reach the advisor: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <h1>Orango AI Logical Advisor</h1>
      <p className="subtitle">A reinforcement-learning policy with a safe response layer.</p>
      <form onSubmit={askAdvisor}>
        <textarea value={message} onChange={(event) => setMessage(event.target.value)}
          placeholder="Describe what you are trying to reason through..." />
        <button disabled={loading}>{loading ? "Thinking..." : "Ask advisor"}</button>
      </form>
      {reply && <section className="reply"><strong>Advisor</strong><p>{reply}</p></section>}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
