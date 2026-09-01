import { useState } from "react";
import TextInput from "../components/TextInput.jsx";
import AnalysisResult from "../components/AnalysisResult.jsx";
import { analyzeText, ApiError } from "../services/api.js";

export default function Home() {
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("auto");
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzeText(text, language);
      setResult(data);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "Unable to analyze the text right now. Please try again.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="home">
      <section className="home__intro">
        <h2>Analyze text for signals associated with AI-generated writing</h2>
        <p>
          Paste a submission below to get a HUMAN-WRITTEN, AI-GENERATED, or
          UNCERTAIN result, along with a confidence score and a breakdown of
          the underlying evidence. Supports English, Telugu, and Hindi.
        </p>
      </section>

      <TextInput
        text={text}
        setText={setText}
        language={language}
        setLanguage={setLanguage}
        onAnalyze={handleAnalyze}
        isLoading={isLoading}
      />

      {isLoading && (
        <div className="loading-state" role="status">
          <div className="loading-state__bar">
            <div className="loading-state__bar-fill" />
          </div>
          <p>Analyzing the submission…</p>
        </div>
      )}

      {error && (
        <div className="error-state" role="alert">
          <p>{error}</p>
        </div>
      )}

      {!isLoading && !error && !result && (
        <div className="empty-state">
          <p>
            Your result will appear here once you analyze a piece of text.
          </p>
        </div>
      )}

      {result && !isLoading && <AnalysisResult result={result} />}
    </main>
  );
}
