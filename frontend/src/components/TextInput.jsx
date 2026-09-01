const LANGUAGES = [
  { value: "auto", label: "Auto Detect" },
  { value: "en", label: "English" },
  { value: "te", label: "Telugu" },
  { value: "hi", label: "Hindi" },
];

const MIN_LENGTH = 50;
const MAX_LENGTH = 20000;

export default function TextInput({
  text,
  setText,
  language,
  setLanguage,
  onAnalyze,
  isLoading,
}) {
  const charCount = text.length;
  const tooShort = charCount > 0 && charCount < MIN_LENGTH;
  const tooLong = charCount > MAX_LENGTH;
  const canAnalyze = charCount >= MIN_LENGTH && charCount <= MAX_LENGTH && !isLoading;

  const handleSubmit = (event) => {
    event.preventDefault();
    if (canAnalyze) onAnalyze();
  };

  return (
    <form className="text-input" onSubmit={handleSubmit}>
      <div className="text-input__row">
        <label htmlFor="language-select" className="text-input__label">
          Language
        </label>
        <select
          id="language-select"
          className="text-input__select"
          value={language}
          onChange={(event) => setLanguage(event.target.value)}
        >
          {LANGUAGES.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <label htmlFor="text-area" className="text-input__label">
        Text to analyze
      </label>
      <textarea
        id="text-area"
        className="text-input__area"
        placeholder="Paste or type a submission here…"
        value={text}
        onChange={(event) => setText(event.target.value)}
        rows={12}
      />

      <div className="text-input__footer">
        <span
          className={
            "text-input__count" +
            (tooShort || tooLong ? " text-input__count--warn" : "")
          }
        >
          {charCount.toLocaleString()} / {MAX_LENGTH.toLocaleString()} characters
          {tooShort && ` — at least ${MIN_LENGTH} recommended`}
        </span>
        <button type="submit" className="button button--primary" disabled={!canAnalyze}>
          {isLoading ? "Analyzing…" : "Analyze text"}
        </button>
      </div>
    </form>
  );
}

export { MIN_LENGTH, MAX_LENGTH };
