const VERDICT_STYLES = {
  "AI-GENERATED": { className: "ai", label: "AI-generated" },
  "HUMAN-WRITTEN": { className: "human", label: "Human-written" },
  UNCERTAIN: { className: "uncertain", label: "Uncertain" },
};

export default function ConfidenceMeter({ prediction, confidence }) {
  const style = VERDICT_STYLES[prediction] || VERDICT_STYLES.UNCERTAIN;
  const percent = Math.round(confidence * 100);

  return (
    <div className={`confidence-meter confidence-meter--${style.className}`}>
      <div className="confidence-meter__track" role="img" aria-label={`${percent}% confidence`}>
        <div className="confidence-meter__fill" style={{ width: `${percent}%` }} />
      </div>
      <span className="confidence-meter__value">{percent}% confidence</span>
    </div>
  );
}

export { VERDICT_STYLES };
