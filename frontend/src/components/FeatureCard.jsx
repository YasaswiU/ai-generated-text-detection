/**
 * A single evidence panel (e.g. "Stylometric Signals", "Predictability").
 * Uses a left border-accent rather than a full rounded-shadow card so the
 * four cards read as related instrument readouts, not identical widgets.
 */
export default function FeatureCard({ title, value, description, barPercent }) {
  return (
    <div className="feature-card">
      <div className="feature-card__head">
        <h3 className="feature-card__title">{title}</h3>
        {value !== undefined && <span className="feature-card__value">{value}</span>}
      </div>
      {barPercent !== undefined && (
        <div className="feature-card__bar-track">
          <div className="feature-card__bar-fill" style={{ width: `${barPercent}%` }} />
        </div>
      )}
      {description && <p className="feature-card__desc">{description}</p>}
    </div>
  );
}
