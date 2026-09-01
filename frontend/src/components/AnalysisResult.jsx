import ConfidenceMeter, { VERDICT_STYLES } from "./ConfidenceMeter.jsx";
import FeatureCard from "./FeatureCard.jsx";
import PerplexityChart from "./PerplexityChart.jsx";

function clampPercent(value) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

export default function AnalysisResult({ result }) {
  const {
    prediction,
    confidence,
    language,
    word_count: wordCount,
    sentence_count: sentenceCount,
    stylometric_features: stylo,
    pseudo_perplexity: pp,
    curvature,
    segments,
    explanation,
    disclaimer,
    model_version: modelVersion,
  } = result;

  const style = VERDICT_STYLES[prediction] || VERDICT_STYLES.UNCERTAIN;

  // Rough, bounded visual summaries for the three signal bars. These are
  // for orientation only, not precise probabilities in themselves — the
  // model's calibrated confidence is what is reported as "confidence".
  const stylometricSignal = clampPercent((stylo?.type_token_ratio || 0) * 100);
  const predictabilitySignal = clampPercent(100 - Math.min(pp?.mean || 0, 100));
  const curvatureSignal = clampPercent((curvature?.max_absolute_curvature || 0) * 10);

  return (
    <section className="result" aria-live="polite">
      <div className={`result__verdict result__verdict--${style.className}`}>
        <p className="result__eyebrow">Detection result</p>
        <h2 className="result__label">{style.label}</h2>
        <ConfidenceMeter prediction={prediction} confidence={confidence} />
        <dl className="result__meta">
          <div>
            <dt>Detected language</dt>
            <dd>{language}</dd>
          </div>
          <div>
            <dt>Words analyzed</dt>
            <dd>{wordCount}</dd>
          </div>
          <div>
            <dt>Sentences</dt>
            <dd>{sentenceCount}</dd>
          </div>
          <div>
            <dt>Model version</dt>
            <dd>{modelVersion}</dd>
          </div>
        </dl>
      </div>

      <div className="result__explanation">
        <h3>Why this result?</h3>
        <p>{explanation}</p>
      </div>

      <div className="result__cards">
        <FeatureCard
          title="Stylometric signals"
          value={stylo?.type_token_ratio?.toFixed(2)}
          barPercent={stylometricSignal}
          description="Sentence and word-level patterns such as vocabulary variety, sentence length, and punctuation use."
        />
        <FeatureCard
          title="Pseudo-perplexity"
          value={pp?.mean?.toFixed(1)}
          barPercent={predictabilitySignal}
          description="How predictable the text is to a masked language model. Lower values mean more predictable word choices."
        />
        <FeatureCard
          title="Curvature analysis"
          value={curvature?.max_absolute_curvature?.toFixed(2)}
          barPercent={curvatureSignal}
          description="How much predictability shifts from one segment of the text to the next."
        />
        <FeatureCard
          title="Model confidence"
          value={`${Math.round(confidence * 100)}%`}
          barPercent={clampPercent(confidence * 100)}
          description="The classifier's calibrated confidence in the reported verdict, after abstention is applied."
        />
      </div>

      <div className="result__chart">
        <h3>Segment-level predictability</h3>
        <PerplexityChart segments={segments} />
      </div>

      <p className="result__disclaimer">{disclaimer}</p>
    </section>
  );
}
