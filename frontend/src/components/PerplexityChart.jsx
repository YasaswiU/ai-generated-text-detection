/**
 * Segment-level predictability chart, rendered as a lightweight inline SVG
 * (no charting library dependency needed for a single line/bar view).
 */
export default function PerplexityChart({ segments }) {
  if (!segments || segments.length === 0) {
    return (
      <p className="perplexity-chart__empty">
        Not enough segments to plot a predictability curve for this text.
      </p>
    );
  }

  const width = 640;
  const height = 160;
  const padding = 28;
  const scores = segments.map((s) => s.score);
  const maxScore = Math.max(...scores, 1);
  const minScore = Math.min(...scores, 0);
  const range = Math.max(maxScore - minScore, 1);

  const points = segments.map((segment, index) => {
    const x =
      segments.length === 1
        ? width / 2
        : padding + (index / (segments.length - 1)) * (width - padding * 2);
    const y =
      height -
      padding -
      ((segment.score - minScore) / range) * (height - padding * 2);
    return { x, y, segment };
  });

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(" ");

  return (
    <svg
      className="perplexity-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Segment-level predictability chart"
    >
      <line
        x1={padding}
        y1={height - padding}
        x2={width - padding}
        y2={height - padding}
        className="perplexity-chart__axis"
      />
      <path d={pathD} className="perplexity-chart__line" fill="none" />
      {points.map((p) => (
        <g key={p.segment.segment}>
          <circle cx={p.x} cy={p.y} r="4" className="perplexity-chart__dot" />
          <text x={p.x} y={height - padding + 16} className="perplexity-chart__tick">
            {p.segment.segment}
          </text>
        </g>
      ))}
    </svg>
  );
}
