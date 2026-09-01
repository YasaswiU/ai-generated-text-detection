export default function About() {
  return (
    <main className="about">
      <h2>About this project</h2>

      <p>
        AI-Generated Text Detection is a multilingual system that estimates
        whether a piece of writing was more likely produced by a human or by
        an AI language model. It combines stylometric analysis, language-model
        predictability (pseudo-perplexity), and predictability curvature into
        a calibrated classifier that can also abstain when the evidence is
        weak.
      </p>

      <h3>How it works</h3>
      <p>
        The backend extracts Unicode-aware stylometric features (sentence
        length, vocabulary variety, punctuation patterns, and more), computes
        pseudo-perplexity for segments of the text using a multilingual masked
        language model (XLM-RoBERTa), and measures how that predictability
        shifts across the text (curvature). These features feed a calibrated
        classifier, which reports HUMAN-WRITTEN, AI-GENERATED, or UNCERTAIN.
      </p>

      <h3>Why "pseudo-perplexity"?</h3>
      <p>
        XLM-RoBERTa is a masked language model, not a causal, autoregressive
        model like GPT. Standard perplexity is only defined for autoregressive
        models. For masked language models, this project uses
        pseudo-perplexity, computed by masking each token in turn and scoring
        how well the model predicts it from its surrounding context.
      </p>

      <h3>Limitations</h3>
      <p>
        AI-text detection is probabilistic. Human and AI writing styles can
        overlap substantially, short texts are harder to classify reliably
        than long ones, and heavily edited AI text (or unusual human writing
        styles) can be misclassified. No result from this tool should be
        treated as definitive proof of authorship, and it should not be used
        as the sole basis for an academic-integrity decision.
      </p>

      <h3>Privacy</h3>
      <p>
        Submitted text is processed in memory to produce a result and is not
        stored by default. Server logs record metadata such as timing,
        detected language, and the prediction — never the submitted text
        itself.
      </p>
    </main>
  );
}
