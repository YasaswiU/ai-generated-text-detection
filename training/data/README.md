# Training Data

This project ships with **no real dataset** because a legitimate detector
must be trained on genuinely human-written and genuinely AI-generated text
that you collect and license yourself. Distributing a fake "research-grade"
dataset would be scientifically dishonest, so we don't.

## Expected structure

```
training/data/
    english/
        human/   *.txt  -- one human-written document per file
        ai/      *.txt  -- one AI-generated document per file
    telugu/
        human/
        ai/
    hindi/
        human/
        ai/
```

## Building a real dataset

1. **Human samples**: essays, reports, forum posts, or assignments you have
   the rights/permission to use. Aim for a range of topics, authors, and
   writing levels so the model doesn't just learn "this one person's style".
2. **AI samples**: generate text with a range of different LLMs and prompts
   covering the same topics as your human samples, so the model learns to
   separate *authorship*, not *topic*.
3. **Avoid topic leakage**: don't let the same essay topic appear only in
   the AI set or only in the human set -- split by topic as well as by
   document, and keep the same topic distribution across train/val/test.
4. **Avoid duplicate/near-duplicate texts** across train/validation/test
   splits (`training/prepare_dataset.py` does a normalized-text hash check
   for exact duplicates, but you are responsible for near-duplicates).
5. Aim for at least a few hundred documents per class per language before
   trusting evaluation numbers -- fewer than that produces very noisy
   metrics.

## Demo data

`training/prepare_dataset.py --demo` generates a tiny synthetic dataset
purely so you can smoke-test that the pipeline (feature extraction ->
training -> calibration -> evaluation) runs end-to-end.

**DEMO DATA — NOT SUITABLE FOR RESEARCH EVALUATION.** Any accuracy/F1/ROC-AUC
numbers produced from it are meaningless and must not be reported as real
results in a report, paper, or viva.
