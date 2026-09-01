"""
Builds a manifest CSV (training/data/manifest.csv) from the raw .txt files in
training/data/<language>/<human|ai>/*.txt, with a stratified 70/15/15
train/validation/test split and duplicate-text removal.

Usage:
    python prepare_dataset.py                 # build manifest from real data
    python prepare_dataset.py --demo           # first generate tiny demo data, then build manifest

DEMO DATA — NOT SUITABLE FOR RESEARCH EVALUATION.
See training/data/README.md for how to assemble a real dataset.
"""
import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
LANGUAGES = ["english", "telugu", "hindi"]
LABELS = ["human", "ai"]

# A handful of clearly-labelled synthetic sentences per language/label, only
# used to smoke-test the pipeline. Real projects must replace this with
# genuine collected data (see data/README.md).
_DEMO_TEXT = {
    ("english", "human"): [
        "I woke up late today and had to rush to catch the bus. My friend "
        "waited for me at the corner, laughing about how I always oversleep "
        "on Mondays. We talked about the weekend trip while walking to class.",
        "Honestly, I don't think the meeting went the way anyone expected. "
        "Half the room disagreed with the proposal, and by the end we just "
        "agreed to revisit it next week over coffee instead.",
    ],
    ("english", "ai"): [
        "The implementation of renewable energy infrastructure represents a "
        "critical component of sustainable development strategies. It is "
        "important to note that comprehensive policy frameworks must be "
        "established to ensure effective adoption across various sectors.",
        "In conclusion, artificial intelligence continues to transform "
        "numerous industries by streamlining processes and enhancing "
        "efficiency. Organizations should carefully evaluate the potential "
        "benefits and challenges associated with this technology.",
    ],
    ("telugu", "human"): [
        "నిన్న సాయంత్రం మా అమ్మతో కలిసి బజారుకు వెళ్ళాను. దారిలో వర్షం పడింది, "
        "అందుకని ఒక చిన్న దుకాణంలో ఆగి కాఫీ తాగాము. చాలా బాగుంది ఆ సాయంత్రం.",
    ],
    ("telugu", "ai"): [
        "సాంకేతిక పరిజ్ఞానం అభివృద్ధి చెందడం వలన అనేక పరిశ్రమలలో సమర్థత మెరుగుపడింది. "
        "ఈ మార్పులు భవిష్యత్తులో మరింత ప్రాముఖ్యత సంతరించుకుంటాయని నిపుణులు అభిప్రాయపడుతున్నారు.",
    ],
    ("hindi", "human"): [
        "कल रात बारिश बहुत तेज़ थी, इसलिए हम छत पर बैठकर चाय पीते रहे। मेरी बहन "
        "ने पुरानी यादें साझा कीं और हम देर तक हँसते रहे।",
    ],
    ("hindi", "ai"): [
        "आधुनिक तकनीक ने शिक्षा के क्षेत्र में महत्वपूर्ण परिवर्तन लाए हैं। "
        "इन परिवर्तनों का उद्देश्य छात्रों के लिए सीखने की प्रक्रिया को अधिक "
        "प्रभावी और सुलभ बनाना है।",
    ],
}


def generate_demo_data() -> None:
    for (language, label), texts in _DEMO_TEXT.items():
        out_dir = DATA_DIR / language / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts):
            # Duplicate a couple of times with slight variation so each split
            # has at least a couple of examples per class.
            for j in range(3):
                path = out_dir / f"demo_{i}_{j}.txt"
                suffix = " " if j == 0 else f" (sample {j})"
                path.write_text(text + suffix, encoding="utf-8")
    print("DEMO DATA — NOT SUITABLE FOR RESEARCH EVALUATION.")
    print(f"Generated demo files under {DATA_DIR}")


def _hash_text(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_manifest(seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    seen_hashes = set()

    for language in LANGUAGES:
        for label in LABELS:
            folder = DATA_DIR / language / label
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.txt")):
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                text_hash = _hash_text(text)
                if text_hash in seen_hashes:
                    continue  # exact-duplicate removal
                seen_hashes.add(text_hash)
                rows.append(
                    {
                        "path": str(path),
                        "text": text,
                        "language": language,
                        "label": label,  # human | ai
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(
            "No training data found. Run with --demo to generate a smoke-test "
            "dataset, or add real .txt files under training/data/<language>/<human|ai>/."
        )

    df["split"] = "train"
    for (language, label), group in df.groupby(["language", "label"]):
        idx = group.sample(frac=1.0, random_state=seed).index.tolist()
        n = len(idx)
        n_val = max(1, round(n * 0.15)) if n >= 4 else 0
        n_test = max(1, round(n * 0.15)) if n >= 4 else 0
        val_idx = idx[:n_val]
        test_idx = idx[n_val : n_val + n_test]
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Generate tiny demo data first.")
    args = parser.parse_args()

    if args.demo:
        generate_demo_data()

    df = build_manifest()
    out_path = DATA_DIR / "manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote manifest with {len(df)} documents to {out_path}")
    print(df.groupby(["language", "label", "split"]).size())


if __name__ == "__main__":
    main()
