"""
StudySnap — offline-first study companion for Indian college students.
Qualcomm Snapdragon AI Lab Build & Present Challenge 2026

Core pipeline: lecture audio -> on-device transcript -> summary -> flashcards.
Design constraint: ZERO network calls at runtime. Everything on-device.

Two execution paths:
  1. NPU path   (Snapdragon X Elite / X2 via Qualcomm AI Hub)  -- see qualcomm_npu.py
  2. Demo path  (pure-Python extractive summarizer, no downloads) -- `python app.py --demo`

The demo path exists so the pipeline can be verified end-to-end without a
network connection or a 200MB model download. It is a real algorithm
(TF-based extractive summarisation + cloze generation), not a mock.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "of",
    "to", "in", "on", "at", "by", "for", "with", "from", "as", "is", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "we", "you", "they", "he", "she", "i", "me", "my", "our", "your",
    "their", "there", "here", "what", "which", "who", "whom", "when", "where",
    "why", "how", "can", "could", "will", "would", "should", "may", "might",
    "must", "do", "does", "did", "have", "has", "had", "not", "no", "yes",
    "also", "such", "into", "over", "under", "between", "about", "each",
    "other", "more", "most", "some", "any", "all", "both", "very", "just",
    "only", "own", "same", "too", "s", "t", "d", "ll", "m", "re", "ve",
}

# Cost of 1 GB of mobile data in India, INR. Used for the savings estimate.
INR_PER_GB = 12.0
# Rough payload for sending 1 hour of audio to a cloud ASR API, in GB.
GB_PER_HOUR_AUDIO = 0.115  # 16kHz mono WAV ~= 115 MB/hour


@dataclass
class Result:
    transcript: str
    summary: list[str]
    flashcards: list[dict]
    keywords: list[str]
    elapsed_s: float
    inference_target: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Text utilities — real extractive summarisation, no external deps
# ----------------------------------------------------------------------------

def sentences(text: str) -> list[str]:
    """Split on sentence boundaries, keeping reasonably sized chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS]


def score_sentences(sents: list[str]) -> list[tuple[float, int, str]]:
    """
    Classic TF-based extractive scoring (Luhn-style), length-normalised.

    Words that appear often across the document but are not stopwords are
    treated as salient. Each sentence is scored by the summed frequency of
    its content words, normalised by sqrt(length) so long sentences don't
    automatically win.
    """
    freq = Counter(tokenize(" ".join(sents)))
    if not freq:
        return [(0.0, i, s) for i, s in enumerate(sents)]

    max_f = max(freq.values())
    scored = []
    for i, s in enumerate(sents):
        toks = tokenize(s)
        if not toks:
            scored.append((0.0, i, s))
            continue
        raw = sum(freq[t] / max_f for t in toks)
        scored.append((raw / math.sqrt(len(toks)), i, s))
    return scored


def summarize(text: str, n: int = 5) -> list[str]:
    """Return the top-n sentences in original document order."""
    sents = sentences(text)
    if len(sents) <= n:
        return sents
    ranked = sorted(score_sentences(sents), key=lambda x: x[0], reverse=True)[:n]
    return [s for _, _, s in sorted(ranked, key=lambda x: x[1])]


def keywords(text: str, n: int = 8) -> list[str]:
    return [w for w, _ in Counter(tokenize(text)).most_common(n)]


# ----------------------------------------------------------------------------
# Flashcard generation — cloze deletion on high-salience sentences
# ----------------------------------------------------------------------------

def make_flashcards(text: str, summary: list[str], per_sentence: int = 1) -> list[dict]:
    """
    Build cloze-deletion cards. For each summary sentence, blank out a salient
    content word and store it as the answer. Purely deterministic.

    Answer words are de-duplicated across cards: a deck that answers "model"
    four times tests nothing. If a sentence's best candidate is already used,
    we fall to the next-best candidate, and skip the sentence only if every
    candidate is exhausted.
    """
    freq = Counter(tokenize(text))
    cards: list[dict] = []
    used_answers: set[str] = set()

    for s in summary:
        toks = tokenize(s)
        # Prefer specific, informative terms: frequent in the document, but
        # break ties toward longer words (len >= 6 is a decent proxy for a
        # term that actually carries meaning).
        candidates = [t for t in dict.fromkeys(toks) if t in freq and t not in used_answers]
        if not candidates:
            continue
        ranked = sorted(candidates, key=lambda t: (-freq[t], -len(t)))

        for target in ranked:
            blanked = re.sub(rf"\b{re.escape(target)}\b", "______", s, count=1, flags=re.I)
            if blanked == s:
                continue
            cards.append({
                "question": f"Fill in the blank: {blanked}",
                "answer": target,
                "source_sentence": s,
            })
            used_answers.add(target)
            break  # one card per sentence

    return cards


# ----------------------------------------------------------------------------
# Audio path — real Whisper if available, otherwise clear instruction
# ----------------------------------------------------------------------------

def transcribe(audio_path: str) -> str:
    """
    On-device transcription.

    Production path (Snapdragon NPU): Whisper-Small compiled for Hexagon NPU
    via Qualcomm AI Hub -- see qualcomm_npu.py for the compile + profile flow.

    This function uses faster-whisper as the portable fallback, which runs on
    CPU. If neither is installed, we fail loudly rather than fabricating output.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit(
            "No speech-to-text backend available.\n"
            "  pip install faster-whisper   (portable CPU fallback)\n"
            "  -- or --\n"
            "  Build the Whisper-Small NPU graph with:  python qualcomm_npu.py\n"
            "Or use --demo to verify the summary/flashcard pipeline with no audio."
        )

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="en", vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments)


# ----------------------------------------------------------------------------
# Sample data for the zero-dependency demo run
# ----------------------------------------------------------------------------

SAMPLE = """Machine learning is a subfield of artificial intelligence that gives
systems the ability to learn from data without being explicitly programmed for every
rule. Instead of hand-coding logic, we expose a model to examples and let it adjust
its internal parameters to reduce error. A dataset is normally divided into three
parts: training data, validation data, and test data. The training set is used to fit
the model, the validation set is used to tune hyperparameters, and the test set is
used exactly once to estimate how the model behaves on unseen inputs. If you tune
your model against the test set, you are leaking information and your reported
accuracy becomes meaningless. Overfitting happens when a model memorises the training
examples rather than learning the underlying pattern. An overfitted model performs
extremely well on training data and poorly on validation data. Regularisation,
dropout, and early stopping are three standard techniques used to reduce overfitting.
Underfitting is the opposite problem, where the model is too simple to capture the
real structure of the data. Bias measures the error introduced by approximating a
complex problem with a simpler model, while variance measures how much the model
changes when trained on a different sample of data. There is a trade-off between bias
and variance, and the goal of model selection is to find the point that minimises
total error. In classification problems we commonly report accuracy, precision,
recall, and the F1 score. Precision answers how many of the predicted positives were
actually correct, while recall answers how many of the real positives were found.
The F1 score is the harmonic mean of precision and recall, which makes it useful when
the classes are imbalanced. A confusion matrix shows the full breakdown of true
positives, true negatives, false positives, and false negatives. For regression
problems we instead report mean absolute error, root mean squared error, and R
squared. Feature engineering is often more valuable than swapping algorithms, because
better inputs give the model a much easier problem to solve. Finally, always establish
a simple baseline before building anything complex, because a baseline tells you
whether your sophisticated model is actually adding value."""


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(text: str, target: str) -> Result:
    t0 = time.perf_counter()
    summary = summarize(text, n=5)
    cards = make_flashcards(text, summary)
    kws = keywords(text, n=8)
    return Result(
        transcript=text[:400] + ("..." if len(text) > 400 else ""),
        summary=summary,
        flashcards=cards,
        keywords=kws,
        elapsed_s=round(time.perf_counter() - t0, 4),
        inference_target=target,
    )


def savings_report(transcript: str) -> dict:
    """Quantifies the offline benefit — this is the accessibility argument."""
    words = len(transcript.split())
    # ~150 spoken words per minute is a normal lecture pace
    minutes = max(words / 150.0, 1e-6)
    hours = minutes / 60.0
    gb = hours * GB_PER_HOUR_AUDIO
    return {
        "approx_lecture_minutes": round(minutes, 1),
        "data_a_cloud_asr_would_upload_gb": round(gb, 3),
        "cost_avoided_inr": round(gb * INR_PER_GB, 2),
        "bytes_sent_to_network_by_studysnap": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="StudySnap — offline lecture-to-flashcard pipeline")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="Path to lecture audio (needs faster-whisper)")
    src.add_argument("--text", help="Path to a transcript .txt file")
    src.add_argument("--demo", action="store_true", help="Run on bundled sample transcript")
    ap.add_argument("--out", default="studysnap_output.json", help="Where to write results")
    ap.add_argument("--target", default="CPU fallback (portable)",
                    help="Label for the inference target, e.g. 'Snapdragon X Elite NPU'")
    args = ap.parse_args()

    if args.demo:
        transcript = SAMPLE
        target = "Demo path — extractive NLP, no model download"
    elif args.text:
        transcript = Path(args.text).read_text(encoding="utf-8")
        target = args.target
    else:
        transcript = transcribe(args.audio)
        target = args.target

    result = run(transcript, target)

    print("=" * 68)
    print("StudySnap — lecture processed entirely on-device")
    print("=" * 68)
    print(f"\nInference target : {result.inference_target}")
    print(f"Pipeline time    : {result.elapsed_s}s")

    print("\n--- KEY CONCEPTS ---")
    for k in result.keywords:
        print(f"  • {k}")

    print("\n--- SUMMARY ---")
    for i, s in enumerate(result.summary, 1):
        print(f"  {i}. {s}")

    print(f"\n--- FLASHCARDS ({len(result.flashcards)}) ---")
    for i, c in enumerate(result.flashcards, 1):
        print(f"  Q{i}: {c['question']}")
        print(f"       answer: {c['answer']}")

    print("\n--- DATA / ACCESSIBILITY REPORT ---")
    for k, v in savings_report(transcript).items():
        print(f"  {k}: {v}")

    Path(args.out).write_text(result.to_json(), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
