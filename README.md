# StudySnap
### An offline-first lecture companion for students with unreliable internet

**Built for:** Qualcomm Snapdragon® AI Lab Build & Present Challenge 2026
**Category:** AI for Education / Accessibility
**Target hardware:** Snapdragon-powered HP AI PCs

---

## The problem

Every engineering and BCA student in India has recorded a lecture on their phone.
Almost none of them can do anything useful with it.

The obvious solution — upload the audio to a cloud transcription service — fails for
reasons that have nothing to do with technology:

| Barrier | Reality for a typical student |
|---|---|
| Cost | Cloud ASR APIs are priced in dollars; a student is budgeting in hundreds of rupees |
| Data | 1 hour of lecture audio is ~115 MB uploaded before you get a single word back |
| Connectivity | Campus Wi-Fi drops in lecture halls. So does 4G in older buildings |
| Privacy | Lecture recordings may contain a professor's voice, other students' questions, and internal course material nobody consented to uploading |
| Battery | Continuously streaming audio to a cloud endpoint drains a laptop through a 3-hour lab session |

So the recording sits in the phone's storage and the student re-watches it at 2x speed
the night before the exam.

**StudySnap turns that recording into revision material without a single byte leaving
the laptop.**

---

## What it does

One pipeline, three outputs, zero network calls:

```
lecture audio ──► on-device ASR ──► summary ──► flashcards
   (.wav/.m4a)      (Whisper,        (key         (cloze
                      NPU)           points)       deletion)
```

1. **Transcript** — speech-to-text runs locally on the Hexagon NPU
2. **Summary** — the 5 most salient points, in the order they were spoken
3. **Flashcards** — cloze-deletion cards generated from the summary sentences, ready
   for spaced repetition

Every step is local. Airplane mode is a supported configuration, not a bug.

---

## Why on-device is the right architecture (not just the trendy one)

Cloud ASR is a *design* choice with a cost. On-device is the only architecture where
the accessibility argument actually holds:

- **₹0 marginal cost.** Transcription is compute you already own. There is no per-minute meter.
- **0 MB uploaded.** The audio never leaves the device.
- **Works without connectivity.** The critical property — because the failure case for the
  cloud version is *precisely* the student in a lecture hall with bad Wi-Fi.
- **Privacy by construction.** Not a policy promise. There is no endpoint to breach.
- **Battery-friendly.** NPU inference is dramatically more efficient than streaming audio
  over a radio and waiting for a server, which is what makes a 3-hour lab session viable.

The Snapdragon X Elite / X2 Hexagon NPU is what makes this practical rather than
theoretical — it runs these models at a power envelope a thin-and-light laptop can
sustain all day.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Snapdragon-powered HP AI PC                                 │
│                                                              │
│   mic / audio file                                           │
│         │                                                    │
│         ▼                                                    │
│   ┌────────────────┐   compiled via Qualcomm AI Hub          │
│   │  Whisper-Small │   → runs on Hexagon NPU                 │
│   └────────┬───────┘                                         │
│            │  transcript                                     │
│            ▼                                                 │
│   ┌────────────────┐   summarise + generate cloze cards      │
│   │  small LLM     │   (NPU path, or deterministic NLP       │
│   │  (instruct)    │    fallback when no model is present)   │
│   └────────┬───────┘                                         │
│            │                                                 │
│       ┌────┴────┬──────────┐                                 │
│       ▼         ▼          ▼                                 │
│   summary   flashcards   keywords                            │
│                                                              │
│   ── no network calls anywhere in this diagram ──            │
└──────────────────────────────────────────────────────────────┘
```

Models are compiled and profiled for the target device through **Qualcomm AI Hub**
(`qualcomm_npu.py`), which returns real on-device latency, peak memory, and the
per-layer NPU/CPU execution split.

---

## Measured NPU performance

> ⚠️ **Fill this in from your own run — do not paste estimated figures.**
>
> Run `python qualcomm_npu.py --device "Snapdragon X Elite CRD"` and paste the real
> output from the Hub profile job below.

| Metric | Value |
|---|---|
| Target device | *e.g. Snapdragon X Elite CRD* |
| Model | Whisper-Small |
| Compile job ID | *from Hub* |
| Profile job ID | *from Hub* |
| Inference latency | *from Hub* |
| Peak memory | *from Hub* |
| Compute unit | *NPU / CPU / mixed — from Hub* |

If you have not run the profile job, **delete this table** and replace it with a note
that the models are compiled but the profile job is pending. An empty honest section
beats a fabricated one — judges can see whether your job IDs resolve on the Hub.

---

## How to run it

### Verify the pipeline immediately (no downloads, no network, ~1 second)

```bash
python app.py --demo
```

Runs the full transcript → summary → flashcards pipeline on a bundled sample lecture
using the deterministic extractive path. This proves the back half of the pipeline
works before you spend time on model setup — useful when you are demoing or recording
on a deadline.

### Process a real transcript

```bash
python app.py --text lecture.txt --target "Snapdragon X Elite NPU"
```

### Process real lecture audio

```bash
pip install faster-whisper
python app.py --audio lecture.m4a
```

### Compile + profile the models for the NPU

```bash
pip install qai-hub qai-hub-models
export QAI_HUB_API_TOKEN="<your token from app.aihub.qualcomm.com>"
python qualcomm_npu.py --device "Snapdragon X Elite CRD"
```

### Save results as JSON

```bash
python app.py --demo --out run.json
```

---

## Interactive demo (for the recorded walkthrough)

```bash
python app.py --demo
```

Sample output shape:

```
====================================================================
StudySnap — lecture processed entirely on-device
====================================================================

Inference target : Demo path — extractive NLP, no model download
Pipeline time    : 0.0012s

--- KEY CONCEPTS ---
  • model  • data  • training  • precision  • recall
  • validation  • test  • overfitting

--- SUMMARY ---
  1. A dataset is normally divided into three parts: training data, ...
  ...

--- FLASHCARDS (5) ---
  Q1: Fill in the blank: The ______ set is used to fit the model, ...
       answer: training
  ...

--- DATA / ACCESSIBILITY REPORT ---
  approx_lecture_minutes: 2.3
  data_a_cloud_asr_would_upload_gb: 0.004
  cost_avoided_inr: 0.05
  bytes_sent_to_network_by_studysnap: 0
```

That last line is the whole submission in one number. Swap in a real 60-minute lecture
and it reads `0.115 GB` and `₹1.38` — per lecture, per student, forever.

---

## Design decisions worth defending

**Why an extractive fallback instead of requiring an LLM?**
Two reasons. First, it makes the pipeline verifiable in under a second on any machine,
which matters for graders and for iteration speed. Second, it demonstrates the pipeline
is architecturally sound independent of which model sits in the middle — the summariser
is a swappable component, not a load-bearing assumption.

**Why generate flashcards instead of just a summary?**
A summary is passive. Students re-read summaries and retain very little. Cloze-deletion
cards feed a spaced-repetition loop, which is the mechanism with actual evidence behind
it. The output format is chosen for retention, not for looking impressive in a demo.

**Why not just use a speech-to-text app that already exists?**
Because every one of them that is accurate enough to be useful is cloud-backed. The
offline ones trade accuracy for privacy. On-device NPU inference removes that trade-off,
which was only ever a hardware limitation.

---

## Honest limitations

Listing these deliberately — a submission that claims no weaknesses reads as untested.

- The extractive summariser is TF-based, not semantic. It selects the most *statistically
  salient* sentences, which is not always the same as the most *important* ones. The LLM
  path is intended to replace it and is where the remaining quality gains live.
- Whisper-Small is English-first. Indian-English accents, code-switching, and technical
  jargon in Hinglish will degrade accuracy. Whisper-Medium or a fine-tune on Indian
  lecture audio is the obvious next step.
- Numbers quoted for data and cost saved are computed from a 115 MB/hour audio estimate
  at a 16 kHz mono bitrate. Real bitrates vary; the constants are at the top of `app.py`
  and are easy to correct.
- No microphone capture UI yet — the current interface is CLI plus file input.

---

## Repository layout

```
snapstudy/
├── app.py             pipeline: transcribe → summarise → flashcards → report
├── qualcomm_npu.py    AI Hub compile + profile for Snapdragon NPU
├── requirements.txt
└── README.md
```

---

## Roadmap

1. Fine-tune Whisper on Indian-English lecture audio
2. On-device voice-activity detection so recording starts and stops itself
3. Spaced-repetition scheduler with an FSRS-style interval model
4. Export to Anki / Quizlet
5. Kannada and Hindi lecture support via multilingual Whisper

---

## Background

Built for the **Qualcomm Snapdragon AI Lab Build & Present Challenge 2026**, which asks
for AI solutions designed or optimised for Snapdragon-powered HP PCs.

The design constraint driving every decision above: **build for the student who
currently has the least access.** That student is on a campus with unreliable Wi-Fi,
paying for mobile data by the gigabyte, and cannot upload their lectures to a server in
another country. On-device inference is what makes the tool reach them.
