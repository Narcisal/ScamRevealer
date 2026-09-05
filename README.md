# ScamRevealer

An early-stage prototype for scam-message detection — shelved after running into a real-data availability problem, before I pivoted to [Care4U](https://github.com/Narcisal/Care4U) for my senior capstone.

## What it does

Takes a batch of Chinese text messages and classifies each as scam / safe:

- Encodes each message into a 768-dim semantic vector using `bert-base-chinese`
- Feeds the vector into a Random Forest classifier trained on labeled scam/ham examples
- Logs predictions, confidence scores, and latency to a Weights & Biases dashboard for each run

## Status

This was a research exploration, not a finished product — no web interface, no live API, just a batch script (`main.py`) run against JSON test data. Development stopped mainly because I couldn't get access to real-world scam message datasets or text; training and testing on synthetic/GPT-generated data meant the model's results wouldn't hold up against real scam patterns, so the whole approach risked being untestable in any meaningful way. I moved on to a different capstone direction ([Care4U](https://github.com/Narcisal/Care4U)) after this phase, so it's no longer actively developed. Keeping it up as a record of that early research process.

## Stack

`bert-base-chinese` (Hugging Face Transformers) · scikit-learn (Random Forest) · Weights & Biases (experiment tracking) · pandas