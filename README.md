# LTT Routing Demo

A toy end-to-end demonstration of [Learn-Then-Test](https://arxiv.org/abs/2110.01052) (Angelopoulos, Bates et al., 2021) applied to LLM routing.

## What this is

Given a cheap model and an expensive model, and a noisy per-query signal of "the cheap model will do fine here", pick the largest routing threshold such that the *risk* of routing (expected quality drop vs the expensive model) is controlled at level `α` with confidence `1 - δ`.

LTT reframes this as a multiple-hypothesis-testing problem over a grid of candidate thresholds. Fixed-sequence testing gives a finite-sample guarantee: with probability at least `1 - δ`, the returned threshold has true risk `≤ α`.

## What this is not

- Not a benchmark against real LLM APIs. Data is synthetic.
- Not a production router. The loss function here is a quality-drop proxy; a real deployment would need a calibrated quality scorer.
- Not a comprehensive LTT demo — uses Hoeffding's bound for simplicity. Tighter bounds (Bentkus, HB) are in the paper.

## Run it

```bash
pip install -r requirements.txt
python ltt_demo.py     # run as script
# or open ltt_demo.py in VS Code / Jupyter — it uses `# %%` cell markers
```

## Context

Written as preparation for a KURF 2026 fellowship application on LLM routing with Dr Nicola Paoletti. Goal: make the LTT mechanics concrete in code before starting the real project.
