# %% [markdown]
# # LTT Routing Demo — Toy Calibration of an LLM Router
#
# End-to-end implementation of Learn-Then-Test (Angelopoulos, Bates et al., 2021)
# applied to LLM routing. Given two models — a cheap one and an expensive one —
# LTT finds the most aggressive routing threshold tau such that the expected
# quality drop stays below alpha with probability at least 1 - delta.
#
# Synthetic data only; no real LLM calls.
# Paper: https://arxiv.org/abs/2110.01052

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)

N = 2000          # total queries
ALPHA = 0.10      # max tolerable true risk (10% average quality drop)
DELTA = 0.05      # failure budget (guarantee holds 95% of the time)
CAL_FRAC = 0.5    # fraction of data used for calibration


# %% [markdown]
# ## 1. Synthetic data
#
# Each query has:
# - `router_signal`     — noisy estimate of "cheap model will do fine" (higher = easier query)
# - `quality_cheap`     — quality of cheap model's answer, correlated with signal
# - `quality_expensive` — quality of expensive model's answer, high on average
#
# The expensive model is better on average; the cheap model is acceptable on easy queries.

# %%
router_signal    = rng.uniform(0, 1, size=N)
quality_cheap    = np.clip(router_signal + rng.normal(0, 0.15, N), 0, 1)
quality_expensive = np.clip(0.85 + rng.normal(0, 0.08, N), 0, 1)

df = pd.DataFrame({
    "router_signal":     router_signal,
    "quality_cheap":     quality_cheap,
    "quality_expensive": quality_expensive,
})
df.head()


# %% [markdown]
# ## 2. Loss function
#
# Per-query loss when routing to cheap under threshold tau:
#
#   loss_i(tau) = max(quality_expensive - quality_cheap, 0)  if router_signal > tau
#               = 0                                           otherwise
#
# Clipping at 0 means we never reward the cheap model for over-performing.
# Empirical risk at tau = mean of these losses over the calibration set.

# %%
def loss_at_tau(df: pd.DataFrame, tau: float) -> np.ndarray:
    """Per-query loss when routing to cheap iff router_signal > tau."""
    mask         = df["router_signal"] > tau
    quality_drop = df["quality_expensive"] - df["quality_cheap"]
    clipped      = np.maximum(quality_drop, 0)
    return (clipped * mask).to_numpy()


# %% [markdown]
# ## 3. Calibration / test split

# %%
idx = rng.permutation(N)
cal_idx, test_idx = idx[:int(CAL_FRAC * N)], idx[int(CAL_FRAC * N):]
df_cal  = df.iloc[cal_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)


# %% [markdown]
# ## 4. LTT — fixed-sequence testing over a tau grid
#
# Walk candidate thresholds from conservative (tau=1.0) to aggressive (tau=0.0).
# At each tau, compute the Hoeffding upper confidence bound on true risk:
#
#   UCB(tau) = emp_risk(tau) + sqrt(log(1/delta) / (2 * n_cal))
#
# Reject ("safe") while UCB <= alpha. Stop at the first failure.
# Return the last accepted tau — the most aggressive threshold with a formal guarantee.
#
# Guarantee: with probability >= 1 - delta, the chosen tau has true risk <= alpha.

# %%
def ltt_select_tau(df_cal: pd.DataFrame, alpha: float, delta: float) -> float:
    """Fixed-sequence LTT over tau grid. Returns the chosen routing threshold."""
    taus      = np.linspace(0, 1, 101)[::-1]   # 1.0 down to 0.0
    n_cal     = len(df_cal)
    eps       = np.sqrt(np.log(1 / delta) / (2 * n_cal))
    chosen_tau = 1.0                             # safe fallback: route nothing cheap

    for tau in taus:
        emp_risk = loss_at_tau(df_cal, tau).mean()
        ucb      = emp_risk + eps
        if ucb <= alpha:
            chosen_tau = tau
        else:
            break

    return chosen_tau


# %% [markdown]
# ## 5. Run calibration + evaluate on test

# %%
tau_hat         = ltt_select_tau(df_cal, ALPHA, DELTA)
test_risk       = loss_at_tau(df_test, tau_hat).mean()
test_cheap_frac = (df_test["router_signal"] > tau_hat).mean()

print(f"Chosen tau              = {tau_hat:.2f}")
print(f"Test risk               = {test_risk:.4f}  (target alpha = {ALPHA})")
print(f"Fraction routed to cheap = {test_cheap_frac:.1%}")


# %% [markdown]
# ## 6. Plot: empirical calibration risk vs tau

# %%
taus  = np.linspace(0, 1, 101)
risks = [loss_at_tau(df_cal, t).mean() for t in taus]

plt.figure(figsize=(7, 4))
plt.plot(taus, risks, label="Empirical calibration risk")
plt.axhline(ALPHA,   linestyle="--", color="red",    label=f"alpha = {ALPHA}")
plt.axvline(tau_hat, linestyle=":",  color="orange",  label=f"chosen tau = {tau_hat:.2f}")
plt.xlabel("Routing threshold tau")
plt.ylabel("Empirical risk")
plt.title("LTT Routing — Risk vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("risk_vs_tau.png", dpi=150)
plt.show()


# %% [markdown]
# ## 7. Caveats
#
# - **Synthetic data** — quality scores are simulated; a real router needs a quality scorer
#   (e.g. model-graded evaluation or a reference-based metric).
# - **Hoeffding bound** — valid but loose. Bentkus or HB bounds are tighter when losses
#   cluster near 0, at the cost of a more complex implementation.
# - **Single scalar threshold** — real deployments may need to select over
#   (threshold, backoff rule) jointly, which changes the multiple-testing story.
# - **Static calibration** — the guarantee breaks under distribution shift. Online/adaptive
#   calibration is an open research direction.
