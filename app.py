#### File 3: `app.py` (Live Python Streamlit Dashboard)
*Note: This keeps the exact equations and parameters from your `sim.py` in Appendix A, while using `@st.cache_resource` so the forecaster trains once and lets professors interactively toggle agents and seeds live.*

```python
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Data Center Thermal Control", layout="wide")

# Constants (matching sim.py in Appendix A)
N = 12
DT = 60.0
T_STEPS = 24 * 60
C = 6.0e5
T_SUPPLY = 18.0
T_LIMIT = 32.0
T_SOFT = 27.0
P_IDLE, P_MAX = 3.0e3, 12.0e3


def workload(seed, load_scale=1.0):
    r = np.random.default_rng(seed)
    t = np.arange(T_STEPS) / 60.0
    base = 0.45 + 0.20 * np.sin(2 * np.pi * (t - 9) / 24)
    util = np.zeros((T_STEPS, N))
    for i in range(N):
        skew = 1.0 + (0.35 if i in (2, 3, 4) else 0.0)
        noise = np.convolve(r.normal(0, 0.05, T_STEPS), np.ones(15) / 15, mode="same")
        u = (base * skew + noise) * load_scale
        for _ in range(3):
            s = r.integers(0, T_STEPS - 120)
            d = r.integers(30, 120)
            u[s : s + d] += r.uniform(0.25, 0.45) * load_scale
        util[:, i] = np.clip(u, 0.05, 1.0)
    return util


def step_thermal(T, P_it, cool, Ts=T_SUPPLY):
    UA = 1500.0 * (cool ** 0.8)
    dTdt = (P_it - UA * (T - Ts)) / C
    return T + dTdt * DT


def power_it(util):
    return P_IDLE + (P_MAX - P_IDLE) * util


def fan_power(cool):
    return 1500.0 * (cool ** 3)


def chiller_power(P_it_total, T_supply=T_SUPPLY):
    return P_it_total / (3.0 + 0.15 * (T_supply - 18.0))


@st.cache_resource
def train_forecaster():
    rng = np.random.default_rng(42)
    X, y = [], []
    for seed in (101, 102, 103, 104):
        util = workload(seed)
        for pol in ("static", "reactive"):
            T = np.full(N, 24.0)
            cool = np.full(N, 0.5)
            Ts, Cs = [], []
            for k in range(T_STEPS):
                if pol == "static":
                    cool = np.full(N, 0.55 + 0.1 * rng.random())
                else:
                    cool = np.where(T > 27.0, 1.0, np.where(T < 25.0, 0.45, cool))
                Ts.append(T.copy())
                Cs.append(cool.copy())
                T = step_thermal(T, power_it(util[k]), cool) + rng.normal(0, 0.02, N)
            Ts, Cs = np.array(Ts), np.array(Cs)
            H = 10
            for k in range(T_STEPS - H):
                for i in range(N):
                    X.append([Ts[k, i], util[k, i], k / T_STEPS, Cs[k, i]])
                    y.append(Ts[k + H, i])
    X, y = np.array(X), np.array(y)
    n = len(X)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n) :]
    gbr = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=0
    )
    gbr.fit(X[tr], y[tr])
    pred = gbr.predict(X[te])
    mae = float(np.mean(np.abs(pred - y[te])))
    rmse = float(np.sqrt(np.mean((pred - y[te]) ** 2)))
    r2 = float(1 - np.sum((pred - y[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    mae_p = float(np.mean(np.abs(X[te][:, 0] - y[te])))
    return gbr, {"mae": mae, "rmse": rmse, "r2": r2, "mae_p": mae_p}


def run_sim(
    policy, util, forecaster=None, sim_seed=42,
    rebalance=True, precision=True, supply=True, shield_on=True, thr_hot=26.5
):
    rng = np.random.default_rng(sim_seed)
    T = np.full(N, 24.0)
    cool = np.full(N, 0.5)
    hist_T, hist_cool, hist_P, hist_Ts = [], [], [], []
    Ts = T_SUPPLY
    viol = 0
    migrated = 0.0
    shield_hits = 0
    u = util.copy()

    for k in range(T_STEPS):
        uk = u[k].copy()
        if policy == "static":
            cool = np.full(N, 0.60)
        elif policy == "reactive":
            cool = np.where(T > 27.0, 1.0, np.where(T < 25.0, 0.45, cool))
        elif policy == "proposed":
            feats = np.column_stack([T, uk, np.full(N, k / T_STEPS), cool])
            Tp = forecaster.predict(feats)
            hot = np.where(Tp > thr_hot)[0] if rebalance else []
            if len(hot):
                order = np.argsort(Tp)
                for h in hot:
                    for c in order:
                        if Tp[c] < 25.5 and uk[c] < 0.85:
                            mv = min(0.15, uk[h] - 0.2, 0.95 - uk[c])
                            if mv > 0:
                                uk[h] -= mv
                                uk[c] += mv
                                migrated += mv
                                break
            need = np.clip((Tp - 24.0) / (27.0 - 24.0), 0.0, 1.0)
            cool = (
                0.25 + 0.75 * need
                if precision
                else np.where(T > 27.0, 1.0, np.where(T < 25.0, 0.45, cool))
            )
            Ts = (
                18.0 + 3.0 * float(np.clip((26.0 - Tp.max()) / 3.0, 0.0, 1.0))
                if supply
                else 18.0
            )
            danger = (T > 29.5) | (Tp > 30.0)
            if danger.any():
                Ts = 18.0
            if not shield_on:
                danger = danger & False
            if danger.any():
                shield_hits += int(danger.sum())
                cool = np.where(danger, 1.0, cool)

        P_it = power_it(uk)
        T = step_thermal(T, P_it, cool, Ts) + rng.normal(0, 0.02, N)
        viol += int((T > T_SOFT).sum())
        hist_T.append(T.copy())
        hist_cool.append(cool.copy())
        hist_Ts.append(Ts)
        hist_P.append(P_it.copy())

    hist_T = np.array(hist_T)
    hist_cool = np.array(hist_cool)
    hist_P = np.array(hist_P)
    E_fan = fan_power(hist_cool).sum() * DT / 3.6e6
    E_chill = np.array([chiller_power(p.sum(), ts) for p, ts in zip(hist_P, hist_Ts)]).sum() * DT / 3.6e6
    E_it = hist_P.sum() * DT / 3.6e6
    E_cool = E_fan + E_chill
    return {
        "T": hist_T,
        "peak": float(hist_T.max()),
        "mean_peak": float(hist_T.max(axis=1).mean()),
        "viol_soft": float(viol / (T_STEPS * N) * 100),
        "viol_hard": int((hist_T > T_LIMIT).sum()),
        "E_fan": float(E_fan),
        "E_chill": float(E_chill),
        "E_cool": float(E_cool),
        "PUE": float((E_it + E_cool) / E_it),
        "water_L": float(1.8 * 0.5 * (E_it + E_chill)),
        "migrated": float(migrated),
        "shield": int(shield_hits),
    }


# ---- UI ----
st.title("Autonomous Thermal Control in Sustainable Data Centers")
st.caption("ML 10-Min Forecaster (GBRT) + Multi-Agent Controller + Safety/SLA Shield | IGDTUW Minor Project")

with st.spinner("Training GBRT Forecaster on 4 baseline days (109,824 samples)..."):
    gbr, f_metrics = train_forecaster()

with st.sidebar:
    st.header("Simulation Controls")
    day_seed = st.selectbox("Evaluation Day Seed (Unseen)", [201, 202, 203, 204, 205], index=0)
    load_scale = st.slider("Workload Multiplier", 0.80, 1.25, 1.00, 0.05)
    st.subheader("Agent & Shield Toggles (Ablation)")
    rebalance = st.checkbox("Workload Rebalancing Agent", value=True)
    precision = st.checkbox("Precision Cooling Agent", value=True)
    supply = st.checkbox("Supply-Temp Headroom Rule", value=True)
    shield_on = st.checkbox("Safety / SLA Shield", value=True)
    thr_hot = st.slider("Hot Rack Threshold (°C)", 25.0, 27.5, 26.5, 0.5)

util = workload(day_seed, load_scale=load_scale)
res_static = run_sim("static", util, sim_seed=day_seed)
res_react = run_sim("reactive", util, sim_seed=day_seed)
res_prop = run_sim(
    "proposed", util, gbr, sim_seed=day_seed,
    rebalance=rebalance, precision=precision, supply=supply,
    shield_on=shield_on, thr_hot=thr_hot
)

# Forecaster Banner
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecaster MAE (10-min)", f"{f_metrics['mae']:.3f} °C", f"-{(1 - f_metrics['mae']/f_metrics['mae_p'])*100:.0f}% vs persistence")
c2.metric("Forecaster R²", f"{f_metrics['r2']:.3f}")
c3.metric("Proposed Time > 27°C", f"{res_prop['viol_soft']:.2f}%", f"Static: {res_static['viol_soft']:.2f}%")
c4.metric("Hard Violations (>32°C)", f"{res_prop['viol_hard']}", f"Shield triggers: {res_prop['shield']}")

# Comparison Table
st.subheader("Controller Comparison (Selected Day)")
st.table({
    "Metric": [
        "Peak Hottest Rack (°C)", "Mean Hottest Rack (°C)",
        "Rack-Min > 27°C (%)", "Cooling Energy (kWh/day)",
        "PUE (cooling only)", "Water Proxy (L/day)", "Load Moved (util-steps)"
    ],
    "Static": [
        f"{res_static['peak']:.2f}", f"{res_static['mean_peak']:.2f}",
        f"{res_static['viol_soft']:.2f}%", f"{res_static['E_cool']:.1f}",
        f"{res_static['PUE']:.3f}", f"{res_static['water_L']:.1f}", "0.0"
    ],
    "Reactive": [
        f"{res_react['peak']:.2f}", f"{res_react['mean_peak']:.2f}",
        f"{res_react['viol_soft']:.2f}%", f"{res_react['E_cool']:.1f}",
        f"{res_react['PUE']:.3f}", f"{res_react['water_L']:.1f}", "0.0"
    ],
    "Proposed (ML + Agents + Shield)": [
        f"{res_prop['peak']:.2f}", f"{res_prop['mean_peak']:.2f}",
        f"{res_prop['viol_soft']:.2f}%", f"{res_prop['E_cool']:.1f}",
        f"{res_prop['PUE']:.3f}", f"{res_prop['water_L']:.1f}", f"{res_prop['migrated']:.1f}"
    ],
})

# Plots
col_a, col_b = st.columns([1.4, 1])
with col_a:
    fig1, ax1 = plt.subplots(figsize=(7, 3.5))
    t_hrs = np.arange(T_STEPS) / 60.0
    ax1.plot(t_hrs, res_static["T"].max(axis=1), color="tab:red", label="Static", lw=1.2)
    ax1.plot(t_hrs, res_react["T"].max(axis=1), color="tab:orange", label="Reactive", lw=1.2)
    ax1.plot(t_hrs, res_prop["T"].max(axis=1), color="tab:green", label="Proposed", lw=1.2)
    ax1.axhline(T_SOFT, ls="--", color="black", lw=0.8, label="27°C Soft Target")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Hottest Rack (°C)")
    ax1.set_title("Hottest Rack Temperature Over 24 Hours")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)

with col_b:
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    names = ["Static", "Reactive", "Proposed"]
    ch = [res_static["E_chill"], res_react["E_chill"], res_prop["E_chill"]]
    fn = [res_static["E_fan"], res_react["E_fan"], res_prop["E_fan"]]
    ax2.bar(names, ch, label="Chiller", color="tab:blue")
    ax2.bar(names, fn, bottom=ch, label="Fans", color="tab:cyan")
    ax2.set_ylabel("Cooling Energy (kWh/day)")
    ax2.set_title("Cooling Energy Split")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

st.subheader("12-Rack Thermal Heatmaps (Rack Index vs. Time)")
fig3, axs = plt.subplots(1, 3, figsize=(11, 2.8), sharey=True)
for ax, title, data in zip(
    axs,
    ["Static", "Reactive", "Proposed"],
    [res_static["T"], res_react["T"], res_prop["T"]],
):
    im = ax.imshow(data[::10].T, aspect="auto", cmap="inferno", vmin=22, vmax=33, extent=[0, 24, N, 0])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (h)")
axs[0].set_ylabel("Rack Index")
fig3.colorbar(im, ax=axs, label="°C", shrink=0.85)
st.pyplot(fig3)
plt.close(fig3)
