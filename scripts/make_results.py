"""Build LaTeX tables, number macros and result figures from results/eval/*.json."""
import glob, json, os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figstyle import setup, save, SINGLE, DOUBLE, STYLE, panel, INK, INK2

setup()
E = "results/eval"
OUT = "results/tex"; os.makedirs(OUT, exist_ok=True); os.makedirs("figures", exist_ok=True)
CTRL = ["UC", "RB", "DAO", "PPO"]


def load(name):
    return json.load(open(f"{E}/{name}.json"))


base = {c: load(c) for c in ["UC", "RB", "DAO"]}
ppo_runs = sorted(glob.glob(f"{E}/ppo_main_s*.json"))
ppo = [json.load(open(p)) for p in ppo_runs]
NSEED = len(ppo)


def arr(split, ctrl, key):
    """Per-day values. PPO: shape (seeds, days)."""
    if ctrl == "PPO":
        return np.array([[d[key] for d in r[split]] for r in ppo])
    return np.array([d[key] for d in base[ctrl][split]])


def cost(split, ctrl):
    return -10 * arr(split, ctrl, "reward")      # daily cost c = -r / 0.1


METRICS = [  # key, label, scale, decimals
    ("COST", "Daily cost $\\sum_t c_t$", 1, 1),
    ("viol_hours", "Voltage-violation hours (h)", 1, 1),
    ("uv_pu_h", "Undervoltage (p.u.$\\cdot$h)", 1, 2),
    ("ov_pu_h", "Overvoltage (p.u.$\\cdot$h)", 1, 3),
    ("vmin_day", "Minimum voltage (p.u.)", 1, 3),
    ("vmax_day", "Maximum voltage (p.u.)", 1, 3),
    ("nadir_mean_hz", "Mean hourly nadir (mHz)", 1e3, 1),
    ("nadir_max_hz", "Worst nadir (mHz)", 1e3, 1),
    ("f_rms_mhz", "RMS frequency deviation (mHz)", 1, 2),
    ("rocof_max", "Maximum RoCoF (Hz/s)", 1, 3),
    ("tie_rms_mw", "RMS tie-line deviation (MW)", 1, 2),
    ("reserve_mean_mw", "Mean V2G reserve, both areas (MW)", 1, 1),
    ("throughput_mwh", "Fleet throughput, area level (MWh)", 1, 0),
    ("soc_def_pct", "Terminal SoC deficit (\\%)", 1, 2),
    ("proj_events", "Projection events per day", 1, 1),
    ("loss_mwh", "Energy loss, two feeders (MWh)", 1, 3),
    ("unmet_mwh", "Unmet mobility energy (MWh)", 1, 2),
]


def get(split, ctrl, key):
    if key == "COST":
        return cost(split, ctrl)
    return arr(split, ctrl, key)


def fmt(x, d):
    return f"{x:.{d}f}"


def table(split, keys=None, label="", caption=""):
    rows = []
    for key, lab, sc, d in METRICS:
        if keys and key not in keys:
            continue
        cells = []
        for c in CTRL:
            v = get(split, c, key) * sc
            m = v.mean(); s = v.std(ddof=1) if v.ndim == 1 else v.reshape(-1).std(ddof=1)
            cell = f"{fmt(m, d)} $\\pm$ {fmt(s, d)}"
            if c == "PPO" and v.ndim == 2:
                ss = v.mean(axis=1).std(ddof=1)
                cell += f" ({fmt(ss, d)})"
            cells.append(cell)
        rows.append(lab + " & " + " & ".join(cells) + " \\\\")
    n = len(base["UC"][split])
    txt = ("\\begin{table*}[!t]\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\centering\n\\footnotesize\n"
           "\\begin{tabular}{lcccc}\n\\toprule\n\\textbf{Metric} & \\textbf{UC} & \\textbf{RB} & \\textbf{DAO} & "
           "\\textbf{PPO (proposed)}\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\end{table*}\n")
    return txt


cap_t = (f"Results on the {len(base['UC']['test'])} held-out test days (mean $\\pm$ standard deviation across days). "
         f"PPO values are pooled over {NSEED} seeds; the value in parentheses is the standard deviation of the per-seed means.")
cap_s = (f"Results on the {len(base['UC']['stress'])} out-of-distribution stress days (mean $\\pm$ standard deviation across days). "
         f"PPO values are pooled over {NSEED} seeds; the value in parentheses is the standard deviation of the per-seed means.")
open(f"{OUT}/table_test.tex", "w").write(table("test", label="tab:test", caption=cap_t))
open(f"{OUT}/table_stress.tex", "w").write(table("stress", keys=["COST", "viol_hours", "uv_pu_h", "ov_pu_h", "vmin_day",
     "vmax_day", "nadir_mean_hz", "nadir_max_hz", "rocof_max", "soc_def_pct", "proj_events", "unmet_mwh"], label="tab:stress", caption=cap_s))

# ---- macros for numbers quoted in the text
mac = {}
def setm(name, val, d):
    mac[name] = fmt(val, d)
for split, S in [("test", "T"), ("stress", "S")]:
    for c in CTRL:
        for key, lab, sc, d in METRICS:
            v = get(split, c, key) * sc
            nm = "".join(ch for ch in key.title() if ch.isalpha())
            setm(f"{S}{c}{nm}", v.mean(), d)
            setm(f"{S}{c}{nm}Sd", v.std(ddof=1) if v.ndim == 1 else v.reshape(-1).std(ddof=1), d)
# relative changes and paired tests (PPO seed-mean per day vs baseline)
for split, S in [("test", "T"), ("stress", "S")]:
    pc = cost(split, "PPO").mean(axis=0)
    for c in ["UC", "RB", "DAO"]:
        bc = cost(split, c)
        setm(f"{S}RedCost{c}", 100 * (1 - pc.mean() / bc.mean()), 1)
        st = wilcoxon(pc, bc)
        mac[f"{S}P{c}"] = f"{st.pvalue:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}" \
            if st.pvalue < 1e-3 else f"{st.pvalue:.3f}"
        mac[f"{S}Wins{c}"] = str(int(np.sum(pc < bc)))
    for key in ["uv_pu_h", "nadir_mean_hz", "throughput_mwh", "viol_hours", "nadir_max_hz", "rocof_max"]:
        nm = "".join(ch for ch in key.title() if ch.isalpha())
        p = get(split, "PPO", key).mean()
        for c in ["UC", "RB", "DAO"]:
            b = get(split, c, key).mean()
            setm(f"{S}Red{nm}{c}", 100 * (1 - p / b) if b != 0 else 0.0, 1)
mac["NSeed"] = str(NSEED)
mac["NTest"] = str(len(base["UC"]["test"])); mac["NStress"] = str(len(base["UC"]["stress"]))

# ---- learning curves
DONE = [d for d in sorted(glob.glob("results/ppo_main_s*")) if os.path.exists(f"{d}/meta.json")]
curves = []
for d in DONE:
    L = json.load(open(f"{d}/learning_curve.json"))
    curves.append((np.array([l["step"] for l in L]), np.array([l["val_mean"] for l in L]),
                   np.array([l["train_mean"] if l["train_mean"] is not None else np.nan for l in L])))
n = min(len(c[0]) for c in curves)
steps = curves[0][0][:n]
V = np.stack([-10 * c[1][:n] for c in curves])
T = np.stack([-10 * c[2][:n] for c in curves])
last50 = steps >= steps[-1] - 50000
conv = np.abs(V[:, last50].mean(axis=1) - V[:, -1]) / V[:, -1] * 100
first = V[:, last50][:, 0]
mac["ConvChangePct"] = fmt(np.max(np.abs(V[:, -1] - first) / V[:, -1] * 100), 1)
mac["FinalValCost"] = fmt(V[:, -1].mean(), 1); mac["FinalValCostSd"] = fmt(V[:, -1].std(ddof=1), 1)
best = np.array([v.min() for v in V]); mac["BestValCost"] = fmt(best.mean(), 1)
for c in ["UC", "RB", "DAO"]:
    setm(f"Val{c}", cost("val", c).mean(), 1)
mac["TrainSteps"] = f"{int(steps[-1]):,}".replace(",", "\\,")
walls = [json.load(open(f"{d}/meta.json"))["wall_s"] / 60 for d in DONE]
setm("TrainMin", np.mean(walls), 0)

fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
k = steps / 1e3
for X, lab, col, ls in [(T, "PPO, training episodes", "#6250d6", "--"), (V, "PPO, validation days", STYLE["PPO"]["color"], "-")]:
    m, s = np.nanmean(X, axis=0), np.nanstd(X, axis=0, ddof=1)
    ax.plot(k, m, color=col, ls=ls, lw=1.1, label=lab)
    ax.fill_between(k, m - s, m + s, color=col, alpha=0.18, lw=0)
for c in ["UC", "RB", "DAO"]:
    ax.axhline(cost("val", c).mean(), color=STYLE[c]["color"], ls=STYLE[c]["ls"], lw=1.0, label=f"{c}, validation days")
ax.set_xlabel("Environment steps ($\\times 10^3$)"); ax.set_ylabel("Daily cost")
ax.set_xlim(0, k[-1]); ax.set_ylim(100, 560)
ax.legend(fontsize=6, ncol=1, loc="center right", bbox_to_anchor=(1.0, 0.62))
save(fig, "fig_learning")

# ---- representative test day (day 0, PPO seed 0)
tr = {c: pickle.load(open(f"{E}/trace_{c}_test.pkl", "rb")) for c in ["UC", "RB", "DAO"]}
tr["PPO"] = pickle.load(open(f"{E}/trace_ppo_main_s0_test.pkl", "rb"))
hours = [r["hour"] for r in tr["UC"]]
xh = np.arange(24); xt = [f"{h:02d}" for h in hours]
fig, axs = plt.subplots(3, 2, figsize=(DOUBLE, 4.6), sharex=True)
def line(ax, c, y):
    st = STYLE[c]; ax.plot(xh, y, color=st["color"], ls=st["ls"], marker=st["marker"], label=st["label"], markersize=2.6)
for c in CTRL:
    R = tr[c]
    line(axs[0, 0], c, [r["vmin"][0] for r in R]); line(axs[0, 1], c, [r["vmax"][1] for r in R])
    line(axs[1, 0], c, [r["p_set_mw"][0] * 25 for r in R]); line(axs[1, 1], c, [r["p_set_mw"][1] * 25 for r in R])
    line(axs[2, 0], c, [100 * r["soc"][0] for r in R]); line(axs[2, 1], c, [r["reserve_mw_area"].sum() for r in R])
axs[0, 0].axhline(0.95, color=INK2, lw=0.6); axs[0, 1].axhline(1.05, color=INK2, lw=0.6)
labs = ["Area 1 minimum voltage (p.u.)", "Area 2 maximum voltage (p.u.)", "Area 1 V2G power (MW)",
        "Area 2 V2G power (MW)", "Area 1 fleet SoC (%)", "Total V2G reserve (MW)"]
for a, l, ch in zip(axs.flat, labs, "abcdef"):
    a.set_ylabel(l); panel(a, ch)
for a in axs[2]:
    a.set_xticks(xh[::3]); a.set_xticklabels(xt[::3]); a.set_xlabel("Hour of day (h)"); a.set_xlim(-0.3, 23.3)
axs[0, 0].legend(ncol=2, fontsize=6, loc="lower right")
fig.tight_layout(h_pad=0.4, w_pad=1.0)
save(fig, "fig_day")

# ---- voltage heat maps, UC vs PPO
fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 3.6), sharex=True, sharey=True)
vmn, vmx = 0.88, 1.06
for j, c in enumerate(["UC", "PPO"]):
    for i in range(2):
        M = np.array([r[f"v{i+1}"] for r in tr[c]]).T   # bus x hour
        im = axs[i, j].imshow(M, aspect="auto", origin="lower", cmap="RdBu", vmin=vmn, vmax=vmx,
                              extent=[-0.5, 23.5, 0.5, 33.5], interpolation="nearest")
        axs[i, j].contour(np.arange(24), np.arange(1, 34), M, levels=[0.95, 1.05], colors=INK, linewidths=0.6)
        axs[i, j].set_title(f"Area {i+1}, {STYLE[c]['label']}", fontsize=8)
        axs[i, j].grid(False)
for a in axs[:, 0]: a.set_ylabel("Bus number")
for a in axs[1]:
    a.set_xticks(xh[::3]); a.set_xticklabels(xt[::3]); a.set_xlabel("Hour of day (h)")
cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.015); cb.set_label("Voltage (p.u.)")
cb.ax.axhline(0.95, color=INK, lw=0.8); cb.ax.axhline(1.05, color=INK, lw=0.8)
save(fig, "fig_heatmap")

# ---- frequency transient for the hour with the lowest inertia in Area 2 where the step hits
hsel = None
for k_, r in enumerate(tr["UC"]):
    if r["hour"] == 13:
        hsel = k_
fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 3.4), sharex=True)
tt = np.arange(3000) * 0.02
for c in ["UC", "RB", "PPO"]:
    s = tr[c][hsel]["trace"]; st = STYLE[c]
    kw = dict(color=st["color"], ls=st["ls"], label=st["label"], lw=1.0)
    axs[0, 0].plot(tt, s["df_hz"][:, 0] * 1e3, **kw); axs[0, 1].plot(tt, s["df_hz"][:, 1] * 1e3, **kw)
    axs[1, 0].plot(tt, s["ptie_mw"], **kw); axs[1, 1].plot(tt, s["pev_mw"].sum(axis=1), **kw)
for a, l, ch in zip(axs.flat, ["$\\Delta f_1$ (mHz)", "$\\Delta f_2$ (mHz)", "$\\Delta P_{\\mathrm{tie}}$ (MW)",
                               "V2G droop power (MW)"], "abcd"):
    a.set_ylabel(l); panel(a, ch); a.set_xlim(0, 60)
for a in axs[1]: a.set_xlabel("Time (s)")
axs[0, 1].legend(fontsize=6.5, loc="lower right")
fig.tight_layout(h_pad=0.4, w_pad=1.0)
save(fig, "fig_freq")
sc0 = None
from v2g.profiles import sample_scenario
_rt = np.random.default_rng(2026); sc0 = sample_scenario(_rt, "test")
h_ = tr["UC"][hsel]["hour"]
setm("HrHtwo", 4.0 * (1 - 0.35 * sc0["pv"][1, h_]), 2)
for c in ["UC", "RB", "PPO"]:
    setm(f"Hr{c}NadirTwo", 1e3 * tr[c][hsel]["nadir_hz"][1], 1)
mac["FreqHour"] = f"{tr['UC'][hsel]['hour']:02d}:00"
for c in ["UC", "RB", "PPO"]:
    setm(f"Hr{c}Nadir", 1e3 * max(tr[c][hsel]["nadir_hz"]), 1)
    setm(f"Hr{c}Res", tr[c][hsel]["reserve_mw_area"].sum(), 1)

# ---- distributions across test days
fig, axs = plt.subplots(1, 4, figsize=(DOUBLE, 1.9))
for a, (key, lab, sc) in zip(axs, [("uv_pu_h", "Undervoltage (p.u.$\\cdot$h)", 1), ("nadir_mean_hz", "Mean nadir (mHz)", 1e3),
                                   ("throughput_mwh", "Throughput (MWh)", 1), ("soc_def_pct", "SoC deficit (%)", 1)]):
    data = [get("test", c, key).reshape(-1) * sc for c in CTRL]
    bp = a.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                   flierprops=dict(markersize=2, markeredgewidth=0.4), medianprops=dict(color=INK, lw=0.9),
                   whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
    for p_, c in zip(bp["boxes"], CTRL):
        p_.set_facecolor(STYLE[c]["color"]); p_.set_alpha(0.75)
    a.set_xticks(range(1, 5)); a.set_xticklabels(CTRL); a.set_ylabel(lab)
fig.tight_layout(w_pad=0.6)
save(fig, "fig_box")

# ---- learned policy: hourly reserve share and set-point vs inertia (test days, all seeds)
rho = np.array([[d["hourly"]["rho"] for d in r["test"]] for r in ppo])      # seed, day, hour, area
pa = np.array([[d["hourly"]["p_area_mw"] for d in r["test"]] for r in ppo])
hrs = ppo[0]["test"][0]["hourly"]["hour"]
order = np.argsort(hrs)
fig, axs = plt.subplots(2, 1, figsize=(SINGLE, 3.3), sharex=True)
for i, (col, ls, mk) in enumerate([(STYLE["PPO"]["color"], "-", "o"), ("#eb6834", "--", "s")]):
    m = rho[..., i].reshape(-1, 24).mean(0)[order]; s = rho[..., i].reshape(-1, 24).std(0)[order]
    axs[0].plot(np.sort(hrs), m, color=col, ls=ls, marker=mk, label=f"Area {i+1}")
    axs[0].fill_between(np.sort(hrs), np.clip(m - s, 0, 0.6), np.clip(m + s, 0, 0.6), color=col, alpha=0.15, lw=0)
    m = pa[..., i].reshape(-1, 24).mean(0)[order]; s = pa[..., i].reshape(-1, 24).std(0)[order]
    axs[1].plot(np.sort(hrs), m, color=col, ls=ls, marker=mk, label=f"Area {i+1}")
    axs[1].fill_between(np.sort(hrs), m - s, m + s, color=col, alpha=0.15, lw=0)
axs[0].set_ylabel("Reserve share $\\rho_{i,t}$"); axs[1].set_ylabel("V2G set-point (MW)")
axs[1].axhline(0, color=INK2, lw=0.6)
axs[1].set_xlabel("Hour of day (h)"); axs[1].set_xticks(range(0, 24, 3)); axs[1].set_xlim(-0.3, 23.3)
axs[0].legend(fontsize=6.5, loc="lower left"); panel(axs[0], "a"); panel(axs[1], "b")
for a in axs: a.texts[-1].set_position((0.93, 0.98))
fig.tight_layout(h_pad=0.3)
save(fig, "fig_policy")
R = rho.reshape(-1, 24, 2)
hh = np.array(hrs)
mid = (hh >= 10) & (hh <= 14); night = (hh <= 4)
setm("RhoMidTwo", R[:, mid, 1].mean(), 2); setm("RhoNightTwo", R[:, night, 1].mean(), 2)
setm("RhoMidOne", R[:, mid, 0].mean(), 2); setm("RhoNightOne", R[:, night, 0].mean(), 2)
eve = (hh >= 18) & (hh <= 21)
setm("RhoEveOne", R[:, eve, 0].mean(), 2)

# ---- ablation and reward-weight sensitivity (corrected environment, 200k steps)
TAGNAME = {"ref200": "Ref", "noOV": "NoOV", "noF": "NoF", "noDeg": "NoDeg",
           "wV25": "WvLow", "wV100": "WvHigh", "wF2": "WfLow", "wF8": "WfHigh", "squash": "Squash"}
LABEL = {"ref200": "Full reward (reference)", "noOV": "$w_{\\mathrm{ov}}=0$ (no overvoltage term)",
         "noF": "$w_f=w_{\\mathrm{tie}}=0$ (no frequency terms)", "noDeg": "$w_{\\mathrm{deg}}=0$ (no throughput term)",
         "wV25": "$w_V=25$ (half)", "wV100": "$w_V=100$ (double)", "wF2": "$w_f=2$ (half)", "wF8": "$w_f=8$ (double)",
         "squash": "Squashed Gaussian policy (gSDE)"}
AKEYS = [("COST", 1, 1), ("uv_pu_h", 1, 2), ("ov_pu_h", 1, 3), ("nadir_mean_hz", 1e3, 1),
         ("reserve_mean_mw", 1, 1), ("throughput_mwh", 1, 0), ("soc_def_pct", 1, 2)]
def runs_of(tag):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{E}/ppo_{tag}_s*.json"))]
def seedvals(runs, key, sc, split="test"):
    if key == "COST":
        return np.array([-10 * np.mean([d["reward"] for d in r[split]]) for r in runs])
    return sc * np.array([np.mean([d[key] for d in r[split]]) for r in runs])
def conv_change(tag):
    out = []
    for d in sorted(glob.glob(f"results/ppo_{tag}_s*")):
        L = json.load(open(f"{d}/learning_curve.json"))
        st = np.array([l["step"] for l in L]); v = np.array([l["val_mean"] for l in L])
        first = v[st >= st[-1] - 50000][0]
        out.append(abs(v[-1] - first) / abs(v[-1]) * 100)
    return out
def abl_table(tags, label, caption):
    rows = []
    for tag in tags:
        runs = runs_of(tag)
        if not runs:
            continue
        cells = []
        for key, sc, d in AKEYS:
            v = seedvals(runs, key, sc)
            cells.append(f"{fmt(v.mean(), d)} $\\pm$ {fmt(v.std(ddof=1), d)}" if len(v) > 1 else fmt(v.mean(), d))
            nm = "".join(ch for ch in key.title() if ch.isalpha())
            setm(f"Abl{TAGNAME[tag]}{nm}", v.mean(), d); setm(f"Abl{TAGNAME[tag]}{nm}Sd", v.std(ddof=1) if len(v) > 1 else 0, d)
        cs = seedvals(runs, "COST", 1)
        mac[f"Abl{TAGNAME[tag]}SeedCosts"] = ", ".join(fmt(c_, 1) for c_ in cs)
        sd_ = seedvals(runs, "soc_def_pct", 1)
        mac[f"Abl{TAGNAME[tag]}SeedSocDefs"] = ", ".join(fmt(c_, 1) for c_ in sd_)
        setm(f"Abl{TAGNAME[tag]}SocDefMaxSeed", sd_.max(), 1)
        mac[f"Abl{TAGNAME[tag]}N"] = str(len(runs))
        cc = conv_change(tag); setm(f"Abl{TAGNAME[tag]}ConvMax", max(cc), 1)
        rows.append(LABEL[tag] + f" ({len(runs)})" + " & " + " & ".join(cells) + " \\\\")
    txt = ("\\begin{table*}[!t]\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\centering\n\\footnotesize\n"
           "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccccc}\n\\toprule\n"
           "\\textbf{Reward variant (seeds)} & \\textbf{Cost} & \\textbf{UV (p.u.$\\cdot$h)} & \\textbf{OV (p.u.$\\cdot$h)} & "
           "\\textbf{Nadir (mHz)} & \\textbf{Reserve (MW)} & \\textbf{Thr. (MWh)} & \\textbf{SoC def. (\\%)}\\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return txt
cap_a = ("Reward ablation on the test days (mean $\\pm$ standard deviation across seeds). Every policy is trained for "
         "200\\,000 steps in the corrected environment with the modified reward and evaluated with the full cost \\eqref{eq:cost}.")
cap_w = ("Reward-weight sensitivity on the test days (mean $\\pm$ standard deviation across seeds; 200\\,000 steps; "
         "evaluation with the nominal weights of Table~\\ref{tab:params}).")
if runs_of("ref200"):
    open(f"{OUT}/table_ablation.tex", "w").write(abl_table(["ref200", "noOV", "noF", "noDeg"], "tab:ablation", cap_a))
    open(f"{OUT}/table_sensitivity.tex", "w").write(abl_table(["ref200", "wV25", "wV100", "wF2", "wF8"], "tab:sensitivity", cap_w))
    abl_table(["squash"], "tab:unused", "")   # macros only
    # Welch tests of each variant against the reference (seed means)
    from scipy.stats import ttest_ind
    ref = seedvals(runs_of("ref200"), "COST", 1)
    for tag in ["noOV", "noF", "noDeg", "wV25", "wV100", "wF2", "wF8", "squash"]:
        r_ = runs_of(tag)
        if len(r_) > 1:
            p = ttest_ind(seedvals(r_, "COST", 1), ref, equal_var=False).pvalue
            setm(f"Abl{TAGNAME[tag]}Pval", p, 2)
    # seed-paired differences (same seed = same initial weights and scenario stream)
    refr = {os.path.basename(p): json.load(open(p)) for p in sorted(glob.glob(f"{E}/ppo_ref200_s*.json"))}
    for tag in ["noOV", "noF", "noDeg", "wV25", "wV100", "wF2", "wF8", "squash"]:
        diffs = []
        for p in sorted(glob.glob(f"{E}/ppo_{tag}_s*.json")):
            sd_ = p.split("_s")[-1]
            rp = f"ppo_ref200_s{sd_}"
            if rp in refr:
                a_ = -10 * np.mean([d["reward"] for d in json.load(open(p))["test"]])
                b_ = -10 * np.mean([d["reward"] for d in refr[rp]["test"]])
                diffs.append(a_ - b_)
        if diffs:
            mac[f"Abl{TAGNAME[tag]}PairDiffs"] = ", ".join(("+" if x_ >= 0 else "$-$") + fmt(abs(x_), 1) for x_ in diffs)
            setm(f"Abl{TAGNAME[tag]}PairMean", np.mean(diffs), 1)
    # nominal-weight cost decomposition (voltage, frequency, throughput, terminal SoC, unmet energy)
    def decomp(runs):
        tot = seedvals(runs, "COST", 1)
        v = np.array([np.mean([d["cost_v"] for d in r["test"]]) for r in runs])
        f_ = np.array([np.mean([d["cost_f"] for d in r["test"]]) for r in runs])
        dg = np.array([np.mean([0.05 * 100 * d["throughput_mwh"] / (25 * 18.0) for d in r["test"]]) for r in runs])
        tm = np.array([np.mean([1.0 * d["soc_def_pct"] for d in r["test"]]) for r in runs])
        return dict(Tot=tot.mean(), V=v.mean(), F=f_.mean(), Deg=dg.mean(), Term=tm.mean(), Rest=(tot - v - f_ - dg - tm).mean())
    for tag in ["ref200", "noOV", "noF", "noDeg", "wV25", "wV100", "wF2", "wF8", "squash"]:
        r_ = runs_of(tag)
        if r_:
            setm(f"Abl{TAGNAME[tag]}ReserveRedPct", 100 * (1 - seedvals(r_, "reserve_mean_mw", 1).mean() / seedvals(runs_of("ref200"), "reserve_mean_mw", 1).mean()), 0)
            for k_, v_ in decomp(r_).items():
                setm(f"Abl{TAGNAME[tag]}Cost{k_}", v_, 1)
    dm = decomp(ppo)
    for k_, v_ in dm.items():
        setm(f"AblMainCost{k_}", v_, 1)
# ---- regulation energy and feasibility audit (per controller, test and stress)
REGK = ["reg_batt_mwh", "reg_batt_abs_max_mwh", "reg_batt_mwh_w", "sched_batt_abs_mwh", "pw_excess_max_mw",
        "res_excess_max_mw", "e_viol_mwh", "energy_closure_err_mwh", "raw_clip_events", "raw_clip_steps", "proj_events"]
def regv(split, c, key):
    return get(split, c, key).reshape(-1)
rrows = []
for c in CTRL:
    cum = regv("test", c, "reg_batt_mwh"); mx = regv("test", c, "reg_batt_abs_max_mwh")
    win = regv("test", c, "reg_batt_mwh_w"); sch = regv("test", c, "sched_batt_abs_mwh")
    runs_c = ppo if c == "PPO" else [base[c]]
    hourly_both = np.array([np.array(d["hourly"]["reg_batt_mwh"]).sum(axis=1) for r in runs_c for d in r["test"]])  # (days, 24)
    gross = np.array([np.abs(np.array(d["hourly"]["reg_batt_mwh"])).sum() for r in runs_c for d in r["test"]])
    share_net = 100 * np.abs(cum).sum() / sch.sum()
    share_gross = 100 * gross.sum() / sch.sum()
    nm = c
    setm(f"Reg{nm}CumMean", cum.mean(), 2); setm(f"Reg{nm}CumMin", cum.min(), 2); setm(f"Reg{nm}CumMax", cum.max(), 2)
    setm(f"Reg{nm}PosDaysPct", 100 * np.mean(cum > 0), 0)
    setm(f"Reg{nm}HrMax", np.abs(hourly_both).max(), 3); setm(f"Reg{nm}HrMean", np.abs(hourly_both).mean(), 3)
    setm(f"Reg{nm}WinCum", win.mean(), 4); setm(f"Reg{nm}Share", share_net, 2); setm(f"Reg{nm}ShareGross", share_gross, 2)
    setm(f"Reg{nm}GrossDay", gross.mean(), 1)
    mx = hourly_both  # for the table row below
    for split, S in [("test", "T"), ("stress", "S")]:
        setm(f"{S}{nm}UnmetMax", regv(split, c, "unmet_mwh").max(), 1); setm(f"{S}{nm}UnmetMean", regv(split, c, "unmet_mwh").mean(), 2)
    setm(f"Reg{nm}SocPct", 100 * np.abs(cum).max() / (2 * 25 * 18.0), 2)
    for split, S in [("test", "T"), ("stress", "S")]:
        mac[f"{S}{nm}RawClip"] = str(int(regv(split, c, "raw_clip_events").sum()))
        mac[f"{S}{nm}Proj"] = str(int(regv(split, c, "proj_events").sum()))
        setm(f"{S}{nm}PwExcess", regv(split, c, "pw_excess_max_mw").max(), 3)
        setm(f"{S}{nm}EViol", regv(split, c, "e_viol_mwh").max(), 3)
    nd = len(cum)
    rrows.append(f"{STYLE[c]['label']} & {fmt(np.abs(hourly_both).mean(), 3)} & {fmt(np.abs(hourly_both).max(), 3)} & {fmt(cum.mean(), 2)} $\\pm$ {fmt(cum.std(ddof=1), 2)} & "
                 f"{fmt(win.mean(), 4)} & {fmt(share_gross, 2)} & {int(regv('test', c, 'raw_clip_events').sum())} & "
                 f"{int(regv('test', c, 'proj_events').sum())} & {fmt(regv('test', c, 'pw_excess_max_mw').max(), 3)} & "
                 f"{fmt(regv('test', c, 'e_viol_mwh').max(), 3)} \\\\")
setm("RegClosureMax", max(regv(s_, c, "energy_closure_err_mwh").max() for s_ in ["test", "stress"] for c in CTRL), 1)
mac["RegClosureMax"] = f"{max(regv(s_, c, 'energy_closure_err_mwh').max() for s_ in ['test', 'stress'] for c in CTRL):.1e}"
txt = ("\\begin{table*}[!t]\n\\caption{Energy exchanged by the V2G droop loop and feasibility audit on the test days "
       "(area level, both areas). Regulation energy is the battery-side energy including $\\eta_{\\mathrm{ch}}$ and "
       "$\\eta_{\\mathrm{dis}}$; negative values reduce stored energy. Values use the default decomposed treatment "
       "\\eqref{eq:decomposed}; the window-only column counts the simulated 60~s. The share is the gross (absolute) regulation energy "
       "relative to the battery energy of the scheduled set-points. Counts are summed over all test days "
       "(and seeds for PPO); action-clipping counts refer to individual action components.}\n\\label{tab:regenergy}\n"
       "\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lccccccccc}\n\\toprule\n"
       "& \\multicolumn{5}{c}{\\textbf{Regulation energy}} & \\multicolumn{4}{c}{\\textbf{Feasibility audit}}\\\\\n"
       "\\cmidrule(lr){2-6}\\cmidrule(lr){7-10}\n"
       "\\textbf{Controller} & \\textbf{Mean $|\\cdot|$ per h} & \\textbf{Max $|\\cdot|$ per h} & \\textbf{Daily sum} & "
       "\\textbf{Daily sum, window} & \\textbf{Gross share of} & \\textbf{Raw action} & \\textbf{Power} & \\textbf{Max power} & "
       "\\textbf{Max SoC-limit}\\\\\n"
       " & \\textbf{(MWh)} & \\textbf{(MWh)} & \\textbf{(MWh)} & \\textbf{only (MWh)} & \\textbf{sched. (\\%)} & "
       "\\textbf{clippings} & \\textbf{projections} & \\textbf{excess (MW)} & \\textbf{violation (MWh)}\\\\\n\\midrule\n"
       + "\n".join(rrows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
open(f"{OUT}/table_regenergy.tex", "w").write(txt)

# ---- per-area SoC effect of regulation energy (largest daily |sum| in one area, % of area fleet capacity)
for c in CTRL:
    runs = ppo if c == "PPO" else [base[c]]
    worst = max(abs(np.sum(np.array(d["hourly"]["reg_batt_mwh"])[:, i])) for r in runs for d in r["test"] for i in range(2))
    setm(f"Reg{c}SocAreaMax", 100 * worst / (25 * 18.0), 2)
# ---- v1 (uncorrected model) reference values from the archived evaluation
V1 = "../v2g_rl_v1_backup/results/eval"
if os.path.exists(V1):
    v1p = [json.load(open(p)) for p in sorted(glob.glob(f"{V1}/ppo_main_s*.json"))]
    setm("VOnePPOCost", np.mean([[-10 * d["reward"] for d in r["test"]] for r in v1p]), 1)
    for c in ["UC", "RB", "DAO"]:
        setm(f"VOne{c}Cost", np.mean([-10 * d["reward"] for d in json.load(open(f"{V1}/{c}.json"))["test"]]), 1)
# ---- raw-action clipping breakdown
if os.path.exists("results/raw_clip_breakdown.json"):
    rb_ = json.load(open("results/raw_clip_breakdown.json"))
    mac["RawClipComponents"] = str(rb_["steps"] * 4)
    setm("RawClipFracPct", 100 * rb_["frac_components"], 1)
    fr = np.array(rb_["frac_above"]) + np.array(rb_["frac_below"])
    setm("RawClipRhoPct", 100 * fr[2:].mean(), 1); setm("RawClipPPct", 100 * fr[:2].mean(), 1)

# ---- numbers quoted in the text that come from traces / curves
for c in ["UC", "RB", "PPO"]:
    setm(f"Hr{c}NadirOne", 1e3 * tr[c][hsel]["nadir_hz"][0], 1)
setm("DayUCVminOne", min(r["vmin"][0] for r in tr["UC"]), 2)
setm("DayPPOPeakOne", max(r["p_set_mw"][0] * 25 for r in tr["PPO"]), 0)
setm("DayRBNightVmin", max(r["vmin"][0] for r in tr["RB"] if r["hour"] in (0, 1, 2, 3, 4)), 2)
res_ppo = np.array([r["reserve_mw_area"].sum() for r in tr["PPO"]])
hrs_d = np.array([r["hour"] for r in tr["PPO"]])
outside = ~np.isin(hrs_d, [17, 18, 19, 20])
setm("DayResLo", np.floor(res_ppo[outside].min()), 0); setm("DayResHi", np.ceil(res_ppo[outside].max()), 0)
setm("DayResEveMin", res_ppo[~outside].min(), 1)
A = [json.load(open(f"{d}/learning_curve.json")) for d in DONE]
cross = []
for L in A:
    st = np.array([l["step"] for l in L]); v = -10 * np.array([l["val_mean"] for l in L])
    cross.append(st[np.argmax(v < cost("val", "DAO").mean())])
mac["CrossDAOLo"] = f"{min(cross):,}".replace(",", "\\,"); mac["CrossDAOHi"] = f"{max(cross):,}".replace(",", "\\,")
# paired daily nadir test
pn_ = get("test", "PPO", "nadir_mean_hz").mean(axis=0)
for c in ["UC", "RB", "DAO"]:
    bn = get("test", c, "nadir_mean_hz")
    mac[f"TNadirWins{c}"] = str(int(np.sum(pn_ < bn)))
    pv = wilcoxon(pn_, bn).pvalue
    mac[f"TNadirP{c}"] = f"{pv:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}"
# median SoC deficit and zero-deficit share
for c in ["DAO", "PPO"]:
    v = get("test", c, "soc_def_pct").reshape(-1)
    setm(f"T{c}SocDefMedian", np.median(v), 2); setm(f"T{c}SocDefZeroPct", 100 * np.mean(v < 1e-6), 0)


# ---- regulation-energy treatments (identical scenarios and controllers), from results/eval_modes
MODES = ["none", "window", "hour", "decomposed", "fullhour"]
MODELAB = {"none": "None (neglected)", "window": "60-s window only", "hour": "Hourly extrapolation ($\\times 60$)",
           "decomposed": "Decomposed (default)", "fullhour": "Exact 3600-s simulation"}
if all(os.path.exists(f"results/eval_modes/{m_}/UC.json") for m_ in MODES):
    def mload(mode, c):
        if c == "PPO":
            return [json.load(open(p)) for p in sorted(glob.glob(f"results/eval_modes/{mode}/ppo_main_s*.json"))]
        return [json.load(open(f"results/eval_modes/{mode}/{c}.json"))]
    def mval(mode, c, key, split="test"):
        runs = mload(mode, c)
        v = np.array([[(-10 * d["reward"] if key == "COST" else d[key]) for d in r[split]] for r in runs])
        return v.reshape(-1)
    rows = []
    MK = [("COST", 1, 1), ("soc_def_pct", 1, 2), ("unmet_mwh", 1, 2), ("uv_pu_h", 1, 3), ("viol_hours", 1, 1),
          ("nadir_mean_hz", 1e3, 1), ("reg_batt_mwh", 1, 2), ("e_viol_mwh", 1, 3)]
    for c in CTRL:
        for mode in MODES:
            cells = []
            for key, sc, d in MK:
                v = mval(mode, c, key) * sc
                cells.append(fmt(v.mean(), d))
                nm = "".join(ch for ch in key.title() if ch.isalpha())
                setm(f"Mode{mode.title()}{c}{nm}", v.mean(), d)
            rows.append((f"\\multirow{{5}}{{*}}{{{STYLE[c]['label']}}} " if mode == MODES[0] else "") + "& " + MODELAB[mode] + " & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    txt = ("\\begin{table*}[!t]\n\\caption{Sensitivity of the daily results on the " + str(len(mload('none','UC')[0]['test'])) +
           " test days to the treatment of the regulation energy. Every row uses the same scenarios, disturbances and controllers; "
           "only the energy credited to the droop loop in \\eqref{eq:energy} differs. PPO rows pool the five final policies. "
           "The last column is the energy by which the stored energy would have left $[E^{\\min},E^{\\max}]$ without the limit.}\n"
           "\\label{tab:modes}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llcccccccc}\n\\toprule\n"
           "\\textbf{Controller} & \\textbf{Treatment} & \\textbf{Cost} & \\textbf{SoC def. (\\%)} & "
           "\\textbf{Unmet (MWh)} & \\textbf{UV (p.u.$\\cdot$h)} & \\textbf{Viol. h} & \\textbf{Nadir (mHz)} & "
           "\\textbf{Reg. energy (MWh/d)} & \\textbf{Limit viol. (MWh)}\\\\\n\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    open(f"{OUT}/table_modes.tex", "w").write(txt)
    # how well does each cheap treatment track the exact one, hour by hour (PPO, test days)?
    for c in ["RB", "DAO", "PPO"]:
        runs = mload("fullhour", c)
        full = np.array([h for r in runs for d in r["test"] for h in np.array(d["hourly"]["reg_batt_mwh_full"]).sum(axis=1)])
        hourx = np.array([h for r in runs for d in r["test"] for h in np.array(d["hourly"]["reg_batt_mwh_hour"]).sum(axis=1)])
        gross_full = np.array([d["reg_gross_full_mwh"] for r in runs for d in r["test"]])
        gross_dec = np.array([d["reg_gross_dec_mwh"] for r in runs for d in r["test"]])
        gross_hour = np.array([d["reg_gross_hour_extrap_mwh"] for r in runs for d in r["test"]])
        gross_win = np.array([d["reg_gross_window_mwh"] for r in runs for d in r["test"]])
        net_dec = np.array([d["reg_dec_mwh"] for r in runs for d in r["test"]])
        net_full = np.array([d["reg_full_mwh"] for r in runs for d in r["test"]])
        net_hour = np.array([d["reg_hour_extrap_mwh"] for r in runs for d in r["test"]])
        net_win = np.array([d["reg_batt_mwh_w"] for r in runs for d in r["test"]])
        setm(f"Trk{c}GrossFull", gross_full.mean(), 2); setm(f"Trk{c}GrossDec", gross_dec.mean(), 2)
        setm(f"Trk{c}GrossHour", gross_hour.mean(), 2); setm(f"Trk{c}GrossWin", gross_win.mean(), 2)
        setm(f"Trk{c}GrossHourRatio", gross_hour.mean() / gross_full.mean(), 1); setm(f"Trk{c}GrossDecRatio", gross_dec.mean() / gross_full.mean(), 2)
        setm(f"Trk{c}NetFull", net_full.mean(), 2); setm(f"Trk{c}NetDec", net_dec.mean(), 2)
        setm(f"Trk{c}NetHour", net_hour.mean(), 2); setm(f"Trk{c}NetWin", net_win.mean(), 2)
        setm(f"Trk{c}SignFlipPct", 100 * np.mean(np.sign(full) * np.sign(hourx) < 0), 0)
        setm(f"Trk{c}SatFrac", 100 * np.mean([d["sat_frac_full"] for r in runs for d in r["test"]]), 2)
        setm(f"Trk{c}MarginMin", min(d["sust_margin_min_mwh"] for r in runs for d in r["test"] + r["stress"]), 3)
        setm(f"Trk{c}BindHours", np.mean([d["sust_margin_bind_hours"] for r in runs for d in r["test"]]), 1)
    # PPO cost change between treatments, as % of the seed-to-seed std of the final policies
    seed_sd = get("test", "PPO", "COST").mean(axis=1).std(ddof=1) if False else cost("test", "PPO").mean(axis=1).std(ddof=1)
    base_c = mval("decomposed", "PPO", "COST").mean()
    for mode in MODES:
        setm(f"ModeDeltaCost{mode.title()}", mval(mode, "PPO", "COST").mean() - base_c, 1)
        setm(f"ModeDeltaCostSd{mode.title()}", (mval(mode, "PPO", "COST").mean() - base_c) / seed_sd, 1)
    setm("PPOSeedSd", seed_sd, 1)

# ---- raw-action clipping per dimension (final policies and squashed alternative)
def clip_by_dim(tag):
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"{E}/ppo_{tag}_s*.json"))]
    if not runs:
        return None
    cnt = np.sum([np.array(d["raw_clip_by_dim"]) for r in runs for d in r["test"]], axis=0)
    n = sum(len(r["test"]) * 24 for r in runs)
    return cnt / n * 100, n, len(runs)
cd = clip_by_dim("main")
if cd is not None:
    pct, n, k = cd
    for i, nm in enumerate(["PsetOne", "PsetTwo", "RhoOne", "RhoTwo"]):
        setm(f"ClipDim{nm}", pct[i], 1)
    setm("ClipDimAll", pct.mean(), 1); mac["ClipDimSteps"] = f"{n:,}".replace(",", "\\,")
    if os.path.exists("results/raw_clip_breakdown.json"):
        rb_ = json.load(open("results/raw_clip_breakdown.json"))
        for i, nm in enumerate(["PsetOne", "PsetTwo", "RhoOne", "RhoTwo"]):
            setm(f"ClipAbove{nm}", 100 * rb_["frac_above"][i], 1); setm(f"ClipBelow{nm}", 100 * rb_["frac_below"][i], 1)
        setm("ClipMedianExcess", rb_["median_excess_when_clipped"], 2)
cs_ = clip_by_dim("squash")
if cs_ is not None:
    setm("ClipSquashAll", cs_[0].mean(), 1)
    rows = []
    for tag, lab in [("ref200", "Clipped Gaussian (reference, 200k)"), ("squash", "Squashed Gaussian + gSDE (200k)"), ("main", "Clipped Gaussian (final, 400k)")]:
        c_ = clip_by_dim(tag)
        if c_ is None:
            continue
        runs = runs_of(tag) if tag != "main" else ppo
        cost_ = seedvals(runs, "COST", 1); res_ = seedvals(runs, "reserve_mean_mw", 1); uv_ = seedvals(runs, "uv_pu_h", 1)
        nad_ = seedvals(runs, "nadir_mean_hz", 1e3); soc_ = seedvals(runs, "soc_def_pct", 1)
        rows.append(f"{lab} ({c_[2]}) & " + " & ".join(fmt(x, 1) for x in c_[0]) + f" & {fmt(c_[0].mean(), 1)} & "
                    f"{fmt(cost_.mean(), 1)} $\\pm$ {fmt(cost_.std(ddof=1), 1)} & {fmt(uv_.mean(), 2)} & {fmt(nad_.mean(), 1)} & {fmt(res_.mean(), 1)} & {fmt(soc_.mean(), 2)} \\\\")
    txt = ("\\begin{table*}[!t]\n\\caption{Raw-action saturation per action dimension on the test days (percentage of hourly decisions "
           "whose deterministic action lies outside $[-1,1]$ before clipping; for the squashed policy the action is bounded by construction) "
           "and the resulting performance. Seeds in parentheses.}\n\\label{tab:clip}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{lcccccccccc}\n\\toprule\n\\textbf{Policy (seeds)} & $a_1$ ($P_1$) & $a_2$ ($P_2$) & $a_3$ ($\\rho_1$) & $a_4$ ($\\rho_2$) & "
           "\\textbf{All} & \\textbf{Cost} & \\textbf{UV (p.u.$\\cdot$h)} & \\textbf{Nadir (mHz)} & \\textbf{Reserve (MW)} & \\textbf{SoC def. (\\%)}\\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    open(f"{OUT}/table_clip.tex", "w").write(txt)

val = json.load(open("results/validation.json"))
setm("ValVmin", val["bfs_base_vmin_pu"], 4); setm("ValLoss", val["bfs_base_loss_kw"], 2)
mac["ValVerr"] = f"{val['bfs_vs_pandapower_max_abs_v_err_pu']:.1e}".replace("e-", "\\times10^{-") + "}"
setm("ValDfSim", val["lfc_noagc_df_ss_hz_sim"], 5); setm("ValDfTh", val["lfc_noagc_df_ss_hz_theory"], 5)
setm("ValTieSim", val["lfc_noagc_tie_ss_mw_sim"], 3); setm("ValTieTh", val["lfc_noagc_tie_ss_mw_theory"], 3)
setm("ValMode", val["inter_area_mode_hz_approx"], 2)


# ---- supplementary: all training runs
rows = []
for d in sorted(glob.glob("results/ppo_*_s*")):
    if not os.path.exists(f"{d}/meta.json"):
        continue
    meta = json.load(open(f"{d}/meta.json")); L = json.load(open(f"{d}/learning_curve.json"))
    st = np.array([l["step"] for l in L]); v = -10 * np.array([l["val_mean"] for l in L])
    first = v[st >= st[-1] - 50000][0]
    rows.append(f"{os.path.basename(d).replace('_', '\\_')} & {meta['args']['steps']//1000}k & {meta['cfg'] if meta['cfg'] else 'nominal'}"
                f"{', squashed' if meta['args'].get('squash') else ''} & {fmt(v[-1], 1)} & {fmt(v.min(), 1)} & {fmt(abs(v[-1]-first)/v[-1]*100, 1)} & {meta['wall_s']/60:.0f} \\\\")
txt = ("\\begin{longtable}{llrrrrr}\n\\caption{All training runs (final model version v3).}\\label{tab:s_runs}\\\\\n\\toprule\n"
       "\\textbf{Run} & \\textbf{Steps} & \\textbf{Reward config} & \\textbf{Final val. cost} & \\textbf{Best val. cost} & "
       "\\textbf{Change, last 50k (\\%)} & \\textbf{Wall (min)}\\\\\n\\midrule\n\\endhead\n" + "\n".join(rows).replace('{"', '\\{"').replace('"}', '"\\}').replace('_', '\\_').replace('\\\\_', '\\_') + "\n\\bottomrule\n\\end{longtable}\n")
open(f"{OUT}/table_s_runs.tex", "w").write(txt)
with open(f"{OUT}/macros.tex", "w") as f:
    for k_, v in sorted(mac.items()):
        f.write(f"\\newcommand{{\\{k_}}}{{{v}}}\n")
json.dump(mac, open(f"{OUT}/macros.json", "w"), indent=1)
print("macros", len(mac))
