"""Automated verification: physics tests, pandapower re-verification, and an
independent recomputation of every manuscript number from the raw result files.

Usage: python scripts/verify_all.py <manuscript_dir>
Writes results/verification_report.md and exits with status 1 if any check fails.
The recomputation below deliberately does NOT import scripts/make_results.py."""
import glob, hashlib, json, os, pickle, re, subprocess, sys, time, traceback
import numpy as np
from scipy.stats import wilcoxon
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
MS = sys.argv[1] if len(sys.argv) > 1 else "../manuscript"
E = "results/eval"
REPORT = []
NPASS = NFAIL = 0


def check(name, ok, detail=""):
    global NPASS, NFAIL
    NPASS += bool(ok); NFAIL += (not ok)
    REPORT.append(f"| {'PASS' if ok else '**FAIL**'} | {name} | {detail} |")
    return ok


def section(title):
    REPORT.append(f"\n## {title}\n\n| Result | Check | Detail |\n|---|---|---|")


# ------------------------------------------------------------------ 1 physics tests
section("1. Physics and bookkeeping tests (tests/test_physics.py)")
import tests.test_physics as T
for fn in [T.test_bfs_vs_pandapower, T.test_lfc_steady_state, T.test_regulation_energy_integral,
           T.test_feasibility_random_policy, T.test_feasibility_fullhour, T.test_no_reserve_from_mobility_shortfall, T.test_clip_counters_distinct]:
    try:
        out = fn()
        check(fn.__name__, True, ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()))
    except AssertionError:
        check(fn.__name__, False, traceback.format_exc(limit=1).splitlines()[-1])

# ------------------------------------------------------------------ 2 pandapower re-verification
section("2. IEEE 33-bus verification against pandapower (scripts/validate_models.py re-run)")
r = subprocess.run([sys.executable, "scripts/validate_models.py"], capture_output=True, text=True)
val = json.load(open("results/validation.json"))
check("validate_models.py runs", r.returncode == 0, f"return code {r.returncode}")
check("max |V_BFS - V_pandapower| < 1e-8 p.u.", val["bfs_vs_pandapower_max_abs_v_err_pu"] < 1e-8,
      f"{val['bfs_vs_pandapower_max_abs_v_err_pu']:.3e} p.u. over 20 random cases")
check("max loss difference < 1e-5 kW", val["bfs_vs_pandapower_max_abs_loss_err_kw"] < 1e-5,
      f"{val['bfs_vs_pandapower_max_abs_loss_err_kw']:.3e} kW")
check("base case Vmin = 0.9131 p.u. at bus 18", abs(val["bfs_base_vmin_pu"] - 0.91309) < 1e-4 and val["bfs_base_vmin_bus"] == 18,
      f"{val['bfs_base_vmin_pu']:.5f} p.u., bus {val['bfs_base_vmin_bus']}")
check("base case loss = 202.68 kW", abs(val["bfs_base_loss_kw"] - 202.677) < 0.01, f"{val['bfs_base_loss_kw']:.3f} kW")
check("LFC steady-state df = -0.0238095 Hz", abs(val["lfc_noagc_df_ss_hz_sim"] + 0.0238095238) < 1e-6,
      f"sim {val['lfc_noagc_df_ss_hz_sim']:.10f} Hz, theory {val['lfc_noagc_df_ss_hz_theory']:.10f} Hz")
check("LFC steady-state dPtie = -5 MW", abs(val["lfc_noagc_tie_ss_mw_sim"] + 5.0) < 1e-6,
      f"sim {val['lfc_noagc_tie_ss_mw_sim']:.9f} MW")

# ------------------------------------------------------------------ 3 provenance of results
section("3. Provenance: results come from the corrected model")
files = sorted(glob.glob(f"{E}/*.json"))
corr = all("reg_batt_mwh" in d and "raw_clip_events" in d for f in files for s in json.load(open(f)).values() for d in s)
check("every evaluation record contains regulation-energy and clipping fields", corr, f"{len(files)} files")
allrec = [d for f_ in files for s_ in json.load(open(f_)).values() for d in s_]
check("energy balance closes in every evaluated hour (max |error| < 1e-9 MWh)", max(d["energy_closure_err_mwh"] for d in allrec) < 1e-9,
      f"max {max(d['energy_closure_err_mwh'] for d in allrec):.1e} MWh over {len(allrec)} controller-days")
check("scheduled plus droop power never exceeds charger or energy limits", max(d["pw_excess_max_mw"] for d in allrec) == 0.0,
      f"max excess {max(d['pw_excess_max_mw'] for d in allrec):.3e} MW")
check("droop power never exceeds the committed reserve", max(d["res_excess_max_mw"] for d in allrec) < 1e-6)
check("no SoC-limit violation caused by regulation", max(d["e_viol_mwh"] for d in allrec) == 0.0)
check("sustained-activation margin never negative", min(d["sust_margin_min_mwh"] for d in allrec) > -1e-9,
      f"min margin {min(d['sust_margin_min_mwh'] for d in allrec):.3e} MWh")
check("raw-action clipping and projection counted separately", all("raw_clip_events" in d and "proj_events" in d for d in allrec))
env_src = open("v2g/env.py").read()
check("environment default reg_energy_mode = 'decomposed' and ENV_VERSION = v3", 'reg_energy_mode="decomposed"' in env_src and 'ENV_VERSION = "v3"' in env_src)
mt_env = os.path.getmtime("v2g/env.py")
stale = [f for f in files if os.path.getmtime(f) < mt_env]
check("all evaluation files newer than the corrected environment", not stale, ", ".join(os.path.basename(s) for s in stale) or "none stale")
for d in sorted(glob.glob("results/ppo_*_s*")):
    meta = json.load(open(f"{d}/meta.json")) if os.path.exists(f"{d}/meta.json") else None
    ok = meta is not None and meta.get("env_version") == "v3"
    check(f"{os.path.basename(d)} trained in environment v3 (meta.env_version)", ok,
          f"env_version={meta.get('env_version') if meta else 'missing meta'}, wall {meta['wall_s']/60:.0f} min" if meta else "")
seeds = sorted(int(os.path.basename(d).split("_s")[-1]) for d in glob.glob("results/ppo_main_s*") if os.path.exists(f"{d}/meta.json"))
check("five final main seeds 0-4 present", seeds == [0, 1, 2, 3, 4], f"seeds {seeds}")

# ------------------------------------------------------------------ 4 independent recomputation
section("4. Manuscript numbers recomputed independently from raw results")
macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(f"{MS}/results_macros.tex").read()))
CTRL = ["UC", "RB", "DAO", "PPO"]
base = {c: json.load(open(f"{E}/{c}.json")) for c in ["UC", "RB", "DAO"]}
ppo = [json.load(open(f)) for f in sorted(glob.glob(f"{E}/ppo_main_s*.json"))]


def vals(split, c, key):
    if c == "PPO":
        a = np.array([[(-10 * d["reward"] if key == "COST" else d[key]) for d in r[split]] for r in ppo])
    else:
        a = np.array([(-10 * d["reward"] if key == "COST" else d[key]) for d in base[c][split]])
    return a


def f(x, d):
    return f"{x:.{d}f}"


METRIC = {"Cost": ("COST", 1, 1), "ViolHours": ("viol_hours", 1, 1), "UvPuH": ("uv_pu_h", 1, 2),
          "OvPuH": ("ov_pu_h", 1, 3), "VminDay": ("vmin_day", 1, 3), "VmaxDay": ("vmax_day", 1, 3),
          "NadirMeanHz": ("nadir_mean_hz", 1e3, 1), "NadirMaxHz": ("nadir_max_hz", 1e3, 1), "FRmsMhz": ("f_rms_mhz", 1, 2),
          "RocofMax": ("rocof_max", 1, 3), "TieRmsMw": ("tie_rms_mw", 1, 2), "ReserveMeanMw": ("reserve_mean_mw", 1, 1),
          "ThroughputMwh": ("throughput_mwh", 1, 0), "SocDefPct": ("soc_def_pct", 1, 2), "ProjEvents": ("proj_events", 1, 1),
          "LossMwh": ("loss_mwh", 1, 3), "UnmetMwh": ("unmet_mwh", 1, 2)}
checked = set(); mism = []


def cmp(name, expected):
    checked.add(name)
    got = macros.get(name)
    if got is None:
        mism.append(f"{name}: missing")
    elif got != expected:
        mism.append(f"{name}: manuscript {got} vs recomputed {expected}")


for S, split in [("T", "test"), ("S", "stress")]:
    for c in CTRL:
        for nm, (k, sc, d) in METRIC.items():
            v = vals(split, c, k) * sc
            cmp(f"{S}{c}{nm}", f(v.mean(), d))
            cmp(f"{S}{c}{nm}Sd", f(v.reshape(-1).std(ddof=1), d))
    pc = vals(split, "PPO", "COST").mean(axis=0)
    for c in ["UC", "RB", "DAO"]:
        bc = vals(split, c, "COST")
        cmp(f"{S}RedCost{c}", f(100 * (1 - pc.mean() / bc.mean()), 1))
        cmp(f"{S}Wins{c}", str(int(np.sum(pc < bc))))
        for nm in ["UvPuH", "NadirMeanHz", "ThroughputMwh", "ViolHours", "NadirMaxHz", "RocofMax"]:
            k, sc, d = METRIC[nm]
            p_, b_ = vals(split, "PPO", k).mean(), vals(split, c, k).mean()
            cmp(f"{S}Red{nm}{c}", f(100 * (1 - p_ / b_) if b_ != 0 else 0.0, 1))
        pv = wilcoxon(pc, bc).pvalue
        exp = f"{pv:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}" if pv < 1e-3 else f"{pv:.3f}"
        cmp(f"{S}P{c}", exp)
# regulation energy and audit
for c in CTRL:
    cum = vals("test", c, "reg_batt_mwh").reshape(-1)
    cmp(f"Reg{c}CumMean", f(cum.mean(), 2))
    sch = vals("test", c, "sched_batt_abs_mwh").reshape(-1)
    cmp(f"Reg{c}Share", f(100 * np.abs(cum).sum() / sch.sum(), 2))
    runs_c = ppo if c == "PPO" else [base[c]]
    hb = np.array([np.array(d["hourly"]["reg_batt_mwh"]).sum(axis=1) for r in runs_c for d in r["test"]])
    cmp(f"Reg{c}HrMax", f(np.abs(hb).max(), 3)); cmp(f"Reg{c}HrMean", f(np.abs(hb).mean(), 3))
    cmp(f"Reg{c}PosDaysPct", f(100 * np.mean(cum > 0), 0))
    cmp(f"Reg{c}ShareGross", f(100 * sum(np.abs(np.array(d["hourly"]["reg_batt_mwh"])).sum() for r in runs_c for d in r["test"]) / sch.sum(), 2))
    cmp(f"Reg{c}WinCum", f(vals("test", c, "reg_batt_mwh_w").reshape(-1).mean(), 4))
    for S, split in [("T", "test"), ("S", "stress")]:
        cmp(f"{S}{c}RawClip", str(int(vals(split, c, "raw_clip_events").sum())))
        cmp(f"{S}{c}Proj", str(int(vals(split, c, "proj_events").sum())))
# ablation / sensitivity
TAGN = {"ref200": "Ref", "noOV": "NoOV", "noF": "NoF", "noDeg": "NoDeg", "wV25": "WvLow", "wV100": "WvHigh", "wF2": "WfLow", "wF8": "WfHigh", "squash": "Squash"}
for tag, nm in TAGN.items():
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"{E}/ppo_{tag}_s*.json"))]
    if not runs:
        continue
    cs = np.array([-10 * np.mean([d["reward"] for d in r["test"]]) for r in runs])
    cmp(f"Abl{nm}Cost", f(cs.mean(), 1)); cmp(f"Abl{nm}N", str(len(runs)))
    for key, kk, sc, d in [("ReserveMeanMw", "reserve_mean_mw", 1, 1), ("NadirMeanHz", "nadir_mean_hz", 1e3, 1),
                           ("SocDefPct", "soc_def_pct", 1, 2), ("UvPuH", "uv_pu_h", 1, 2), ("ThroughputMwh", "throughput_mwh", 1, 0)]:
        v = sc * np.array([np.mean([d_[kk] for d_ in r["test"]]) for r in runs])
        cmp(f"Abl{nm}{key}", f(v.mean(), d))
# regulation-energy treatments and per-dimension clipping
MODES = ["none", "window", "hour", "decomposed", "fullhour"]
if all(os.path.exists(f"results/eval_modes/{m_}/UC.json") for m_ in MODES):
    for mode in MODES:
        for c in CTRL:
            if c == "PPO":
                runs = [json.load(open(p)) for p in sorted(glob.glob(f"results/eval_modes/{mode}/ppo_main_s*.json"))]
            else:
                runs = [json.load(open(f"results/eval_modes/{mode}/{c}.json"))]
            for key, nm, sc, d in [("COST", "Cost", 1, 1), ("soc_def_pct", "SocDefPct", 1, 2), ("unmet_mwh", "UnmetMwh", 1, 2),
                                   ("uv_pu_h", "UvPuH", 1, 3), ("nadir_mean_hz", "NadirMeanHz", 1e3, 1), ("reg_batt_mwh", "RegBattMwh", 1, 2)]:
                vv = np.array([(-10 * x["reward"] if key == "COST" else x[key]) for r in runs for x in r["test"]]) * sc
                cmp(f"Mode{mode.title()}{c}{nm}", f(vv.mean(), d))
    check("regulation-energy treatment evaluations present for all five treatments", True, "results/eval_modes/*")
else:
    check("regulation-energy treatment evaluations present for all five treatments", False)
cnt = np.sum([np.array(d["raw_clip_by_dim"]) for r in ppo for d in r["test"]], axis=0); n = sum(len(r["test"]) * 24 for r in ppo)
for i, nm in enumerate(["PsetOne", "PsetTwo", "RhoOne", "RhoTwo"]):
    cmp(f"ClipDim{nm}", f(cnt[i] / n * 100, 1))
# validation / verification numbers
cmp("ValVmin", f(val["bfs_base_vmin_pu"], 4)); cmp("ValLoss", f(val["bfs_base_loss_kw"], 2))
cmp("ValDfSim", f(val["lfc_noagc_df_ss_hz_sim"], 5)); cmp("ValTieSim", f(val["lfc_noagc_tie_ss_mw_sim"], 3))
# trace-derived values (representative day, frequency hour)
tr = {c: pickle.load(open(f"{E}/trace_{c}_test.pkl", "rb")) for c in ["UC", "RB"]}
tr["PPO"] = pickle.load(open(f"{E}/trace_ppo_main_s0_test.pkl", "rb"))
h = [k for k, r_ in enumerate(tr["UC"]) if r_["hour"] == 13][0]
for c in ["UC", "RB", "PPO"]:
    cmp(f"Hr{c}NadirOne", f(1e3 * tr[c][h]["nadir_hz"][0], 1)); cmp(f"Hr{c}NadirTwo", f(1e3 * tr[c][h]["nadir_hz"][1], 1))
    cmp(f"Hr{c}Res", f(tr[c][h]["reserve_mw_area"].sum(), 1))
cmp("DayUCVminOne", f(min(r_["vmin"][0] for r_ in tr["UC"]), 2))
check("all recomputed macros equal the manuscript values", not mism,
      f"{len(checked)} macros recomputed; " + ("; ".join(mism[:12]) if mism else "0 mismatches"))
unchecked = sorted(set(macros) - checked)
REPORT.append(f"\nMacros not independently recomputed ({len(unchecked)}; formatting helpers, counts of seeds/days, learning-curve "
              f"statistics checked in section 5): {', '.join(unchecked)}\n")

# tables: parse every numeric cell and compare with recomputation
section("5. Tables, learning curves and figures")
def table_cells(path):
    rows = {}
    for line in open(path):
        if "&" in line and "\\\\" in line and "textbf" not in line:
            parts = [p.strip() for p in line.split("\\\\")[0].split("&")]
            rows[parts[0]] = parts[1:]
    return rows
TLAB = {"Daily cost": "Cost", "Voltage-violation hours": "ViolHours", "Undervoltage": "UvPuH", "Overvoltage": "OvPuH",
        "Minimum voltage": "VminDay", "Maximum voltage": "VmaxDay", "Mean hourly nadir": "NadirMeanHz", "Worst nadir": "NadirMaxHz",
        "RMS frequency": "FRmsMhz", "Maximum RoCoF": "RocofMax", "RMS tie-line": "TieRmsMw", "Mean V2G reserve": "ReserveMeanMw",
        "Fleet throughput": "ThroughputMwh", "Terminal SoC": "SocDefPct", "Projection events": "ProjEvents", "Energy loss": "LossMwh",
        "Unmet mobility": "UnmetMwh"}
for tab, split in [("table_test.tex", "test"), ("table_stress.tex", "stress")]:
    bad = []; n = 0
    for lab, cells in table_cells(f"{MS}/{tab}").items():
        key = next((v for k, v in TLAB.items() if lab.startswith(k)), None)
        if key is None:
            continue
        k, sc, d = METRIC[key]
        for c, cell in zip(CTRL, cells):
            v = vals(split, c, k) * sc
            exp = f"{f(v.mean(), d)} $\\pm$ {f(v.reshape(-1).std(ddof=1), d)}"
            if c == "PPO":
                exp += f" ({f(v.mean(axis=1).std(ddof=1), d)})"
            n += 1
            if cell != exp:
                bad.append(f"{lab[:20]}/{c}: {cell} vs {exp}")
    check(f"{tab}: every cell equals recomputation", not bad and n > 0, f"{n} cells; " + ("; ".join(bad[:5]) if bad else "0 mismatches"))
# generated artefacts identical between code and manuscript folders
def h_(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()
for t in sorted(glob.glob("results/tex/table_*.tex")):
    m = f"{MS}/{os.path.basename(t)}"
    check(f"{os.path.basename(t)} in manuscript identical to generated file", os.path.exists(m) and h_(m) == h_(t))
check("results_macros.tex identical to generated macros.tex", h_(f"{MS}/results_macros.tex") == h_("results/tex/macros.tex"))
newest_eval = max(os.path.getmtime(f_) for f_ in files)
for fig in sorted(glob.glob("figures/*.pdf")):
    m = f"{MS}/figures/{os.path.basename(fig)}"
    fresh = os.path.getmtime(fig) >= newest_eval or os.path.basename(fig) in ("fig_feeder.pdf", "fig_profiles.pdf")
    check(f"{os.path.basename(fig)} regenerated after final evaluation and copied", fresh and os.path.exists(m) and h_(m) == h_(fig))
# learning-curve statistics
DONE = [d for d in sorted(glob.glob("results/ppo_main_s*")) if os.path.exists(f"{d}/meta.json")]
V = [-10 * np.array([l["val_mean"] for l in json.load(open(f"{d}/learning_curve.json"))]) for d in DONE]
check("FinalValCost", macros.get("FinalValCost") == f(np.mean([v[-1] for v in V]), 1), f"{macros.get('FinalValCost')}")

# ------------------------------------------------------------------ 6 manuscript text scan
section("6. Manuscript text: literal numbers outside macros")
WHITELIST = set("""0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 50 60 100 128 256 300 384 500
0.01 0.02 0.03 0.05 0.06 0.08 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.92 0.94 0.95 0.965 1.0 1.05 1.035 1.1 1.2 1.3 1.6 2.0 4.0 5.0
0.20 0.35 0.40 0.45 0.50 0.55 0.60 0.70 0.85 0.90 1.10 1.20 1.5 4.2 7.4 9 12.66 18 0.006 0.995 0.25 0.3 2.6 0.0 6144 50000 200000 400000 1989 2016
00 06 09 15 21 1.2 3 10 40 400 80 33-bus 2026 0.10 05 3600 90 0.34 0.60 0.0 1.0 0.9 0.5 186 190 45 59 65 0.001 10.5281 23851585""".split())
txt = "".join(open(f"{MS}/{p}").read() for p in ["sec_abstract.tex", "sec_intro_methods.tex", "sec_results.tex"])
txt = re.sub(r"%.*", "", txt)
txt = re.sub(r"\\(cite|ref|eqref|label|includegraphics)(\[[^]]*\])?\{[^}]*\}", "", txt)
lits = re.findall(r"(?<![\\\w{.])(\d+(?:\.\d+)?)(?![\w}])", txt.replace("\\,", ""))
odd = sorted(set(l for l in lits if l not in WHITELIST))
check("no result-like literal numbers outside the whitelist of model parameters", not odd,
      "non-whitelisted literals: " + (", ".join(odd) if odd else "none"))
log = open(f"{MS}/access.log", errors="ignore").read() if os.path.exists(f"{MS}/access.log") else ""
check("manuscript compiles without undefined control sequences or references",
      "Undefined control sequence" not in log and "There were undefined references" not in log and os.path.exists(f"{MS}/access.pdf"))

# ------------------------------------------------------------------ write report
head = [f"# Verification report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/verify_all.py`.", "",
        f"**Summary: {NPASS} PASS, {NFAIL} FAIL.**", ""]
open("results/verification_report.md", "w").write("\n".join(head + REPORT) + "\n")
print(f"PASS {NPASS}  FAIL {NFAIL}")
sys.exit(1 if NFAIL else 0)
