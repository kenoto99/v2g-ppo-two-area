"""Evaluate all controllers on held-out test days (in-distribution) and stress
days (out-of-distribution). Usage: python scripts/evaluate_all.py UC RB DAO PPO ABL"""
import json, os, sys, glob, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2g.env import V2GTwoAreaEnv
from v2g.controllers import Uncontrolled, RuleBased, DayAheadOpt
from v2g.evaluate import rollout, daily_metrics
from v2g.profiles import sample_scenario

N_TEST, N_STRESS = 30, 20
rt, rs, rv = np.random.default_rng(2026), np.random.default_rng(777), np.random.default_rng(10_000)
SETS = dict(test=[sample_scenario(rt, "test") for _ in range(N_TEST)],
            stress=[sample_scenario(rs, "stress") for _ in range(N_STRESS)],
            val=[sample_scenario(rv, "val") for _ in range(10)])   # same days as training-time validation
os.makedirs("results/eval", exist_ok=True)


def run(name, controller=None, model=None, cfg=None, trace_day=True):
    cfg = dict(dict(reg_energy_diag=True), **(cfg or {}))   # exact full-hour energy recorded for every hour
    out = {}
    for split, scs in SETS.items():
        env = V2GTwoAreaEnv(cfg=cfg, scenario_list=scs)
        M = []
        for k in range(len(scs)):
            keep = trace_day and k == 0
            if keep:
                env.keep_traces = True
            rows = rollout(env, controller, model, keep=keep)
            env.keep_traces = False
            M.append(daily_metrics(rows, env))
            if keep:
                pickle.dump(rows, open(f"results/eval/trace_{name}_{split}.pkl", "wb"))
        out[split] = M
    json.dump(out, open(f"results/eval/{name}.json", "w"))
    t = {k: round(float(np.mean([m[k] for m in out["test"]])), 4) for k in out["test"][0] if k != "hourly"}
    print(name, t, flush=True)


if __name__ == "__main__":
    which = sys.argv[1:]
    if "UC" in which: run("UC", Uncontrolled())
    if "RB" in which: run("RB", RuleBased())
    if "DAO" in which: run("DAO", DayAheadOpt(maxiter=20))
    if "PPO" in which or "ABL" in which:
        from stable_baselines3 import PPO
        tags = ["main"] if "PPO" in which else []
        if "ABL" in which: tags += ["ref200", "noOV", "noF", "noDeg", "wV25", "wV100", "wF2", "wF8", "squash"]
        for tag in tags:
            for d in sorted(glob.glob(f"results/ppo_{tag}_s*")):
                if not os.path.exists(f"{d}/final_model.zip"):
                    continue
                outf = f"results/eval/{os.path.basename(d)}.json"
                if tag != "main" and os.path.exists(outf) and os.path.getmtime(outf) > os.path.getmtime(f"{d}/final_model.zip"):
                    continue   # already evaluated
                meta = json.load(open(f"{d}/meta.json"))
                model = PPO.load(f"{d}/best_model.zip", device="cpu")
                # evaluate under the default reward so costs are comparable
                run(os.path.basename(d), model=model, trace_day=(tag == "main"))
