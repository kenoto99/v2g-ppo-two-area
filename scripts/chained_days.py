"""Consecutive-day evaluation with carried-over fleet energy.

In the main evaluation every test day starts from the initial SoC of its
scenario. Here the 30 test days are run back to back: the terminal energy of
day k becomes the initial energy of day k+1, so a terminal SoC deficit cannot
be "forgiven" by the next day's reset. The terminal term of the cost keeps its
definition (deficit relative to the energy at the start of that day). The
controllers see exactly the same load, PV, availability and disturbance
sequences as in Table test; only the initial energy of days 2 to 30 changes.

Usage: python scripts/chained_days.py   (writes results/eval/chained_days.json)
"""
import copy, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2g.env import V2GTwoAreaEnv
from v2g.controllers import RuleBased, DayAheadOpt
from v2g.evaluate import rollout, daily_metrics
from v2g.profiles import sample_scenario

N_TEST = 30
rt = np.random.default_rng(2026)
SCS = [sample_scenario(rt, "test") for _ in range(N_TEST)]


def run(controller=None, model=None):
    scs = copy.deepcopy(SCS)
    days = []
    soc = None
    for k in range(N_TEST):
        if soc is not None:
            scs[k]["soc0"] = soc.copy()
        env = V2GTwoAreaEnv(cfg=dict(reg_energy_diag=False), scenario_list=[scs[k]])
        rows = rollout(env, controller, model)
        m = daily_metrics(rows, env); m.pop("hourly", None)
        m["soc0_1"], m["soc0_2"] = float(scs[k]["soc0"][0]), float(scs[k]["soc0"][1])
        m["soc_min_day"] = float(min(r["soc"].min() for r in rows))
        days.append(m)
        soc = np.array([m["soc_end_1"], m["soc_end_2"]])
    return days


def summarise(days):
    c = -10 * np.array([d["reward"] for d in days])
    e1 = np.array([d["soc_end_1"] for d in days]); e2 = np.array([d["soc_end_2"] for d in days])
    return dict(cost_mean=float(c.mean()), cost_sd=float(c.std(ddof=1)),
                soc_end_final_1=float(e1[-1]), soc_end_final_2=float(e2[-1]),
                soc_end_min=float(min(e1.min(), e2.min())), soc_end_mean=float(np.mean([e1, e2])),
                soc_end_first_1=float(e1[0]), soc_end_first_2=float(e2[0]),
                soc_min_seq=float(min(d["soc_min_day"] for d in days)),
                soc_def_pct_mean=float(np.mean([d["soc_def_pct"] for d in days])),
                unmet_mwh_total=float(sum(d["unmet_mwh"] for d in days)),
                unmet_days=int(sum(d["unmet_mwh"] > 1e-9 for d in days)),
                uv_pu_h_mean=float(np.mean([d["uv_pu_h"] for d in days])),
                viol_hours_mean=float(np.mean([d["viol_hours"] for d in days])),
                soc_end_1=e1.tolist(), soc_end_2=e2.tolist(), cost=c.tolist())


if __name__ == "__main__":
    out = {}
    out["RB"] = summarise(run(RuleBased()))
    print("RB", {k: round(v, 3) for k, v in out["RB"].items() if not isinstance(v, list)}, flush=True)
    out["DAO"] = summarise(run(DayAheadOpt(maxiter=20)))
    print("DAO", {k: round(v, 3) for k, v in out["DAO"].items() if not isinstance(v, list)}, flush=True)
    from stable_baselines3 import PPO
    seeds = []
    for d in sorted(glob.glob("results/ppo_main_s*")):
        m = PPO.load(f"{d}/best_model.zip", device="cpu")
        seeds.append(summarise(run(model=m)))
        print(os.path.basename(d), {k: round(v, 3) for k, v in seeds[-1].items() if not isinstance(v, list)}, flush=True)
    out["PPO_seeds"] = seeds
    scal = [k for k in seeds[0] if not isinstance(seeds[0][k], list)]
    out["PPO"] = {k: float(np.mean([s[k] for s in seeds])) for k in scal}
    out["PPO_sd"] = {k: float(np.std([s[k] for s in seeds], ddof=1)) for k in scal}
    json.dump(out, open("results/eval/chained_days.json", "w"), indent=1)
