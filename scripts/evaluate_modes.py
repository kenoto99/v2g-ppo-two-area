"""Evaluate every controller under each regulation-energy treatment on identical scenarios.
Usage: python scripts/evaluate_modes.py none window [hour fullhour]
Writes results/eval_modes/<mode>/<controller>.json. DAO day-ahead plans are cached and
shared across modes because the DAO optimiser does not model regulation energy."""
import glob, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2g.env import V2GTwoAreaEnv
from v2g.controllers import Uncontrolled, RuleBased, DayAheadOpt
from v2g.evaluate import rollout, daily_metrics
from v2g.profiles import sample_scenario
from stable_baselines3 import PPO

rt, rs = np.random.default_rng(2026), np.random.default_rng(777)
SETS = dict(test=[sample_scenario(rt, "test") for _ in range(30)], stress=[sample_scenario(rs, "stress") for _ in range(20)])
CACHE = "results/eval_modes/dao_plans.pkl"
os.makedirs("results/eval_modes", exist_ok=True)
plans = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}


class CachedDAO(DayAheadOpt):
    def reset(self, env):
        key = (env.sc["seed"], env.c["reg_energy_mode"] == "x")  # plan independent of mode
        if env.sc["seed"] in plans:
            self.plan = plans[env.sc["seed"]]
        else:
            super().reset(env); plans[env.sc["seed"]] = self.plan
            pickle.dump(plans, open(CACHE, "wb"))


def run(mode, name, controller=None, model=None):
    cfg = dict(reg_energy_mode=mode, reg_energy_diag=True)
    out = {}
    for split, scs in SETS.items():
        env = V2GTwoAreaEnv(cfg=cfg, scenario_list=scs)
        out[split] = [daily_metrics(rollout(env, controller, model), env) for _ in scs]
    os.makedirs(f"results/eval_modes/{mode}", exist_ok=True)
    json.dump(out, open(f"results/eval_modes/{mode}/{name}.json", "w"))
    t = out["test"]
    print(mode, name, "cost", round(-10 * np.mean([d["reward"] for d in t]), 2), "soc_def", round(np.mean([d["soc_def_pct"] for d in t]), 2),
          "unmet", round(np.mean([d["unmet_mwh"] for d in t]), 3), "reg_full", round(np.mean([d["reg_full_mwh"] for d in t]), 2),
          "reg_hour", round(np.mean([d["reg_hour_extrap_mwh"] for d in t]), 2), "reg_dec", round(np.mean([d["reg_dec_mwh"] for d in t]), 2), flush=True)


for mode in sys.argv[1:]:
    run(mode, "UC", Uncontrolled()); run(mode, "RB", RuleBased()); run(mode, "DAO", CachedDAO(maxiter=20))
    for d in sorted(glob.glob("results/ppo_main_s*")):
        if os.path.exists(f"{d}/meta.json"):
            run(mode, os.path.basename(d), model=PPO.load(f"{d}/best_model.zip", device="cpu"))
