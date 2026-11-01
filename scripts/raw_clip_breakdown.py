"""Per-component raw-action clipping of the deterministic PPO policies on the test days.
Writes results/raw_clip_breakdown.json."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from stable_baselines3 import PPO
from v2g.env import V2GTwoAreaEnv
from v2g.profiles import sample_scenario

rt = np.random.default_rng(2026)
test = [sample_scenario(rt, "test") for _ in range(30)]
names = ["p1", "p2", "rho1", "rho2"]
cnt_hi = np.zeros(4); cnt_lo = np.zeros(4); n = 0; excess = []
for d in sorted(p for p in glob.glob("results/ppo_main_s*") if os.path.exists(f"{p}/meta.json")):
    model = PPO.load(f"{d}/best_model.zip", device="cpu")
    env = V2GTwoAreaEnv(scenario_list=test)
    for _ in test:
        o, _ = env.reset(); done = False
        while not done:
            with torch.no_grad():
                ot, _ = model.policy.obs_to_tensor(o)
                dist = model.policy.get_distribution(ot)
                a = dist.mode().cpu().numpy()[0] if getattr(model.policy, "squash_output", False) else dist.distribution.mean.cpu().numpy()[0]
            cnt_hi += a > 1; cnt_lo += a < -1; n += 1
            excess.append(np.maximum(np.abs(a) - 1, 0))
            o, _, done, _, _ = env.step(a)
ex = np.array(excess)
out = dict(steps=n, components=names, frac_above=(cnt_hi / n).tolist(), frac_below=(cnt_lo / n).tolist(),
           frac_any=float(np.mean((ex > 0).any(axis=1))), frac_components=float(np.mean(ex > 0)),
           median_excess_when_clipped=float(np.median(ex[ex > 0])) if (ex > 0).any() else 0.0)
json.dump(out, open("results/raw_clip_breakdown.json", "w"), indent=1)
print(out)
