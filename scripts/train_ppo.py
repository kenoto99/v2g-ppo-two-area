"""Train PPO (Stable-Baselines3) on the two-time-scale V2G environment.
Usage: python scripts/train_ppo.py --seed 0 --steps 200000 --tag main [--cfg '{"w_f":0}']"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from v2g.env import V2GTwoAreaEnv, ENV_VERSION
from v2g.profiles import sample_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=200000)
ap.add_argument("--tag", default="main")
ap.add_argument("--cfg", default="{}")
ap.add_argument("--squash", action="store_true", help="tanh-squashed Gaussian policy with gSDE (bounded actions)")
args = ap.parse_args()
cfg = json.loads(args.cfg)
torch.set_num_threads(1)
out = f"results/ppo_{args.tag}_s{args.seed}"
os.makedirs(out, exist_ok=True)

val_rng = np.random.default_rng(10_000)          # fixed validation days (not used for training)
val_sc = [sample_scenario(val_rng, "val") for _ in range(10)]


def make(i):
    return lambda: Monitor(V2GTwoAreaEnv(cfg=cfg, split="train", seed=1000 * args.seed + i))


class EvalCB(BaseCallback):
    def __init__(self, every=6144):
        super().__init__(); self.every = every; self.log = []; self.next = 0
        self.best = -np.inf

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            env = V2GTwoAreaEnv(cfg=cfg, scenario_list=val_sc)
            R = []
            for _ in range(len(val_sc)):
                o, _ = env.reset(); done = False; tot = 0.0
                while not done:
                    a, _ = self.model.predict(o, deterministic=True)
                    o, r, done, _, _ = env.step(a); tot += r
                R.append(tot)
            ep = [e["r"] for e in self.model.ep_info_buffer]
            self.log.append(dict(step=self.num_timesteps, val_mean=float(np.mean(R)), val_std=float(np.std(R)),
                                 train_mean=float(np.mean(ep)) if ep else None, time=time.time() - t0))
            if np.mean(R) > self.best:
                self.best = np.mean(R); self.model.save(f"{out}/best_model")
            json.dump(self.log, open(f"{out}/learning_curve.json", "w"))
        return True


venv = DummyVecEnv([make(i) for i in range(4)])
venv.seed(args.seed)
pk = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=torch.nn.Tanh, log_std_init=-0.5)
extra = {}
if args.squash:
    # bounded-action alternative: tanh-squashed output with generalised state-dependent exploration
    pk.update(squash_output=True)
    extra = dict(use_sde=True, sde_sample_freq=4)
model = PPO("MlpPolicy", venv, seed=args.seed, learning_rate=lambda f: 3e-4 * f,  # linear decay to 0
            n_steps=384, batch_size=256,
            n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
            max_grad_norm=0.5, policy_kwargs=pk, verbose=0, **extra)
t0 = time.time()
model.learn(total_timesteps=args.steps, callback=EvalCB())
model.save(f"{out}/final_model")
json.dump(dict(args=vars(args), cfg=cfg, env_version=ENV_VERSION, wall_s=time.time() - t0,
               hyper=dict(lr="3e-4 linear decay to 0", n_steps=384, n_envs=4, batch=256, epochs=10, gamma=0.995,
                          gae=0.95, clip=0.2, net="128-128 tanh", log_std_init=-0.5)),
          open(f"{out}/meta.json", "w"), indent=2)
print("done", args.tag, args.seed, time.time() - t0)
