"""Feeder power flow under full reserve activation.

The slow layer evaluates feeder voltages at the scheduled V2G power P_set of
each hour, while the fast layer applies the droop power on top of that
schedule during a disturbance. This script re-solves the AC power flow of
every test-day hour with the scheduled power shifted by the full reserve in
each direction,

    upward   (frequency low, fleet discharges more):  P_set + R
    downward (frequency high, fleet charges more):    P_set - R

and records the resulting bus voltages and branch currents. It answers the
question whether a momentary activation of the committed reserve can push the
feeder outside the voltage band or raise branch currents materially above the
scheduled operating point. The check is a worst case: the droop loop saturates
at R only for a few seconds after the 50 MW step (Section IV), so the sustained
operating point stays at P_set.

Usage: python scripts/activation_powerflow.py   (writes results/eval/activation_pf.json)
"""
import glob, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2g.env import V2GTwoAreaEnv
from v2g.feeder import S_BASE_MVA, V_BASE_KV
from v2g.controllers import RuleBased, DayAheadOpt
from v2g.profiles import sample_scenario

N_TEST = 30
rt = np.random.default_rng(2026)
SCS = [sample_scenario(rt, "test") for _ in range(N_TEST)]
I_BASE_KA = S_BASE_MVA / (np.sqrt(3) * V_BASE_KV)
CACHE = "results/eval_modes/dao_plans.pkl"
plans = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}


class CachedDAO(DayAheadOpt):
    def reset(self, env):
        if env.sc["seed"] in plans:
            self.plan = plans[env.sc["seed"]]
        else:
            super().reset(env); plans[env.sc["seed"]] = self.plan


def policy_action(model, obs):
    import torch
    with torch.no_grad():
        ot, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(ot)
        return dist.distribution.mean.cpu().numpy()[0]


def run(name, controller=None, model=None):
    env = V2GTwoAreaEnv(cfg=dict(reg_energy_diag=False), scenario_list=SCS)
    env.keep_traces = True          # realised droop power of the simulated 60-s window
    c = env.c
    rec = []           # one row per (day, hour, area)
    for k in range(N_TEST):
        obs, _ = env.reset()
        if controller is not None:
            controller.reset(env)
        done = False
        while not done:
            a = policy_action(model, obs) if model is not None else controller.act(env)
            t = env.h
            obs, r, done, _, info = env.step(a)
            p_set = info["p_set_mw"]; res = info["reserve_mw_area"] / c["n_feeders"]
            pev_f = info["trace"]["pev_mw"] / c["n_feeders"]      # per-feeder droop power during the window
            for i in range(2):
                row = dict(day=k, hour=int(t), area=i + 1, p_set=float(p_set[i]), res=float(res[i]),
                           p_dis_av=float(info["p_rate_mw"][i]), pev_peak_up=float(pev_f[:, i].max()),
                           pev_peak_down=float(pev_f[:, i].min()))
                for tag, pev in (("sched", p_set[i]), ("up", p_set[i] + res[i]), ("down", p_set[i] - res[i]),
                                 ("upreal", p_set[i] + pev_f[:, i].max()), ("downreal", p_set[i] + pev_f[:, i].min())):
                    vm, loss, ib = env.feeder.branch_currents(*env._injections(i, t, pev))
                    v = vm[1:]
                    row[f"vmin_{tag}"] = float(v.min()); row[f"vmax_{tag}"] = float(v.max())
                    row[f"uv_{tag}"] = float(np.sum(np.maximum(0, c["v_lo"] - v)))
                    row[f"ov_{tag}"] = float(np.sum(np.maximum(0, v - c["v_hi"])))
                    row[f"imax_{tag}_ka"] = float(ib.max() * I_BASE_KA)
                    row[f"isub_{tag}_ka"] = float(ib[0] * I_BASE_KA)
                    row[f"loss_{tag}_mw"] = float(loss)
                rec.append(row)
    return rec


def summarise(rec, c_lo=0.95, c_hi=1.05):
    R = {k: np.array([r[k] for r in rec]) for k in rec[0]}
    n = len(rec)
    out = dict(n_area_hours=n)
    for tag in ("sched", "up", "down", "upreal", "downreal"):
        vio = (R[f"vmin_{tag}"] < c_lo - 1e-9) | (R[f"vmax_{tag}"] > c_hi + 1e-9)
        out[f"viol_frac_{tag}"] = float(vio.mean())
        out[f"vmin_{tag}"] = float(R[f"vmin_{tag}"].min())
        out[f"vmax_{tag}"] = float(R[f"vmax_{tag}"].max())
        out[f"uv_sum_{tag}"] = float(R[f"uv_{tag}"].sum() / N_TEST)       # p.u.·h per day (both areas)
        out[f"ov_sum_{tag}"] = float(R[f"ov_{tag}"].sum() / N_TEST)
        out[f"imax_{tag}_ka"] = float(R[f"imax_{tag}_ka"].max())
        out[f"isub_mean_{tag}_ka"] = float(R[f"isub_{tag}_ka"].mean())
    # hours whose band status changes because of activation
    sched_ok = ~((R["vmin_sched"] < c_lo - 1e-9) | (R["vmax_sched"] > c_hi + 1e-9))
    up_bad = (R["vmin_up"] < c_lo - 1e-9) | (R["vmax_up"] > c_hi + 1e-9)
    dn_bad = (R["vmin_down"] < c_lo - 1e-9) | (R["vmax_down"] > c_hi + 1e-9)
    out["new_viol_frac_up"] = float((sched_ok & up_bad).mean())
    out["new_viol_frac_down"] = float((sched_ok & dn_bad).mean())
    out["dvmin_down_max"] = float((R["vmin_sched"] - R["vmin_down"]).max())   # largest voltage drop caused by downward activation
    out["dvmax_up_max"] = float((R["vmax_up"] - R["vmax_sched"]).max())       # largest rise caused by upward activation
    out["res_mean_mw"] = float(R["res"].mean()); out["res_max_mw"] = float(R["res"].max())
    dnr_bad = (R["vmin_downreal"] < c_lo - 1e-9) | (R["vmax_downreal"] > c_hi + 1e-9)
    out["new_viol_frac_downreal"] = float((sched_ok & dnr_bad).mean())
    out["dvmin_downreal_max"] = float((R["vmin_sched"] - R["vmin_downreal"]).max())
    out["peak_act_frac_mean"] = float(np.mean(np.maximum(R["pev_peak_up"], -R["pev_peak_down"]) / np.maximum(R["res"], 1e-9)))
    out["peak_act_frac_max"] = float(np.max(np.maximum(R["pev_peak_up"], -R["pev_peak_down"]) / np.maximum(R["res"], 1e-9)))
    out["peak_act_mw_mean"] = float(np.mean(np.maximum(R["pev_peak_up"], -R["pev_peak_down"])))
    out["di_up_max_ka"] = float((R["imax_up_ka"] - R["imax_sched_ka"]).max())
    out["di_down_max_ka"] = float((R["imax_down_ka"] - R["imax_sched_ka"]).max())
    out["imax_ratio_down"] = float((R["imax_down_ka"] / np.maximum(R["imax_sched_ka"], 1e-9)).max())
    return out


if __name__ == "__main__":
    os.makedirs("results/eval", exist_ok=True)
    env0 = V2GTwoAreaEnv(cfg=dict(reg_energy_diag=False), scenario_list=SCS)
    # reference: nominal Baran-Wu loading (load factor 1, no PV, no EV)
    vm, loss, ib = env0.feeder.branch_currents(env0.feeder.p0 * S_BASE_MVA, env0.feeder.q0 * S_BASE_MVA)
    ref = dict(isub_nominal_ka=float(ib[0] * I_BASE_KA), vmin_nominal=float(vm[1:].min()))
    results = dict(reference=ref)
    results["RB"] = summarise(run("RB", RuleBased()))
    results["DAO"] = summarise(run("DAO", CachedDAO(maxiter=20)))
    from stable_baselines3 import PPO
    per_seed = []
    for d in sorted(glob.glob("results/ppo_main_s*")):
        m = PPO.load(f"{d}/best_model.zip", device="cpu")
        per_seed.append(summarise(run(os.path.basename(d), model=m)))
    results["PPO_seeds"] = per_seed
    results["PPO"] = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]}
    results["PPO_sd"] = {k: float(np.std([s[k] for s in per_seed], ddof=1)) for k in per_seed[0]}
    json.dump(results, open("results/eval/activation_pf.json", "w"), indent=1)
    for c in ("RB", "DAO", "PPO"):
        s = results[c]
        print(c, {k: round(s[k], 4) for k in ("viol_frac_sched", "viol_frac_up", "viol_frac_down", "new_viol_frac_up",
                                                "new_viol_frac_down", "dvmin_down_max", "dvmax_up_max", "vmin_down",
                                                "vmax_up", "imax_sched_ka", "imax_down_ka", "imax_ratio_down", "viol_frac_downreal", "new_viol_frac_downreal",
                                                "dvmin_downreal_max", "peak_act_frac_mean", "peak_act_frac_max")})
    print("ref", ref)
