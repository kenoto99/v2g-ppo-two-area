"""Benchmark controllers evaluated in the same environment as the PPO agent.

UC   : uncontrolled charging. Plugged-in EVs charge at full rate until SoC_max;
       no V2G and no frequency reserve.
RB   : conventional rule-based controller. Local volt-watt characteristic
       using the forecast feeder voltages, valley/PV charging for SoC
       restoration, and a fixed 30 % droop reserve.
DAO  : day-ahead optimisation with perfect foresight of load, PV and
       availability. Hourly set-points of each area are optimised over the
       24-h horizon with the AC power flow in the loop (voltage, throughput
       and terminal-SoC costs); the droop reserve is fixed at 30 %.
"""
import numpy as np
from scipy.optimize import minimize


def rho_to_action(rho, rho_max):
    return 2.0 * rho / rho_max - 1.0


class Uncontrolled:
    name = "UC"

    def reset(self, env):
        pass

    def act(self, env):
        return np.array([-1.0, -1.0, -1.0, -1.0])


class RuleBased:
    name = "RB"

    def __init__(self, rho=0.3, v_on=0.965, v_full=0.94, vh_on=1.035, vh_full=1.05, soc_guard=0.40):
        self.rho, self.v_on, self.v_full, self.vh_on, self.vh_full = rho, v_on, v_full, vh_on, vh_full
        self.soc_guard = soc_guard

    def reset(self, env):
        pass

    def act(self, env):
        t = env.h
        p = np.zeros(2)
        for i in range(2):
            vmin, vmax = env.v0[i, t]
            up = np.clip((self.v_on - vmin) / (self.v_on - self.v_full), 0, 1)
            dn = np.clip((vmax - self.vh_on) / (self.vh_full - self.vh_on), 0, 1)
            p[i] = up - dn
            if p[i] == 0.0 and env.soc[i] < env.sc["soc0"][i] + 0.1:
                # SoC restoration: off-peak night charging or PV-hour charging
                if t < 6 or (i == 1 and 9 <= t < 15):
                    p[i] = -1.0
            if env.soc[i] < self.soc_guard:      # mobility-energy guard
                p[i] = min(p[i], 0.0)
        r = rho_to_action(self.rho, env.c["rho_max"])
        return np.array([p[0], p[1], r, r])


class DayAheadOpt:
    name = "DAO"

    def __init__(self, rho=0.3, maxiter=60):
        self.rho, self.maxiter = rho, maxiter

    def _area_cost(self, env, i, p):
        c, sc = env.c, env.sc
        e = env.sc["soc0"][i] * env.e_cap
        cost = 0.0
        p_rate = env.p_rate_full * sc["avail"][i] * sc["participation"][i]
        for k in range(24):
            t = (k + c["start_hour"]) % 24
            ead = e - env.e_drive[i, t]
            pdis = max(0.0, min(p_rate[t], (ead - c["soc_min"] * env.e_cap) * c["eta_dis"]))
            pch = max(0.0, min(p_rate[t], (c["soc_max"] * env.e_cap - ead) / c["eta_ch"]))
            ps = np.clip(p[k] * (1 - self.rho) * p_rate[t], -pch, pdis)
            vm, _ = env.feeder.solve(*env._injections(i, t, ps))
            v = vm[1:]
            cost += c["w_v"] * (np.sum(np.maximum(0, c["v_lo"] - v)) + np.sum(np.maximum(0, v - c["v_hi"])))
            cost += c["w_deg"] * abs(ps) / env.e_cap * 100
            e = ead - max(ps, 0) / c["eta_dis"] + max(-ps, 0) * c["eta_ch"]
            if e < c["soc_min"] * env.e_cap:
                cost += c["w_unmet"] * (c["soc_min"] * env.e_cap - e) / env.e_cap
                e = c["soc_min"] * env.e_cap
        cost += c["w_term"] * max(0.0, env.sc["soc0"][i] - e / env.e_cap) * 100
        return cost

    def reset(self, env):
        self.plan = np.zeros((2, 24))
        rb = RuleBased(rho=self.rho)
        for i in range(2):
            # warm start from the rule-based schedule (computed open loop)
            x0 = np.zeros(24)
            for k in range(24):
                t = (k + env.c["start_hour"]) % 24
                vmin, vmax = env.v0[i, t]
                x0[k] = np.clip((rb.v_on - vmin) / (rb.v_on - rb.v_full), 0, 1) - \
                    np.clip((vmax - rb.vh_on) / (rb.vh_full - rb.vh_on), 0, 1)
                if x0[k] == 0 and (t < 6 or (i == 1 and 9 <= t < 15)):
                    x0[k] = -0.6
            res = minimize(lambda x: self._area_cost(env, i, x), x0, method="Powell",
                           bounds=[(-1, 1)] * 24, options=dict(maxiter=self.maxiter, xtol=1e-3, ftol=1e-4))
            self.plan[i] = res.x

    def act(self, env):
        r = rho_to_action(self.rho, env.c["rho_max"])
        return np.array([self.plan[0, env.t], self.plan[1, env.t], r, r])
