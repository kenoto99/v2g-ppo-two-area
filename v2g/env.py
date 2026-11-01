"""Two-time-scale coordinated V2G environment.

Slow layer (dt = 1 h, 24 steps): the agent sets, for each area, the V2G
active-power set-point fraction p_i and the share rho_i of the plugged-in
converter rating that is withheld as frequency-regulation reserve.
The set-points are applied to an IEEE 33-bus feeder per area (AC power flow,
backward/forward sweep). The reserve sets the gain and the saturation of a
V2G droop loop inside a two-area LFC model that is simulated for a 60 s
disturbance window at 20 ms resolution in every hour (fast layer).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import lfilter

from .feeder import RadialFeeder
from .lfc import TwoAreaLFC
from .profiles import sample_scenario

ENV_VERSION = "v3"   # v1: no regulation energy; v2: hour-extrapolated; v3: decomposed + exact sustained-activation headroom

DEFAULT = dict(
    # EV fleet per representative feeder
    n_ev=300, charger_kw=7.4, battery_kwh=60.0, eta_ch=0.92, eta_dis=0.92,
    soc_min=0.20, soc_max=0.90, drive_kwh_day=9.0,
    ev_buses=([12, 16, 17, 31, 32], [15, 16, 17, 30, 31, 32]),  # 0-based
    pv_buses=([13, 17, 32], [15, 16, 17, 30, 31, 32]),
    pv_mw=(1.5, 4.2),
    n_feeders=25,                     # identical feeders per area
    p_base_mw=500.0, H0=(5.0, 4.0), inertia_pv_drop=(0.20, 0.35),
    step_pu=0.03, ou_sigma_pu=0.006, window_s=60.0,
    rho_max=0.6,
    v_lo=0.95, v_hi=1.05,
    w_v=50.0, w_ov=1.0, w_f=4.0, w_tie=0.5, w_deg=0.05, w_term=1.0, w_unmet=100.0,
    reward_scale=0.1,
    start_hour=12,        # episode runs 12:00 -> 12:00 (next day)
    # Energy exchanged by the V2G droop loop (battery side, with efficiencies):
    #   "none"     : neglected (v1 model, diagnostic only)
    #   "window"   : the simulated 60 s window only
    #   "hour"     : window energy scaled by 3600/window (step event repeated every window)
    #   "fullhour" : exact 3600 s fast-layer simulation of the same hour (same step event,
    #                OU fluctuation continued with the same random stream)
    #   "decomposed": step-event window counted once plus the fluctuation-only window
    #                scaled to the remaining 59 minutes (cheap approximation of "fullhour")
    reg_energy_mode="decomposed",
    reg_energy_diag=False,   # also compute the exact full-hour energy for reporting in any mode
)


class V2GTwoAreaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, split="train", scenario_list=None, seed=None):
        self.c = dict(DEFAULT, **(cfg or {}))
        c = self.c
        self.feeder = RadialFeeder()
        self.lfc = TwoAreaLFC(p_base_mw=c["p_base_mw"], H=c["H0"], window_s=c["window_s"])
        self.lfc_long = TwoAreaLFC(p_base_mw=c["p_base_mw"], H=c["H0"], window_s=3600.0)
        self.split = split
        self.scenario_list = scenario_list
        self._sc_idx = 0
        self.rng = np.random.default_rng(seed)
        self.e_cap = c["n_ev"] * c["battery_kwh"] / 1e3          # MWh per feeder fleet
        self.p_rate_full = c["n_ev"] * c["charger_kw"] / 1e3     # MW per feeder fleet
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(18,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _injections(self, i, t, p_ev_mw):
        c, fd = self.c, self.feeder
        sc = self.sc
        p = fd.p0 * sc["load"][i, t]
        q = fd.q0 * sc["load"][i, t]
        pvb = c["pv_buses"][i]
        p = p.copy()
        p[pvb] -= c["pv_mw"][i] * sc["pv"][i, t] / len(pvb)
        evb = c["ev_buses"][i]
        p[evb] -= p_ev_mw / len(evb)      # + discharge = injection
        return p, q

    def _base_case(self):
        self.v0 = np.zeros((2, 24, 2))
        for i in range(2):
            for t in range(24):
                vm, _ = self.feeder.solve(*self._injections(i, t, 0.0))
                self.v0[i, t] = vm[1:].min(), vm[1:].max()

    def _drive_energy(self):
        c = self.c
        away = 1.0 - self.sc["avail"]
        w = away / away.sum(axis=1, keepdims=True)
        return w * c["n_ev"] * c["drive_kwh_day"] / 1e3   # MWh per hour, per feeder fleet

    def _disturbance(self, t, lfc=None, with_step=True):
        """Reproducible per (scenario, hour): step in one area + OU fluctuation.
        With the full-hour LFC instance the first 60 s are identical to the window.
        with_step=False returns the fluctuating component alone (same random stream)."""
        c, sc = self.c, self.sc
        lfc = lfc or self.lfc
        r = np.random.default_rng([sc["seed"], t])
        n, dt = lfc.n, lfc.dt
        d = np.zeros((n, 2))
        area = r.integers(2)
        mag = c["step_pu"] * sc["dist_scale"] * r.uniform(0.5, 1.0) * r.choice([-1.0, 1.0])
        if with_step:
            d[int(2.0 / dt):, area] += mag
        tau = 10.0
        for i in range(2):
            sig = c["ou_sigma_pu"] * (0.3 + 2.0 * sc["pv"][i, t] * (1.2 - sc["cloud"][i]))
            w = r.standard_normal(n)
            a = np.exp(-dt / tau); b = sig * np.sqrt(1 - a * a)
            d[:, i] += lfilter([b], [1.0, -a], w)   # Ornstein-Uhlenbeck process
        return d

    def _obs(self):
        t = self.h
        sc, c = self.sc, self.c
        h = 2 * np.pi * (t + 0.5) / 24
        H = np.array(c["H0"]) * (1 - np.array(c["inertia_pv_drop"]) * sc["pv"][:, t])
        o = [np.sin(h), np.cos(h), sc["load"][0, t], sc["load"][1, t], sc["pv"][0, t], sc["pv"][1, t],
             sc["avail"][0, t] * sc["participation"][0], sc["avail"][1, t] * sc["participation"][1],
             (self.soc[0] - 0.55) / 0.2, (self.soc[1] - 0.55) / 0.2,
             (self.v0[0, t, 0] - 1) / 0.05, (self.v0[0, t, 1] - 1) / 0.05,
             (self.v0[1, t, 0] - 1) / 0.05, (self.v0[1, t, 1] - 1) / 0.05,
             H[0] / c["H0"][0], H[1] / c["H0"][1], sc["dist_scale"], (24 - self.t) / 24]
        return np.asarray(o, np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.scenario_list is not None:
            self.sc = self.scenario_list[self._sc_idx % len(self.scenario_list)]
            self._sc_idx += 1
        else:
            self.sc = sample_scenario(self.rng, self.split)
        self.t = 0
        self.h = self.c["start_hour"]
        self.soc = np.array(self.sc["soc0"], float)
        self.e = self.soc * self.e_cap
        self._base_case()
        self.e_drive = self._drive_energy()
        return self._obs(), {}

    def decode(self, a):
        raw = np.asarray(a, float)
        a = np.clip(raw, -1.0, 1.0)  # explicit bound before use
        self._raw_clip = np.abs(raw) > 1.0 + 1e-12   # raw-action clipping, per component
        p = a[:2]
        rho = self.c["rho_max"] * (a[2:] + 1.0) / 2.0
        return p, rho

    def _reserve_headroom(self, i, p_set, p_rate, rho, e_after_drive):
        """Largest symmetric reserve R such that (a) |p_set| + R <= p_rate and (b) a full
        one-directional activation of R sustained for the whole hour keeps the stored energy
        inside [E_min, E_max], with the efficiency of each path accounted for exactly:
        upward activation first reduces charging (saves eta_ch per MWh) and then discharges
        (costs 1/eta_dis per MWh); downward activation first reduces discharging (saves
        1/eta_dis per MWh) and then charges (adds eta_ch per MWh)."""
        c = self.c
        pd, pc = max(p_set, 0.0), max(-p_set, 0.0)
        e_sched = e_after_drive - pd / c["eta_dis"] + pc * c["eta_ch"]      # energy after the schedule
        A = e_sched - c["soc_min"] * self.e_cap                             # room above the floor
        B = c["soc_max"] * self.e_cap - e_sched                             # room below the ceiling
        if A <= 0.0:
            r_up = 0.0
        elif A / c["eta_ch"] <= pc:
            r_up = A / c["eta_ch"]
        else:
            r_up = pc + c["eta_dis"] * (A - c["eta_ch"] * pc)
        if B <= 0.0:
            r_dn = 0.0
        elif c["eta_dis"] * B <= pd:
            r_dn = c["eta_dis"] * B
        else:
            r_dn = pd + (B - pd / c["eta_dis"]) / c["eta_ch"]
        return max(0.0, min(rho * p_rate, p_rate - abs(p_set), r_up, r_dn))

    def _activation_energy(self, p_set, r):
        """Battery energy change (MWh) of a full upward and a full downward activation of the
        reserve r sustained for one hour on top of the set-point p_set (same path logic as
        _reserve_headroom)."""
        c = self.c
        pd, pc = max(p_set, 0.0), max(-p_set, 0.0)
        up = -(c["eta_ch"] * min(r, pc) + max(r - pc, 0.0) / c["eta_dis"])
        dn = min(r, pd) / c["eta_dis"] + max(r - pd, 0.0) * c["eta_ch"]
        return up, dn

    @staticmethod
    def regulation_energy(pev_mw, dt):
        """Grid-side energy of the droop loop in one window (MWh, >= 0 each):
        delivered to the grid (discharge) and absorbed from the grid (charge)."""
        e_out = np.sum(np.maximum(pev_mw, 0.0), axis=0) * dt / 3600.0
        e_in = np.sum(np.maximum(-pev_mw, 0.0), axis=0) * dt / 3600.0
        return e_out, e_in

    def step(self, action):
        c, sc = self.c, self.sc
        t = self.h  # clock hour used to index profiles
        p, rho = self.decode(action)
        info = dict(hour=t, step=self.t)
        p_set = np.zeros(2); res = np.zeros(2); proj = np.zeros(2, bool)
        p_dis_av = np.zeros(2); p_ch_av = np.zeros(2)
        p_rate = self.p_rate_full * sc["avail"][:, t] * sc["participation"]
        for i in range(2):
            e_after_drive = self.e[i] - self.e_drive[i, t]
            p_dis_av[i] = max(0.0, min(p_rate[i], (e_after_drive - c["soc_min"] * self.e_cap) * c["eta_dis"]))
            p_ch_av[i] = max(0.0, min(p_rate[i], (c["soc_max"] * self.e_cap - e_after_drive) / c["eta_ch"]))
            req = p[i] * (1 - rho[i]) * p_rate[i]
            p_set[i] = np.clip(req, -p_ch_av[i], p_dis_av[i])
            proj[i] = abs(req - p_set[i]) > 1e-6          # physical projection active
            res[i] = self._reserve_headroom(i, p_set[i], p_rate[i], rho[i], e_after_drive)
        # ---- slow layer: AC power flow per area
        vio = np.zeros(2); vmin = np.zeros(2); vmax = np.zeros(2); loss = np.zeros(2)
        under = np.zeros(2); over = np.zeros(2)
        for i in range(2):
            vm, ls = self.feeder.solve(*self._injections(i, t, p_set[i]))
            v = vm[1:]
            under[i] = np.sum(np.maximum(0, c["v_lo"] - v)); over[i] = np.sum(np.maximum(0, v - c["v_hi"]))
            vio[i] = under[i] + c["w_ov"] * over[i]
            vmin[i], vmax[i], loss[i] = v.min(), v.max(), ls
            info[f"v{i+1}"] = vm
        # ---- fast layer: two-area LFC with V2G droop
        H = np.array(c["H0"]) * (1 - np.array(c["inertia_pv_drop"]) * sc["pv"][:, t])
        sim = self.lfc.run(H, res * c["n_feeders"], self._disturbance(t))
        fm = TwoAreaLFC.metrics(sim, self.lfc.dt)
        # ---- regulation energy of the droop loop (per feeder fleet)
        nf = c["n_feeders"]
        pev_f = sim["pev_mw"] / nf                       # per-feeder droop power (MW)
        e_out_w, e_in_w = self.regulation_energy(pev_f, self.lfc.dt)
        mode = c["reg_energy_mode"]
        long = None
        if mode == "fullhour" or c["reg_energy_diag"]:
            long = self.lfc_long.run(H, res * nf, self._disturbance(t, self.lfc_long))
            pev_long = long["pev_mw"] / nf
            e_out_h, e_in_h = self.regulation_energy(pev_long, self.lfc_long.dt)
            sat_frac = float(np.mean(np.abs(long["pev_mw"]) >= 0.999 * np.maximum(res * nf, 1e-9)[None, :]
                                     ) if np.any(res > 0) else 0.0)
        # fluctuation-only window (no step event), same random stream
        sim_ou = self.lfc.run(H, res * nf, self._disturbance(t, with_step=False))
        e_out_ou, e_in_ou = self.regulation_energy(sim_ou["pev_mw"] / nf, self.lfc.dt)
        n_rest = 3600.0 / (self.lfc.n * self.lfc.dt) - 1.0
        e_out_d, e_in_d = e_out_w + n_rest * e_out_ou, e_in_w + n_rest * e_in_ou
        if mode == "fullhour":
            e_out, e_in = e_out_h, e_in_h
        elif mode == "decomposed":
            e_out, e_in = e_out_d, e_in_d
        else:
            scale = {"hour": 3600.0 / (self.lfc.n * self.lfc.dt), "window": 1.0, "none": 0.0}[mode]
            e_out, e_in = e_out_w * scale, e_in_w * scale
        de_reg = -e_out / c["eta_dis"] + c["eta_ch"] * e_in   # battery-side, MWh
        # design guarantee for sustained one-directional activation of the full reserve for 1 h
        e_sched = self.e - self.e_drive[:, t] + np.array([(-max(x, 0) / c["eta_dis"] + max(-x, 0) * c["eta_ch"]) for x in p_set])
        # a shortfall caused by driving alone is unmet mobility energy, not a reserve violation,
        # so the margin is measured from the floor in that case (res is then zero)
        d_up = np.array([self._activation_energy(p_set[i], res[i])[0] for i in range(2)])   # <= 0
        d_dn = np.array([self._activation_energy(p_set[i], res[i])[1] for i in range(2)])   # >= 0
        sust_lo = (np.maximum(e_sched, c["soc_min"] * self.e_cap) + d_up) - c["soc_min"] * self.e_cap  # >= 0 required
        sust_hi = c["soc_max"] * self.e_cap - (e_sched + d_dn)                                        # >= 0 required
        # ---- feasibility audit of scheduled power + droop response (per feeder)
        p_tot_max = p_set + pev_f.max(axis=0); p_tot_min = p_set + pev_f.min(axis=0)
        pw_excess = np.maximum(np.maximum(p_tot_max - p_dis_av, -p_ch_av - p_tot_min), 0.0)
        res_excess = np.maximum(np.abs(pev_f).max(axis=0) - res, 0.0)
        # ---- energy update (per feeder fleet)
        unmet = np.zeros(2); e_viol = np.zeros(2)
        e_before = self.e.copy()
        for i in range(2):
            pd, pc = max(p_set[i], 0.0), max(-p_set[i], 0.0)
            self.e[i] += -pd / c["eta_dis"] + pc * c["eta_ch"] - self.e_drive[i, t] + de_reg[i]
            if self.e[i] < c["soc_min"] * self.e_cap - 1e-9:
                short = c["soc_min"] * self.e_cap - self.e[i]
                # driving shortfall is unmet mobility energy; anything caused by
                # regulation would be an infeasibility of the reserve design
                drive_short = max(0.0, c["soc_min"] * self.e_cap - (e_before[i] - pd / c["eta_dis"]
                                  + pc * c["eta_ch"] - self.e_drive[i, t]))
                unmet[i] = min(short, drive_short)
                e_viol[i] = short - unmet[i] if short - unmet[i] > 1e-9 else 0.0
                self.e[i] = c["soc_min"] * self.e_cap
            elif self.e[i] > c["soc_max"] * self.e_cap + 1e-9:
                e_viol[i] = self.e[i] - c["soc_max"] * self.e_cap
                self.e[i] = c["soc_max"] * self.e_cap
        self.soc = self.e / self.e_cap
        thr = np.abs(p_set) / self.e_cap * 100.0
        cost_v = c["w_v"] * vio.sum()
        cost_f = c["w_f"] * np.sum(fm["nadir_hz"] / 0.1) + c["w_tie"] * fm["tie_rms_mw"] / 10.0
        cost_d = c["w_deg"] * thr.sum()
        cost_u = c["w_unmet"] * unmet.sum() / self.e_cap
        self.t += 1
        done = self.t >= 24
        self.h = (self.t + c["start_hour"]) % 24
        cost_t = 0.0
        if done:
            cost_t = c["w_term"] * np.sum(np.maximum(0.0, np.array(sc["soc0"]) - self.soc)) * 100
        cost = cost_v + cost_f + cost_d + cost_u + cost_t
        info.update(p_set_mw=p_set, rho=rho, reserve_mw_area=res * c["n_feeders"],
                    clip=proj, raw_clip=self._raw_clip.copy(), proj=proj,
                    reg_out_mwh_w=e_out_w * nf, reg_in_mwh_w=e_in_w * nf,          # area level, 60 s window
                    reg_batt_mwh=de_reg * nf, reg_batt_mwh_w=(-e_out_w / c["eta_dis"] + c["eta_ch"] * e_in_w) * nf,
                    sched_batt_mwh=np.array([(-max(x, 0) / c["eta_dis"] + max(-x, 0) * c["eta_ch"]) for x in p_set]) * nf,
                    pw_excess_mw=pw_excess, res_excess_mw=res_excess, e_viol_mwh=e_viol,
                    sust_margin_lo_mwh=sust_lo, sust_margin_hi_mwh=sust_hi,
                    reg_batt_mwh_hour=(-e_out_w * 3600.0 / (self.lfc.n * self.lfc.dt) / c["eta_dis"]
                                       + c["eta_ch"] * e_in_w * 3600.0 / (self.lfc.n * self.lfc.dt)) * nf,
                    reg_batt_mwh_full=((-e_out_h / c["eta_dis"] + c["eta_ch"] * e_in_h) * nf) if long is not None else None,
                    reg_gross_mwh_full=((e_out_h + e_in_h) * nf) if long is not None else None,
                    reg_gross_mwh_w=(e_out_w + e_in_w) * nf,
                    reg_batt_mwh_dec=(-e_out_d / c["eta_dis"] + c["eta_ch"] * e_in_d) * nf,
                    reg_gross_mwh_dec=(e_out_d + e_in_d) * nf,
                    sat_frac_full=sat_frac if long is not None else None,
                    pev_max_full=(np.abs(long["pev_mw"]).max(axis=0)) if long is not None else None,
                    e_before_mwh=e_before, e_after_mwh=self.e.copy(), e_drive_mwh=self.e_drive[:, t].copy(),
                    vmin=vmin, vmax=vmax, under=under, over=over, loss_mw=loss, soc=self.soc.copy(),
                    nadir_hz=fm["nadir_hz"], rms_hz=fm["rms_hz"], rocof=fm["rocof_hz_s"],
                    tie_rms_mw=fm["tie_rms_mw"], tie_max_mw=fm["tie_max_mw"], ace_rms=fm["ace_rms"],
                    unmet_mwh=unmet, p_rate_mw=p_rate,
                    cost=dict(v=cost_v, f=cost_f, d=cost_d, u=cost_u, t=cost_t))
        if getattr(self, "keep_traces", False):
            info["trace"] = sim
        obs = self._obs()
        return obs, -c["reward_scale"] * cost, done, False, info
