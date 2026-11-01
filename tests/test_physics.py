"""Physics and bookkeeping tests. Run: python -m pytest -q tests/ (or python tests/test_physics.py).
Each test returns a dict that scripts/verify_all.py writes into the verification report."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2g.feeder import RadialFeeder
from v2g.lfc import TwoAreaLFC, F0
from v2g.env import V2GTwoAreaEnv
from v2g.profiles import sample_scenario


def test_bfs_vs_pandapower(n=50, tol_v=1e-8, tol_loss_kw=1e-5):
    import pandapower as pp, pandapower.networks as pn
    fd = RadialFeeder(); rng = np.random.default_rng(0)
    ev, el = [], []
    for _ in range(n):
        lf = rng.uniform(0.3, 1.2)
        inj = np.zeros(fd.nb); buses = rng.choice(np.arange(1, fd.nb), 6, replace=False)
        inj[buses] = rng.uniform(-1.0, 1.0, 6)
        vm, loss = fd.solve(fd.p0 * lf - inj, fd.q0 * lf)
        net = pn.case33bw()
        net.load["p_mw"] *= lf; net.load["q_mvar"] *= lf
        for b in buses:
            pp.create_sgen(net, int(b), p_mw=float(inj[b]), q_mvar=0.0)
        pp.runpp(net, tolerance_mva=1e-10)
        ev.append(np.max(np.abs(vm - net.res_bus.vm_pu.to_numpy())))
        el.append(abs(loss - net.res_line.pl_mw.sum()) * 1e3)
    vm, loss = fd.solve(fd.p0, fd.q0)
    out = dict(cases=n, max_v_err_pu=float(max(ev)), max_loss_err_kw=float(max(el)),
               base_vmin_pu=float(vm.min()), base_vmin_bus=int(np.argmin(vm)) + 1, base_loss_kw=loss * 1e3)
    assert out["max_v_err_pu"] < tol_v and out["max_loss_err_kw"] < tol_loss_kw
    assert abs(out["base_vmin_pu"] - 0.9131) < 5e-4 and out["base_vmin_bus"] == 18 and abs(out["base_loss_kw"] - 202.68) < 0.05
    return out


def test_lfc_steady_state():
    l = TwoAreaLFC(window_s=300.0, KI=(0.0, 0.0))
    d = np.zeros((l.n, 2)); d[int(2 / l.dt):, 0] = 0.02
    s = l.run(l.H0, [0, 0], d)
    B = 1 / l.R + l.D
    th_f, th_t = -0.02 / B.sum() * F0, -0.02 * B[1] / B.sum() * l.pb
    out = dict(df_sim_hz=float(s["df_hz"][-1, 0]), df_theory_hz=float(th_f),
               tie_sim_mw=float(s["ptie_mw"][-1]), tie_theory_mw=float(th_t))
    assert abs(out["df_sim_hz"] - th_f) < 1e-6 and abs(out["tie_sim_mw"] - th_t) < 1e-3
    l2 = TwoAreaLFC(window_s=300.0)
    s2 = l2.run(l2.H0, [0, 0], d)
    out.update(agc_df_end_hz=float(s2["df_hz"][-1, 0]), agc_tie_end_mw=float(s2["ptie_mw"][-1]))
    assert abs(out["agc_df_end_hz"]) < 1e-5 and abs(out["agc_tie_end_mw"]) < 1e-2
    return out


def test_regulation_energy_integral():
    """A known droop trace: +2 MW for 30 s then -1 MW for 30 s, dt = 0.02 s."""
    dt = 0.02; n = 3000
    p = np.zeros((n, 2)); p[:1500, 0] = 2.0; p[1500:, 0] = -1.0
    e_out, e_in = V2GTwoAreaEnv.regulation_energy(p, dt)
    exp_out, exp_in = 2.0 * 30 / 3600, 1.0 * 30 / 3600
    eta = 0.92
    batt = -e_out[0] / eta + eta * e_in[0]
    exp_batt = -exp_out / eta + eta * exp_in
    out = dict(e_out_mwh=float(e_out[0]), e_in_mwh=float(e_in[0]), batt_mwh=float(batt), expected_batt_mwh=exp_batt)
    assert abs(e_out[0] - exp_out) < 1e-12 and abs(e_in[0] - exp_in) < 1e-12 and abs(batt - exp_batt) < 1e-12
    assert batt < -exp_out + exp_in  # round-trip losses always reduce stored energy
    return out


def test_feasibility_random_policy(days=40, seed=5, cfg=None):
    """Random (including out-of-range) actions on train/stress days: the scheduled power plus the
    droop response must stay within the charger and energy limits, SoC within bounds,
    and the energy balance must close."""
    rng = np.random.default_rng(seed)
    scs = [sample_scenario(rng, "train") for _ in range(days // 2)] + [sample_scenario(rng, "stress") for _ in range(days - days // 2)]
    env = V2GTwoAreaEnv(cfg=cfg, scenario_list=scs)
    pw = rs = ev = clo = 0.0; socmin, socmax = 1, 0; steps = 0; raw = proj = 0; min_margin = 1e9
    for _ in range(days):
        env.reset(); done = False
        while not done:
            a = rng.normal(0, 1.2, 4)
            _, _, done, _, info = env.step(a)
            steps += 1
            pw = max(pw, info["pw_excess_mw"].max()); rs = max(rs, info["res_excess_mw"].max())
            ev = max(ev, info["e_viol_mwh"].max())
            min_margin = min(min_margin, info["sust_margin_lo_mwh"].min(), info["sust_margin_hi_mwh"].min())
            socmin = min(socmin, info["soc"].min()); socmax = max(socmax, info["soc"].max())
            raw += int(info["raw_clip"].sum()); proj += int(info["proj"].sum())
            c = env.c
            for i in range(2):
                ps = info["p_set_mw"][i]
                pred = (info["e_before_mwh"][i] - max(ps, 0) / c["eta_dis"] + max(-ps, 0) * c["eta_ch"]
                        - info["e_drive_mwh"][i] + info["reg_batt_mwh"][i] / c["n_feeders"] + info["unmet_mwh"][i])
                clo = max(clo, abs(info["e_after_mwh"][i] - pred) if info["e_viol_mwh"][i] == 0 else 0.0)
    out = dict(steps=steps, max_power_excess_mw=float(pw), max_reserve_excess_mw=float(rs),
               min_sustained_margin_mwh=float(min_margin),
               max_energy_violation_mwh=float(ev), soc_min=float(socmin), soc_max=float(socmax),
               max_energy_closure_err_mwh=float(clo), raw_clip_components=raw, projection_events=proj)
    assert pw < 1e-9 and rs < 1e-6 and ev < 1e-9 and clo < 1e-9 and min_margin > -1e-9
    assert socmin >= 0.2 - 1e-9 and socmax <= 0.9 + 1e-9
    assert raw > 0  # random actions with std 1.2 must trigger raw clipping
    return out


def test_feasibility_fullhour(days=6, seed=7):
    """Same as above with the exact full-hour regulation energy (slow)."""
    return test_feasibility_random_policy(days=days, seed=seed, cfg=dict(reg_energy_mode="fullhour"))


def test_no_reserve_from_mobility_shortfall(n=600, seed=11):
    """States near or below the SoC floor after driving, random actions: regulation must never
    take the fleet below E_min beyond the shortfall caused by driving (e_viol == 0)."""
    rng = np.random.default_rng(seed)
    scs = [sample_scenario(rng, "test") for _ in range(6)]
    worst = 0.0; credited = 0
    for k in range(n):
        env = V2GTwoAreaEnv(scenario_list=[scs[k % 6]]); env.reset()
        for _ in range(int(rng.integers(0, 23))):   # advance to a random hour
            env.step(np.array([0.0, 0.0, -1.0, -1.0]))
        t = env.h
        env.e[:] = 0.2 * env.e_cap + env.e_drive[:, t] * rng.uniform(-0.5, 1.5, 2)
        _, _, _, _, info = env.step(rng.uniform(-1, 1, 4) * np.array([1, 1, 1, 1]))
        worst = max(worst, info["e_viol_mwh"].max())
        credited += int((info["reserve_mw_area"] > 0).sum())
    out = dict(cases=n, max_energy_violation_mwh=float(worst), cases_with_reserve=credited)
    assert worst == 0.0
    return out


def test_clip_counters_distinct():
    rng = np.random.default_rng(1)
    sc = sample_scenario(rng, "test")
    env = V2GTwoAreaEnv(scenario_list=[sc]); env.reset()
    # in-range action that requests full charging while SoC is high -> projection, no raw clip
    env.e[:] = 0.895 * env.e_cap
    _, _, _, _, i1 = env.step(np.array([-1.0, -1.0, -1.0, -1.0]))
    env2 = V2GTwoAreaEnv(scenario_list=[sc]); env2.reset()
    _, _, _, _, i2 = env2.step(np.array([1.5, 0.0, -2.0, 0.3]))
    out = dict(case1_raw=i1["raw_clip"].tolist(), case1_proj=i1["proj"].tolist(),
               case2_raw=i2["raw_clip"].tolist())
    assert not i1["raw_clip"].any() and i1["proj"].any()
    assert i2["raw_clip"].tolist() == [True, False, True, False]
    return out


if __name__ == "__main__":
    for f in [test_bfs_vs_pandapower, test_lfc_steady_state, test_regulation_energy_integral,
              test_feasibility_random_policy, test_feasibility_fullhour, test_no_reserve_from_mobility_shortfall, test_clip_counters_distinct]:
        print(f.__name__, f())
