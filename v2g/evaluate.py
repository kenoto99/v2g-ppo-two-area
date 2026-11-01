"""Roll out a controller over a list of scenarios and collect daily metrics."""
import numpy as np


def rollout(env, controller=None, model=None, keep=False):
    obs, _ = env.reset()
    if controller is not None:
        controller.reset(env)
    rows = []
    done = False
    while not done:
        if model is not None:
            # raw (unclipped) deterministic action; the environment clips it and
            # records whether clipping was active. For a squashed policy the
            # deterministic action is tanh(mean) and lies inside the box by design.
            import torch
            with torch.no_grad():
                ot, _ = model.policy.obs_to_tensor(obs)
                dist = model.policy.get_distribution(ot)
                if getattr(model.policy, "squash_output", False):
                    a = dist.mode().cpu().numpy()[0]
                else:
                    a = dist.distribution.mean.cpu().numpy()[0]
        else:
            a = controller.act(env)
        obs, r, done, _, info = env.step(a)
        info["reward"] = r
        if not keep:
            info.pop("v1", None); info.pop("v2", None); info.pop("trace", None)
        rows.append(info)
    return rows


def daily_metrics(rows, env):
    c = env.c
    under = np.array([r["under"] for r in rows]); over = np.array([r["over"] for r in rows])
    vmin = np.array([r["vmin"] for r in rows]); vmax = np.array([r["vmax"] for r in rows])
    nadir = np.array([r["nadir_hz"] for r in rows]); rms = np.array([r["rms_hz"] for r in rows])
    rocof = np.array([r["rocof"] for r in rows])
    p = np.array([r["p_set_mw"] for r in rows]); loss = np.array([r["loss_mw"] for r in rows])
    clip = np.array([r["clip"] for r in rows]); res = np.array([r["reserve_mw_area"] for r in rows])
    viol_hours = np.sum((vmin < c["v_lo"] - 1e-9) | (vmax > c["v_hi"] + 1e-9), axis=0)
    return dict(
        reward=float(sum(r["reward"] for r in rows)),
        cost_v=float(sum(r["cost"]["v"] for r in rows)),
        cost_f=float(sum(r["cost"]["f"] for r in rows)),
        viol_hours=float(viol_hours.sum()),
        uv_pu_h=float(under.sum()), ov_pu_h=float(over.sum()),
        vmin_day=float(vmin.min()), vmax_day=float(vmax.max()),
        nadir_mean_hz=float(nadir.max(axis=1).mean()), nadir_max_hz=float(nadir.max()),
        f_rms_mhz=float(np.sqrt((rms ** 2).mean()) * 1e3),
        rocof_max=float(rocof.max()),
        tie_rms_mw=float(np.mean([r["tie_rms_mw"] for r in rows])),
        reserve_mean_mw=float(res.sum(axis=1).mean()),
        soc_end_1=float(rows[-1]["soc"][0]), soc_end_2=float(rows[-1]["soc"][1]),
        soc_def_pct=float(np.sum(np.maximum(0, np.array(env.sc["soc0"]) - rows[-1]["soc"])) * 100),
        throughput_mwh=float(np.abs(p).sum() * c["n_feeders"]),     # area level
        loss_mwh=float(loss.sum()),                                 # per pair of feeders
        clip_events=int(clip.sum()),
        proj_events=int(np.array([r["proj"] for r in rows]).sum()),
        raw_clip_events=int(np.array([r["raw_clip"] for r in rows]).sum()),       # action components
        raw_clip_steps=int(np.array([r["raw_clip"].any() for r in rows]).sum()),
        raw_clip_by_dim=np.array([r["raw_clip"] for r in rows]).sum(axis=0).tolist(),
        reg_dec_mwh=float(np.sum([r["reg_batt_mwh_dec"].sum() for r in rows])),
        reg_gross_dec_mwh=float(np.sum([r["reg_gross_mwh_dec"].sum() for r in rows])),
        reg_out_mwh_w=float(np.sum([r["reg_out_mwh_w"].sum() for r in rows])),
        reg_in_mwh_w=float(np.sum([r["reg_in_mwh_w"].sum() for r in rows])),
        reg_batt_mwh=float(np.sum([r["reg_batt_mwh"].sum() for r in rows])),
        reg_batt_abs_max_mwh=float(np.max([np.abs(r["reg_batt_mwh"]).max() for r in rows])),
        reg_batt_mwh_w=float(np.sum([r["reg_batt_mwh_w"].sum() for r in rows])),
        sched_batt_abs_mwh=float(np.sum([np.abs(r["sched_batt_mwh"]).sum() for r in rows])),
        pw_excess_max_mw=float(np.max([r["pw_excess_mw"].max() for r in rows])),
        res_excess_max_mw=float(np.max([r["res_excess_mw"].max() for r in rows])),
        e_viol_mwh=float(np.sum([r["e_viol_mwh"].sum() for r in rows]) * c["n_feeders"]),
        energy_closure_err_mwh=float(max(abs(r["e_after_mwh"][i] - (r["e_before_mwh"][i]
            - max(r["p_set_mw"][i], 0) / c["eta_dis"] + max(-r["p_set_mw"][i], 0) * c["eta_ch"]
            - r["e_drive_mwh"][i] + r["reg_batt_mwh"][i] / c["n_feeders"]
            + r["unmet_mwh"][i] + r["e_viol_mwh"][i] * (1 if r["e_after_mwh"][i] <= c["soc_min"] * env.e_cap + 1e-9 else -1)))
            for r in rows for i in range(2))),
        unmet_mwh=float(sum(r["unmet_mwh"].sum() for r in rows) * c["n_feeders"]),
        # regulation-energy treatments (area level, both areas, daily sums)
        reg_hour_extrap_mwh=float(np.sum([r["reg_batt_mwh_hour"].sum() for r in rows])),
        reg_full_mwh=float(np.sum([r["reg_batt_mwh_full"].sum() for r in rows])) if rows[0].get("reg_batt_mwh_full") is not None else None,
        reg_gross_full_mwh=float(np.sum([r["reg_gross_mwh_full"].sum() for r in rows])) if rows[0].get("reg_gross_mwh_full") is not None else None,
        reg_gross_window_mwh=float(np.sum([r["reg_gross_mwh_w"].sum() for r in rows])),
        reg_gross_hour_extrap_mwh=float(np.sum([r["reg_gross_mwh_w"].sum() for r in rows]) * 60.0),
        sat_frac_full=float(np.mean([r["sat_frac_full"] for r in rows])) if rows[0].get("sat_frac_full") is not None else None,
        sust_margin_min_mwh=float(min(min(r["sust_margin_lo_mwh"].min(), r["sust_margin_hi_mwh"].min()) for r in rows)),
        sust_margin_bind_hours=int(sum((r["sust_margin_lo_mwh"] < 1e-9).sum() + (r["sust_margin_hi_mwh"] < 1e-9).sum()
                                       for r in rows if (r["reserve_mw_area"] > 0).any())),
        hourly=dict(hour=[int(r["hour"]) for r in rows],
                    rho=np.array([r["rho"] for r in rows]).tolist(),
                    p_area_mw=(p * c["n_feeders"]).tolist(),
                    reserve_mw=res.tolist(), vmin=vmin.tolist(), vmax=vmax.tolist(),
                    nadir_hz=nadir.tolist(), soc=np.array([r["soc"] for r in rows]).tolist(),
                    reg_batt_mwh=np.array([r["reg_batt_mwh"] for r in rows]).tolist(),
                    reg_out_mwh_w=np.array([r["reg_out_mwh_w"] for r in rows]).tolist(),
                    reg_in_mwh_w=np.array([r["reg_in_mwh_w"] for r in rows]).tolist(),
                    reg_batt_mwh_full=(np.array([r["reg_batt_mwh_full"] for r in rows]).tolist() if rows[0].get("reg_batt_mwh_full") is not None else None),
                    reg_batt_mwh_hour=np.array([r["reg_batt_mwh_hour"] for r in rows]).tolist()),
    )
