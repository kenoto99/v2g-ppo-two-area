"""Validate the BFS feeder solver against pandapower and the LFC model
against closed-form steady-state values. Writes results/validation.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v2g.feeder import RadialFeeder
from v2g.lfc import TwoAreaLFC, F0
import pandapower as pp
import pandapower.networks as pn

out = {}
fd = RadialFeeder()
rng = np.random.default_rng(0)
errs, lerrs = [], []
cases = []
for k in range(20):
    lf = rng.uniform(0.4, 1.1)
    inj = np.zeros(fd.nb)
    buses = rng.choice(np.arange(1, fd.nb), 5, replace=False)
    inj[buses] = rng.uniform(-0.8, 0.8, 5)  # MW, + = generation
    p = fd.p0 * lf - inj
    q = fd.q0 * lf
    vm, loss = fd.solve(p, q)
    net = pn.case33bw()
    net.load["p_mw"] = net.load.p_mw * lf
    net.load["q_mvar"] = net.load.q_mvar * lf
    for b in buses:
        pp.create_sgen(net, int(b), p_mw=float(inj[b]), q_mvar=0.0)
    pp.runpp(net, tolerance_mva=1e-10)
    errs.append(np.max(np.abs(vm - net.res_bus.vm_pu.to_numpy())))
    lerrs.append(abs(loss - net.res_line.pl_mw.sum()))
vm, loss = fd.solve(fd.p0, fd.q0)
out["bfs_base_vmin_pu"] = float(vm.min()); out["bfs_base_vmin_bus"] = int(np.argmin(vm)) + 1
out["bfs_base_loss_kw"] = loss * 1e3
out["bfs_vs_pandapower_max_abs_v_err_pu"] = float(max(errs))
out["bfs_vs_pandapower_max_abs_loss_err_kw"] = float(max(lerrs) * 1e3)

lfc = TwoAreaLFC(window_s=300.0)
dPL = np.zeros((lfc.n, 2)); dPL[int(2 / lfc.dt):, 0] = 0.02
lfc_noagc = TwoAreaLFC(window_s=300.0, KI=(0.0, 0.0))
s = lfc_noagc.run(lfc.H0, [0, 0], dPL)
B = 1 / lfc.R + lfc.D
out["lfc_noagc_df_ss_hz_sim"] = float(s["df_hz"][-1, 0])
out["lfc_noagc_df_ss_hz_theory"] = float(-0.02 / B.sum() * F0)
out["lfc_noagc_tie_ss_mw_sim"] = float(s["ptie_mw"][-1])
out["lfc_noagc_tie_ss_mw_theory"] = float(-0.02 * B[1] / B.sum() * lfc.pb)
s2 = lfc.run(lfc.H0, [0, 0], dPL)
out["lfc_agc_df_end_hz"] = float(s2["df_hz"][-1, 0]); out["lfc_agc_tie_end_mw"] = float(s2["ptie_mw"][-1])
s3 = lfc.run(lfc.H0, [25, 25], dPL)
m2, m3 = TwoAreaLFC.metrics(s2, lfc.dt), TwoAreaLFC.metrics(s3, lfc.dt)
out["lfc_nadir_area1_hz_noev"] = float(m2["nadir_hz"][0]); out["lfc_nadir_area1_hz_ev25MW"] = float(m3["nadir_hz"][0])
out["lfc_rocof_area1_noev"] = float(m2["rocof_hz_s"][0]); out["lfc_rocof_area1_ev25MW"] = float(m3["rocof_hz_s"][0])
# inter-area oscillation frequency from linearised eigenvalues (analytic)
w2 = 2 * np.pi * F0 * lfc.T12 * (1 / (2 * lfc.H0[0]) + 1 / (2 * lfc.H0[1]))
out["inter_area_mode_hz_approx"] = float(np.sqrt(w2) / (2 * np.pi))
os.makedirs("results", exist_ok=True)
json.dump(out, open("results/validation.json", "w"), indent=2)
print(json.dumps(out, indent=2))
