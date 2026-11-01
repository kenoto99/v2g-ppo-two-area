"""Radial distribution feeder model: IEEE 33-bus (Baran and Wu, 1989).

Branch data are taken from pandapower.networks.case33bw() so that the
network is identical to the published benchmark. The power flow is solved
with a backward/forward sweep in matrix form (BIBC/BCBV), which is validated
against pandapower's Newton-Raphson solver in scripts/validate_models.py.
"""
import numpy as np

V_BASE_KV = 12.66
S_BASE_MVA = 1.0
Z_BASE = V_BASE_KV ** 2 / S_BASE_MVA


def load_case33bw():
    import pandapower.networks as pn
    net = pn.case33bw()
    lines = net.line[net.line.in_service]
    fb = lines.from_bus.to_numpy(int)
    tb = lines.to_bus.to_numpy(int)
    r = (lines.r_ohm_per_km * lines.length_km).to_numpy() / Z_BASE
    x = (lines.x_ohm_per_km * lines.length_km).to_numpy() / Z_BASE
    nb = len(net.bus)
    p = np.zeros(nb)
    q = np.zeros(nb)
    p[net.load.bus.to_numpy(int)] = net.load.p_mw.to_numpy()
    q[net.load.bus.to_numpy(int)] = net.load.q_mvar.to_numpy()
    return dict(fb=fb, tb=tb, r=r, x=x, p_mw=p, q_mvar=q, nb=nb)


class RadialFeeder:
    """Backward/forward sweep solver. Bus 0 is the substation (slack)."""

    def __init__(self, data=None, v0=1.0):
        d = data or load_case33bw()
        self.nb = d["nb"]
        self.fb, self.tb = d["fb"], d["tb"]
        self.z = d["r"] + 1j * d["x"]
        self.r = d["r"]
        self.p0 = d["p_mw"] / S_BASE_MVA
        self.q0 = d["q_mvar"] / S_BASE_MVA
        self.v0 = v0
        nbr = len(self.fb)
        # parent map; branch k feeds bus tb[k]
        parent = -np.ones(self.nb, int)
        br_of_bus = -np.ones(self.nb, int)
        for k in range(nbr):
            parent[self.tb[k]] = self.fb[k]
            br_of_bus[self.tb[k]] = k
        # BIBC: branch current = sum of injections downstream
        bibc = np.zeros((nbr, self.nb - 1))
        for b in range(1, self.nb):
            n = b
            while n != 0:
                bibc[br_of_bus[n], b - 1] = 1.0
                n = parent[n]
        self.bibc = bibc
        self.dlf = (bibc.T * self.z) @ bibc  # BCBV @ BIBC

    def solve(self, p_mw, q_mvar, tol=1e-9, itmax=50):
        """p_mw, q_mvar: net load at each bus (positive = consumption)."""
        s = (p_mw + 1j * q_mvar)[1:] / S_BASE_MVA
        v = np.full(self.nb - 1, self.v0 + 0j)
        for _ in range(itmax):
            i_inj = np.conj(s / v)
            v_new = self.v0 - self.dlf @ i_inj
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new
        i_inj = np.conj(s / v)
        i_br = self.bibc @ i_inj
        loss_mw = float(np.sum(np.abs(i_br) ** 2 * self.r)) * S_BASE_MVA
        vm = np.concatenate([[self.v0], np.abs(v)])
        return vm, loss_mw

    def branch_currents(self, p_mw, q_mvar, tol=1e-9, itmax=50):
        """Same solution as solve(); returns (vm, loss_mw, |i_br| in p.u. of S_BASE/V_BASE).
        Diagnostic only; solve() and all results built from it are unchanged."""
        s = (p_mw + 1j * q_mvar)[1:] / S_BASE_MVA
        v = np.full(self.nb - 1, self.v0 + 0j)
        for _ in range(itmax):
            i_inj = np.conj(s / v)
            v_new = self.v0 - self.dlf @ i_inj
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new
        i_inj = np.conj(s / v)
        i_br = self.bibc @ i_inj
        loss_mw = float(np.sum(np.abs(i_br) ** 2 * self.r)) * S_BASE_MVA
        vm = np.concatenate([[self.v0], np.abs(v)])
        return vm, loss_mw, np.abs(i_br)
