"""Two-area load-frequency control (LFC) model with aggregated V2G support.

Per area i (per unit on the area base P_r, frequency in p.u. of f0):
  2H_i dDf_i/dt   = DPm_i + DPev_i - DPL_i - a_i DPtie - D_i Df_i
  Tg_i dDPg_i/dt  = DPc_i - Df_i/R_i - DPg_i            (governor)
  Tt_i dDPm_i/dt  = DPg_i - DPm_i                        (non-reheat turbine)
  dDPtie/dt       = 2 pi f0 T12 (Df_1 - Df_2)            (tie line, 1 -> 2)
  ACE_i           = B_i Df_i + a_i DPtie,  B_i = 1/R_i + D_i
  DPc_i           = -K_I int ACE_i dt                    (AGC)
  Tev dDPev_i/dt  = sat_{+-Res_i}(-K_ev,i Df_i) - DPev_i (V2G droop, saturated)
with a_1 = +1 and a_2 = -1. Integrated with fourth-order Runge-Kutta.
"""
import numpy as np
from numba import njit

F0 = 50.0


@njit(cache=True)
def _rhs(f1, f2, pg1, pg2, pm1, pm2, pev1, pev2, z1, z2, pt, d1, d2,
         H1, H2, D1, D2, R1, R2, Tg1, Tg2, Tt1, Tt2, KI1, KI2, K1, K2, S1, S2, B1, B2, Tev, c12):
    t1 = -K1 * f1
    if t1 > S1:
        t1 = S1
    elif t1 < -S1:
        t1 = -S1
    t2 = -K2 * f2
    if t2 > S2:
        t2 = S2
    elif t2 < -S2:
        t2 = -S2
    return ((pm1 + pev1 - d1 - pt - D1 * f1) / H1,
            (pm2 + pev2 - d2 + pt - D2 * f2) / H2,
            (-KI1 * z1 - f1 * R1 - pg1) / Tg1,
            (-KI2 * z2 - f2 * R2 - pg2) / Tg2,
            (pg1 - pm1) / Tt1,
            (pg2 - pm2) / Tt2,
            (t1 - pev1) / Tev,
            (t2 - pev2) / Tev,
            B1 * f1 + pt,
            B2 * f2 - pt,
            c12 * (f1 - f2))


@njit(cache=True, fastmath=False)
def _simulate(dt, n, H, D, R, Tg, Tt, KI, T12, Tev, Kev, Res, dPL):
    """Classical fourth-order Runge-Kutta on the 11-state model with scalar state
    variables (f1, f2, pg1, pg2, pm1, pm2, pev1, pev2, z1, z2, ptie); the disturbance is
    held constant within each step."""
    c12 = 2.0 * np.pi * F0 * T12
    H1, H2 = 2.0 * H[0], 2.0 * H[1]
    D1, D2 = D[0], D[1]; R1, R2 = 1.0 / R[0], 1.0 / R[1]
    Tg1, Tg2 = Tg[0], Tg[1]; Tt1, Tt2 = Tt[0], Tt[1]; KI1, KI2 = KI[0], KI[1]
    K1, K2 = Kev[0], Kev[1]; S1, S2 = Res[0], Res[1]
    B1, B2 = R1 + D1, R2 + D2
    f1 = f2 = pg1 = pg2 = pm1 = pm2 = pev1 = pev2 = z1 = z2 = pt = 0.0
    out_f = np.zeros((n, 2)); out_tie = np.zeros(n); out_ev = np.zeros((n, 2)); out_ace = np.zeros((n, 2))
    for k in range(n):
        d1 = dPL[k, 0]; d2 = dPL[k, 1]
        # k1
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _rhs(f1, f2, pg1, pg2, pm1, pm2, pev1, pev2, z1, z2, pt, d1, d2,
            H1, H2, D1, D2, R1, R2, Tg1, Tg2, Tt1, Tt2, KI1, KI2, K1, K2, S1, S2, B1, B2, Tev, c12)
        h = 0.5 * dt
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11 = _rhs(f1 + h * a1, f2 + h * a2, pg1 + h * a3, pg2 + h * a4, pm1 + h * a5, pm2 + h * a6,
            pev1 + h * a7, pev2 + h * a8, z1 + h * a9, z2 + h * a10, pt + h * a11, d1, d2,
            H1, H2, D1, D2, R1, R2, Tg1, Tg2, Tt1, Tt2, KI1, KI2, K1, K2, S1, S2, B1, B2, Tev, c12)
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = _rhs(f1 + h * b1, f2 + h * b2, pg1 + h * b3, pg2 + h * b4, pm1 + h * b5, pm2 + h * b6,
            pev1 + h * b7, pev2 + h * b8, z1 + h * b9, z2 + h * b10, pt + h * b11, d1, d2,
            H1, H2, D1, D2, R1, R2, Tg1, Tg2, Tt1, Tt2, KI1, KI2, K1, K2, S1, S2, B1, B2, Tev, c12)
        e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11 = _rhs(f1 + dt * c1, f2 + dt * c2, pg1 + dt * c3, pg2 + dt * c4, pm1 + dt * c5, pm2 + dt * c6,
            pev1 + dt * c7, pev2 + dt * c8, z1 + dt * c9, z2 + dt * c10, pt + dt * c11, d1, d2,
            H1, H2, D1, D2, R1, R2, Tg1, Tg2, Tt1, Tt2, KI1, KI2, K1, K2, S1, S2, B1, B2, Tev, c12)
        w = dt / 6.0
        f1 += w * (a1 + 2.0 * b1 + 2.0 * c1 + e1); f2 += w * (a2 + 2.0 * b2 + 2.0 * c2 + e2)
        pg1 += w * (a3 + 2.0 * b3 + 2.0 * c3 + e3); pg2 += w * (a4 + 2.0 * b4 + 2.0 * c4 + e4)
        pm1 += w * (a5 + 2.0 * b5 + 2.0 * c5 + e5); pm2 += w * (a6 + 2.0 * b6 + 2.0 * c6 + e6)
        pev1 += w * (a7 + 2.0 * b7 + 2.0 * c7 + e7); pev2 += w * (a8 + 2.0 * b8 + 2.0 * c8 + e8)
        z1 += w * (a9 + 2.0 * b9 + 2.0 * c9 + e9); z2 += w * (a10 + 2.0 * b10 + 2.0 * c10 + e10)
        pt += w * (a11 + 2.0 * b11 + 2.0 * c11 + e11)
        out_f[k, 0] = f1; out_f[k, 1] = f2; out_tie[k] = pt
        out_ev[k, 0] = pev1; out_ev[k, 1] = pev2
        out_ace[k, 0] = B1 * f1 + pt; out_ace[k, 1] = B2 * f2 - pt
    return out_f, out_tie, out_ev, out_ace


class TwoAreaLFC:
    def __init__(self, p_base_mw=500.0, H=(5.0, 4.0), D=(1.0, 1.0), R=(0.05, 0.05),
                 Tg=(0.08, 0.10), Tt=(0.40, 0.45), KI=(0.05, 0.05), T12=0.08, Tev=0.15,
                 dt=0.02, window_s=120.0, df_full_hz=0.2):
        self.pb = p_base_mw
        self.H0 = np.array(H, float); self.D = np.array(D, float)
        self.R = np.array(R, float); self.Tg = np.array(Tg, float)
        self.Tt = np.array(Tt, float); self.KI = np.array(KI, float)
        self.T12 = T12; self.Tev = Tev; self.dt = dt
        self.n = int(round(window_s / dt)); self.df_full = df_full_hz / F0
        self.t = np.arange(self.n) * dt

    def run(self, H, reserve_mw, dPL_pu):
        """H: inertia constants (s) for this hour; reserve_mw: V2G regulation
        reserve per area (MW); dPL_pu: (n, 2) load disturbance in p.u."""
        res = np.asarray(reserve_mw, float) / self.pb
        kev = res / self.df_full
        f, tie, ev, ace = _simulate(self.dt, dPL_pu.shape[0], np.asarray(H, float), self.D, self.R,
                                    self.Tg, self.Tt, self.KI, self.T12, self.Tev,
                                    kev, res, dPL_pu)
        return dict(df_hz=f * F0, ptie_mw=tie * self.pb, pev_mw=ev * self.pb, ace=ace)

    @staticmethod
    def metrics(sim, dt):
        df = sim["df_hz"]
        w = int(round(0.5 / dt))  # RoCoF over a 500 ms sliding window
        rocof = np.abs(df[w:] - df[:-w]) / (w * dt)
        return dict(
            nadir_hz=np.max(np.abs(df), axis=0),
            rms_hz=np.sqrt(np.mean(df ** 2, axis=0)),
            rocof_hz_s=np.max(rocof, axis=0),
            tie_rms_mw=float(np.sqrt(np.mean(sim["ptie_mw"] ** 2))),
            tie_max_mw=float(np.max(np.abs(sim["ptie_mw"]))),
            ace_rms=np.sqrt(np.mean(sim["ace"] ** 2, axis=0)),
        )
