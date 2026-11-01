"""Daily profiles and scenario sampling (hourly resolution, hour-mid values)."""
import numpy as np

HOURS = np.arange(24) + 0.5


def _g(t, mu, sig):
    return np.exp(-0.5 * ((t - mu) / sig) ** 2)


def residential_load():
    """Residential-dominated feeder: morning shoulder, evening peak at 19-21 h."""
    x = 0.42 + 0.22 * _g(HOURS, 8.0, 1.6) + 0.58 * _g(HOURS, 20.0, 2.0) \
        + 0.10 * _g(HOURS, 13.0, 3.0)
    return x / x.max()


def mixed_load():
    """Mixed commercial-residential feeder: broad daytime plateau and evening peak."""
    x = 0.40 + 0.30 * _g(HOURS, 13.5, 3.2) + 0.40 * _g(HOURS, 19.5, 1.8)
    return x / x.max()


def clear_sky_pv():
    s = np.sin(np.pi * (HOURS - 6.0) / 12.0)
    return np.clip(s, 0.0, None) ** 1.3


def residential_availability():
    """Fraction of the residential fleet plugged in."""
    t = HOURS
    dep = 1.0 / (1.0 + np.exp(-(t - 7.5) / 0.6))
    arr = 1.0 / (1.0 + np.exp(-(t - 17.8) / 0.8))
    return 0.9 - 0.6 * dep + 0.55 * arr - 0.0 * t  # 0.9 night, 0.3 day, 0.85 evening


def workplace_availability():
    t = HOURS
    arr = 1.0 / (1.0 + np.exp(-(t - 8.0) / 0.6))
    dep = 1.0 / (1.0 + np.exp(-(t - 17.0) / 0.6))
    return 0.08 + 0.80 * arr - 0.80 * dep


def sample_scenario(rng, split="train"):
    """Randomised operating day. 'stress' gives the held-out extreme set."""
    if split == "stress":
        load_scale = rng.uniform(1.10, 1.20, 2)
        cloud = rng.uniform(0.9, 1.0, 2)
        part = rng.uniform(0.55, 0.70, 2)
        dist_scale = rng.uniform(1.3, 1.6)
    else:
        load_scale = rng.uniform(0.85, 1.10, 2)
        cloud = rng.uniform(0.3, 1.0, 2)
        part = rng.uniform(0.7, 1.1, 2)
        dist_scale = rng.uniform(0.8, 1.2)
    pv_shape = clear_sky_pv()
    pv = np.stack([pv_shape * cloud[i] * (1 + 0.25 * (1 - cloud[i])
                   * rng.standard_normal(24)).clip(0.3, 1.5) for i in range(2)])
    pv = pv.clip(0.0, 1.0)
    load = np.stack([residential_load(), mixed_load()]) * load_scale[:, None]
    load = load * (1 + 0.03 * rng.standard_normal((2, 24)))
    avail = np.stack([residential_availability(), workplace_availability()])
    soc0 = rng.uniform(0.5, 0.6, 2)
    return dict(load=load, pv=pv, avail=avail.clip(0.02, 0.95), participation=part,
                cloud=cloud, dist_scale=dist_scale, soc0=soc0,
                seed=int(rng.integers(1 << 31)))
