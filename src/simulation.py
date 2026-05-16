import numpy as np
from scipy.stats import norm


def simulate_gbm(S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int,
                 antithetic: bool = True) -> tuple:
    """Simulates Geometric Brownian Motion paths.

    Antithetic variates: for each random draw Z, also simulate -Z.
    This halves variance of the price estimate at no extra cost in paths run.
    """
    dt = T / steps
    t = np.linspace(0, T, steps + 1)

    if antithetic:
        half = n_paths // 2
        Z = np.random.standard_normal((half, steps))
        Z = np.concatenate([Z, -Z], axis=0)
    else:
        Z = np.random.standard_normal((n_paths, steps))

    log_increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((len(Z), 1)), np.cumsum(log_increments, axis=1)], axis=1)
    paths = S0 * np.exp(log_paths)
    return t, paths


def mc_european(paths: np.ndarray, K: float, r: float, T: float, option_type: str = "call") -> float:
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
    return float(np.exp(-r * T) * np.mean(payoff))


def mc_asian(paths: np.ndarray, K: float, r: float, T: float, option_type: str = "call") -> float:
    """Arithmetic average Asian option — payoff based on mean price over the path."""
    avg = paths.mean(axis=1)
    payoff = np.maximum(avg - K, 0) if option_type == "call" else np.maximum(K - avg, 0)
    return float(np.exp(-r * T) * np.mean(payoff))


def black_scholes(S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))


def compute_greeks(S0: float, K: float, r: float, sigma: float, T: float) -> dict:
    """Computes Black-Scholes Greeks for a call option."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = float(norm.cdf(d1))
    gamma = float(norm.pdf(d1) / (S0 * sigma * np.sqrt(T)))
    vega = float(S0 * norm.pdf(d1) * np.sqrt(T) / 100)   # per 1% vol move
    theta = float((-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                   - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
    rho = float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100)   # per 1% rate move
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def convergence_study(S0, K, r, sigma, T, steps, path_counts, antithetic=True) -> list:
    """Runs MC pricing at increasing path counts to produce a convergence curve."""
    results = []
    for n in path_counts:
        _, paths = simulate_gbm(S0, r, sigma, T, steps, n, antithetic=antithetic)
        price = mc_european(paths, K, r, T, "call")
        results.append({"n_paths": n, "mc_price": price})
    return results
