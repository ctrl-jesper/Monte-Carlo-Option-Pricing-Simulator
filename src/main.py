import argparse
import numpy as np
from simulation import simulate_gbm, mc_european, mc_asian, black_scholes, compute_greeks, convergence_study
from visualization import animate_and_show


def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo Option Pricing Simulator")
    p.add_argument("--S0", type=float, default=100.0, help="Initial spot price")
    p.add_argument("--K", type=float, default=100.0, help="Strike price")
    p.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    p.add_argument("--sigma", type=float, default=0.20, help="Volatility")
    p.add_argument("--T", type=float, default=1.0, help="Time to maturity (years)")
    p.add_argument("--paths", type=int, default=5000, help="Number of simulation paths")
    p.add_argument("--steps", type=int, default=252, help="Steps per path")
    p.add_argument("--no-antithetic", dest="antithetic", action="store_false",
                   help="Disable antithetic variates")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\nSimulating {args.paths:,} paths ({args.steps} steps each)...")
    t, paths = simulate_gbm(args.S0, args.r, args.sigma, args.T, args.steps, args.paths,
                             antithetic=args.antithetic)

    mc_call = mc_european(paths, args.K, args.r, args.T, "call")
    mc_put = mc_european(paths, args.K, args.r, args.T, "put")
    mc_asian_call = mc_asian(paths, args.K, args.r, args.T, "call")
    bs_call = black_scholes(args.S0, args.K, args.r, args.sigma, args.T, "call")
    bs_put = black_scholes(args.S0, args.K, args.r, args.sigma, args.T, "put")
    greeks = compute_greeks(args.S0, args.K, args.r, args.sigma, args.T)

    print(f"\n{'Method':<20} {'Call':>10} {'Put':>10}")
    print("-" * 42)
    print(f"{'Monte Carlo':<20} {mc_call:>10.4f} {mc_put:>10.4f}")
    print(f"{'Black-Scholes':<20} {bs_call:>10.4f} {bs_put:>10.4f}")
    print(f"{'Asian MC (call)':<20} {mc_asian_call:>10.4f} {'—':>10}")
    print(f"\nGreeks (call): {greeks}")

    path_counts = [100, 250, 500, 1000, 2000, 5000]
    convergence_data = convergence_study(args.S0, args.K, args.r, args.sigma, args.T,
                                         args.steps, path_counts, antithetic=args.antithetic)

    animate_and_show(t, paths, args.K, args.r, args.T,
                     mc_call, bs_call, mc_put, bs_put, mc_asian_call,
                     greeks, convergence_data)


if __name__ == "__main__":
    main()
