# Monte Carlo Option Pricing Simulator

Educational option pricing engine implementing Monte Carlo simulation with antithetic variates, Black-Scholes analytical pricing, Asian option valuation, and full Greeks computation. Designed to illustrate the convergence behaviour of Monte Carlo estimators and the relationship between simulation-based and closed-form results.

## What it does

- Simulates Geometric Brownian Motion (GBM) paths under the risk-neutral measure
- Prices European call and put options via Monte Carlo
- Prices arithmetic average Asian call options via Monte Carlo
- Computes closed-form Black-Scholes prices for direct comparison
- Implements antithetic variates to reduce variance of the MC estimator
- Computes option Greeks (Delta, Gamma, Vega, Theta, Rho) analytically
- Runs a convergence study across increasing path counts to show estimator behaviour

## Output panels

| Panel | Content |
|---|---|
| GBM Paths | Animated simulation of 30 sample paths with strike line |
| Convergence | MC call price vs path count, with BS reference line |
| Terminal Distribution | Histogram of S(T) values across all paths |
| Greeks | Bar chart of Delta, Gamma, Vega, Theta, Rho |
| Summary Table | Side-by-side MC vs BS pricing for calls, puts, and Asian call |

## Installation

```bash
git clone https://github.com/jespermathiasnielsen/Monte-Carlo-Option-Pricing-Simulator.git
cd Monte-Carlo-Option-Pricing-Simulator
pip install -r requirements.txt
```

## Usage

```bash
# Default parameters (ATM call, 20% vol, 5% rate, 1 year)
python src/main.py

# Custom contract
python src/main.py --S0 100 --K 105 --sigma 0.25 --r 0.04 --T 0.5 --paths 10000

# Disable antithetic variates to see the variance difference
python src/main.py --no-antithetic --paths 1000
```

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--S0` | `100` | Initial spot price |
| `--K` | `100` | Strike price |
| `--r` | `0.05` | Risk-free rate |
| `--sigma` | `0.20` | Implied volatility |
| `--T` | `1.0` | Time to maturity (years) |
| `--paths` | `5000` | Number of simulation paths |
| `--steps` | `252` | Steps per path (252 = daily for 1 year) |
| `--no-antithetic` | — | Disable variance reduction |

## Antithetic variates

For each random draw Z ~ N(0,1), the simulator also runs a path driven by -Z. The two paths are negatively correlated, which substantially reduces the variance of the estimator. For a fixed path budget, antithetic variates typically halve the standard error of the price estimate.

## Pricing model

GBM under risk-neutral measure:

```
dS = r·S·dt + σ·S·dW
```

Discretised using the exact log-normal step:

```
S(t+dt) = S(t) · exp((r - σ²/2)·dt + σ·√dt·Z)
```

## License

MIT
