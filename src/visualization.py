import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")


def animate_and_show(t, paths, K, r, T, mc_call, bs_call, mc_put, bs_put, mc_asian_call,
                     greeks, convergence_data, n_display=30):
    fig = plt.figure(figsize=(15, 10), facecolor="#111")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    ax_paths = fig.add_subplot(gs[0, :2])
    ax_conv = fig.add_subplot(gs[0, 2])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_greeks = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[1, 2])

    for ax in [ax_paths, ax_conv, ax_dist, ax_greeks, ax_table]:
        ax.set_facecolor("#111")
        ax.tick_params(colors="#999")
        ax.grid(alpha=0.12)
        for spine in ax.spines.values():
            spine.set_color("#333")

    # --- Path animation ---
    ax_paths.set_title("Monte Carlo GBM Paths (antithetic variates)", color="#00e5ff", fontsize=13)
    ax_paths.set_xlabel("Time (years)", color="#aaa")
    ax_paths.set_ylabel("Price", color="#aaa")
    ax_paths.set_xlim(0, T)
    ax_paths.set_ylim(np.min(paths[:n_display]) * 0.95, np.max(paths[:n_display]) * 1.05)
    ax_paths.axhline(K, color="#ff4081", linestyle="--", linewidth=1.2, label=f"Strike K={K}", alpha=0.7)

    lines = [ax_paths.plot([], [], lw=0.8, alpha=0.55)[0] for _ in range(n_display)]
    price_box = ax_paths.text(0.98, 0.97, "", transform=ax_paths.transAxes,
                              fontsize=10, color="white", va="top", ha="right",
                              bbox=dict(facecolor="#222", alpha=0.8, boxstyle="round"))
    ax_paths.legend(fontsize=9, frameon=False, loc="upper left")

    cmap = plt.cm.plasma
    for i, line in enumerate(lines):
        line.set_color(cmap(i / n_display))

    def animate(frame):
        for j, line in enumerate(lines):
            line.set_data(t[:frame], paths[j, :frame])
        price_box.set_text(
            f"MC Call: {mc_call:.4f}\nBS Call: {bs_call:.4f}\n"
            f"MC Put:  {mc_put:.4f}\nBS Put:  {bs_put:.4f}\n"
            f"Asian Call: {mc_asian_call:.4f}"
        )
        return lines + [price_box]

    ani = FuncAnimation(fig, animate, frames=len(t), interval=18, blit=True, repeat=False)

    # --- Convergence plot ---
    ax_conv.set_title("Convergence (MC Call Price)", color="#69ff47", fontsize=11)
    ax_conv.set_xlabel("Paths", color="#aaa")
    ax_conv.set_ylabel("Price", color="#aaa")
    ns = [d["n_paths"] for d in convergence_data]
    prices_conv = [d["mc_price"] for d in convergence_data]
    ax_conv.plot(ns, prices_conv, color="#69ff47", lw=2, marker="o", markersize=4)
    ax_conv.axhline(bs_call, color="#ff4081", linestyle="--", lw=1.2, label=f"BS={bs_call:.4f}")
    ax_conv.legend(fontsize=8, frameon=False)

    # --- Terminal distribution ---
    ax_dist.set_title("Terminal Price Distribution", color="#ffea00", fontsize=11)
    ax_dist.set_xlabel("S(T)", color="#aaa")
    ax_dist.set_ylabel("Frequency", color="#aaa")
    ax_dist.hist(paths[:, -1], bins=60, color="#ffea00", alpha=0.7, edgecolor="none")
    ax_dist.axvline(K, color="#ff4081", linestyle="--", lw=1.2, label=f"Strike={K}")
    ax_dist.legend(fontsize=8, frameon=False)

    # --- Greeks bar chart ---
    ax_greeks.set_title("Option Greeks (Call)", color="#a259ff", fontsize=11)
    greek_names = list(greeks.keys())
    greek_vals = list(greeks.values())
    colors = ["#00e5ff", "#ff4081", "#69ff47", "#ffea00", "#a259ff"]
    bars = ax_greeks.bar(greek_names, greek_vals, color=colors, alpha=0.85)
    ax_greeks.set_ylabel("Value", color="#aaa")
    for bar, val in zip(bars, greek_vals):
        ax_greeks.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                       f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="white")

    # --- Summary table ---
    ax_table.axis("off")
    table_data = [
        ["Method", "Call", "Put"],
        ["Monte Carlo", f"{mc_call:.4f}", f"{mc_put:.4f}"],
        ["Black-Scholes", f"{bs_call:.4f}", f"{bs_put:.4f}"],
        ["Asian MC", f"{mc_asian_call:.4f}", "—"],
        ["Error (call)", f"{abs(mc_call - bs_call):.4f}", "—"],
    ]
    tbl = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333")
        cell.set_facecolor("#111" if row > 0 else "#222")
        cell.set_text_props(color="white")
    ax_table.set_title("Price Summary", color="#00e5ff", fontsize=11)

    plt.suptitle("Monte Carlo Option Pricing Simulator", color="white", fontsize=15, y=1.01)
    plt.show()
