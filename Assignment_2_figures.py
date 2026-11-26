"""
Assignment 2 - Airport security screening: all figures, nothing else.

Run from anywhere:  python Assignment_2_figures.py
Output (in the folder figures/):
    one PNG per figure, for the slides
    Assignment_2_figures.pdf with all figures, one per page

The simulator, parameters and random seeds are the same as in Assignment_2_extra_figures.ipynb,
so the figures are the same as in the notebook.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")   # draw to files, no windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from openpyxl import load_workbook
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # data files are next to this script
OUT_DIR = "figures"
DPI = 200
os.makedirs(OUT_DIR, exist_ok=True)
pdf = PdfPages(os.path.join(OUT_DIR, "Assignment_2_figures.pdf"))


def save(fig, name):
    # save one figure as PNG and add it as a page to the PDF
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=DPI, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("saved", name + ".png")


# ---------------------------------------------------------------------------
# Part 1: arrival rate per lane per minute
# ---------------------------------------------------------------------------
df = pd.read_csv("airport.csv")
sept = df[(df["Month"] == "September") & (df["Year"].between(2015, 2019))]
lane_rate = sept["Total Passengers"].mean() / 30 / 16 / 50 / 60   # 30 days, 16 h, 50 lanes, per minute


# ---------------------------------------------------------------------------
# Simulator (same as the notebook)
# ---------------------------------------------------------------------------
def service_time(mean_B, sigma, rng):
    # draw from a normal distribution, resample if the value is negative
    x = rng.normal(mean_B, sigma)
    while x < 0:
        x = rng.normal(mean_B, sigma)
    return x


def discrete_event_simulator(lam, mean_B, sigma, servers, n_passengers, warmup=0, seed=None):
    rng = np.random.default_rng(seed)
    server_free_at = [0.0] * servers
    t = 0.0
    waits = []
    for i in range(n_passengers):
        t = t + rng.exponential(1 / lam)
        k = server_free_at.index(min(server_free_at))
        start = max(t, server_free_at[k])
        waits.append(start - t)
        server_free_at[k] = start + service_time(mean_B, sigma, rng)
    return np.array(waits[warmup:])


def simulate_times(lam, mean_B, sigma, servers, n_passengers, seed=None):
    # same random draws as discrete_event_simulator, returns arrival, start and departure times
    rng = np.random.default_rng(seed)
    server_free_at = [0.0] * servers
    t = 0.0
    arrival = np.empty(n_passengers)
    start = np.empty(n_passengers)
    departure = np.empty(n_passengers)
    for i in range(n_passengers):
        t = t + rng.exponential(1 / lam)
        k = server_free_at.index(min(server_free_at))
        s = max(t, server_free_at[k])
        arrival[i], start[i] = t, s
        server_free_at[k] = s + service_time(mean_B, sigma, rng)
        departure[i] = server_free_at[k]
    return arrival, start, departure


def run_replications(R, lam, mean_B, sigma, servers, n_passengers, warmup=0, seed_offset=0):
    return np.array([discrete_event_simulator(lam, mean_B, sigma, servers, n_passengers,
                                              warmup=warmup, seed=seed_offset + r).mean()
                     for r in range(R)])


def confidence_interval(data, alpha=0.05):
    n = len(data)
    half_width = stats.t.ppf(1 - alpha / 2, df=n - 1) * data.std(ddof=1) / np.sqrt(n)
    return data.mean() - half_width, data.mean() + half_width


R = 40
alpha = 0.05


# ---------------------------------------------------------------------------
# Part 2A: simulation vs P-K formula
# ---------------------------------------------------------------------------
mean_B, sigma = 1.0, 0.25
lam_test = 0.85 / mean_B
Wq_theory = lam_test * (mean_B**2 + sigma**2) / (2 * (1 - lam_test * mean_B))
Wq_reps = run_replications(R, lam_test, mean_B, sigma, servers=1, n_passengers=51000, warmup=1000)
Wq_sim = Wq_reps.mean()
ci_low, ci_high = confidence_interval(Wq_reps, alpha)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(range(1, R + 1), Wq_reps, s=20, label="Replication mean")
axes[0].axhline(Wq_theory, color="red", label=f"P-K theory = {Wq_theory:.3f}")
axes[0].axhline(Wq_sim, color="black", linestyle="--", label=f"Simulation mean = {Wq_sim:.3f}")
axes[0].axhspan(ci_low, ci_high, color="gray", alpha=0.3, label="95% CI")
axes[0].set_xlabel("Replication")
axes[0].set_ylabel("Mean waiting time (min)")
axes[0].set_title("Mean waiting time per replication")
axes[0].legend()

axes[1].hist(Wq_reps, bins=12, edgecolor="black")
axes[1].axvline(Wq_theory, color="red", label="P-K theory")
axes[1].axvline(Wq_sim, color="black", linestyle="--", label="Simulation mean")
axes[1].set_xlabel("Mean waiting time (min)")
axes[1].set_ylabel("Number of replications")
axes[1].set_title("Distribution of replication means")
axes[1].legend()

fig.suptitle("Part 2A: simulation vs P-K formula (M/G/1, rho = 0.85)")
plt.tight_layout()
save(fig, "part2A_validation")


# ---------------------------------------------------------------------------
# Part 2B: Baseline, Option A, Option B
# ---------------------------------------------------------------------------
n_passengers_2b = 3000
scenarios = {
    "Baseline": {"servers": 1, "sigma": 0.25, "mean_B": 1.0},
    "Option A": {"servers": 2, "sigma": 0.25, "mean_B": 1.0},
    "Option B": {"servers": 1, "sigma": 0.10, "mean_B": 1.0},
}
results_2b = {}
for i, (name, par) in enumerate(scenarios.items()):
    results_2b[name] = run_replications(R, lane_rate, par["mean_B"], par["sigma"], par["servers"],
                                        n_passengers=n_passengers_2b, seed_offset=1000 * (i + 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
names = list(scenarios.keys())
axes[0].boxplot([results_2b[name] for name in names])
axes[0].set_xticks(range(1, len(names) + 1))
axes[0].set_xticklabels(names)
axes[0].set_ylabel("Mean waiting time (min)")
axes[0].set_title("Mean waiting time over 40 replications")

line_style = {
    "Baseline": {"color": "C0", "linestyle": "-", "linewidth": 9, "alpha": 0.35, "zorder": 1},
    "Option A": {"color": "C1", "linestyle": "-", "linewidth": 1.5, "zorder": 2},
    "Option B": {"color": "C2", "linestyle": "--", "linewidth": 1.5, "zorder": 3},
}
for i, (name, par) in enumerate(scenarios.items()):
    waits = discrete_event_simulator(lane_rate, par["mean_B"], par["sigma"], par["servers"],
                                     n_passengers_2b, seed=1000 * (i + 1))
    axes[1].plot(range(1, n_passengers_2b + 1), waits, label=name, **line_style[name])
axes[1].set_xlabel("Passenger number")
axes[1].set_ylabel("Waiting time (min)")
axes[1].set_title("Waiting time per passenger (one replication)")
axes[1].legend()

fig.suptitle("Part 2B: comparison of the scenarios")
plt.tight_layout()
save(fig, "part2B_scenarios")

style = {
    "Baseline": {"color": "C0", "linestyle": "-"},
    "Option A": {"color": "C1", "linestyle": "-"},
    "Option B": {"color": "C2", "linestyle": "--"},
}


# ---------------------------------------------------------------------------
# Figure 1: transient behaviour vs fluid approximation
# ---------------------------------------------------------------------------
n_idx = np.arange(1, n_passengers_2b + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, (name, par) in enumerate(scenarios.items()):
    waits = np.array([discrete_event_simulator(lane_rate, par["mean_B"], par["sigma"], par["servers"],
                                               n_passengers_2b, seed=1000 * (i + 1) + r)
                      for r in range(R)])
    mean_w = waits.mean(axis=0)
    lo, hi = np.percentile(waits, [2.5, 97.5], axis=0)
    slope = par["mean_B"] / par["servers"] - 1 / lane_rate
    theory = (n_idx - 1) * slope

    axes[0].fill_between(n_idx, lo, hi, color=style[name]["color"], alpha=0.15, linewidth=0)
    axes[0].plot(n_idx, mean_w, linewidth=2, label=f"{name} (simulation)", **style[name])
    axes[1].plot(n_idx, mean_w - theory, linewidth=1.5, label=name, **style[name])
    if name != "Option B":   # Option B has the same theory line as the baseline
        axes[0].plot(n_idx, theory, color="black", linestyle=":", linewidth=1.5)
        axes[0].annotate(f"slope = {slope:.3f} min/passenger", xy=(n_idx[-1] * 0.6, theory[-1] * 0.6),
                         xytext=(12, -14), textcoords="offset points", ha="left", fontsize=9)

axes[0].plot([], [], color="black", linestyle=":", label="Fluid approximation")
axes[0].set_xlabel("Passenger number")
axes[0].set_ylabel("Waiting time (min)")
axes[0].set_title("Mean over 40 replications (band = 95% of replications)")
axes[0].legend()
axes[0].grid(alpha=0.3)

noise = 1.96 * np.sqrt((n_idx - 1) * (0.25**2 + 1 / lane_rate**2) / R)
axes[1].fill_between(n_idx, -noise, noise, color="gray", alpha=0.2, linewidth=0,
                     label="Expected noise, 95% (1 server)")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_xlabel("Passenger number")
axes[1].set_ylabel("Simulation - approximation (min)")
axes[1].set_title("Difference with the fluid approximation")
axes[1].set_ylim(-10, 12)
axes[1].legend(loc="upper left", framealpha=0.9)
axes[1].grid(alpha=0.3)

fig.suptitle("Figure 1: transient waiting time grows linearly, as predicted for rho > 1")
plt.tight_layout()
save(fig, "figure1_fluid_approximation")


# ---------------------------------------------------------------------------
# Figure 2: waiting time against the number of servers
# ---------------------------------------------------------------------------
server_range = range(1, 7)
sigmas = {0.25: {"color": "C0", "marker": "o", "linestyle": "-"},
          0.10: {"color": "C2", "marker": "s", "linestyle": "--"}}

rows = []
for j, (sig, st) in enumerate(sigmas.items()):
    for c in server_range:
        data = run_replications(R, lane_rate, 1.0, sig, c, n_passengers=n_passengers_2b,
                                seed_offset=10000 + 1000 * j + 100 * c)
        low, high = confidence_interval(data, alpha)
        rows.append({"sigma": sig, "servers": c, "rho": lane_rate * 1.0 / c,
                     "mean": data.mean(), "low": low, "high": high})
sweep = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(8, 5))
c_critical = lane_rate * 1.0   # rho = 1 when c = lambda * E[B]
ax.axvspan(0.5, c_critical, color="gray", alpha=0.15)
ax.axvline(c_critical, color="gray", linestyle=":")
ax.text(c_critical + 0.08, 0.95, f"rho = 1\n(c = {c_critical:.2f})", ha="left", va="top",
        transform=ax.get_xaxis_transform(), color="dimgray")
ax.text(0.6, 0.04, "unstable (rho > 1)", transform=ax.get_xaxis_transform(), va="bottom", color="dimgray")

for sig, st in sigmas.items():
    d = sweep[sweep["sigma"] == sig]
    err = np.vstack([d["mean"] - d["low"], d["high"] - d["mean"]])
    ax.errorbar(d["servers"], d["mean"], yerr=err, capsize=4, markersize=7, linewidth=2,
                label=f"sigma = {sig:.2f} min", **st)

for _, row in sweep[sweep["sigma"] == 0.25].iterrows():
    ax.annotate(f"{row['mean']:.2f} min\nrho = {row['rho']:.2f}",
                xy=(row["servers"], row["mean"]), xytext=(8, 6), textcoords="offset points", fontsize=8)

ax.set_yscale("log")
ax.set_ylim(0.05, 8000)
ax.set_xticks(list(server_range))
ax.set_xlim(0.5, 6.9)
ax.set_xlabel("Number of servers in the lane (c)")
ax.set_ylabel("Mean waiting time, 3000 passengers (min, log scale)")
ax.set_title("Figure 2: waiting time against the number of servers (R = 40, 95% CI)")
ax.legend(loc="center right")
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
save(fig, "figure2_servers")


# ---------------------------------------------------------------------------
# Figure 3: queue length over the day
# ---------------------------------------------------------------------------
all_scen = {**scenarios, "5 servers": {"servers": 5, "sigma": 0.25, "mean_B": 1.0}}
style["5 servers"] = {"color": "C4", "linestyle": ":"}
label_offset = {"Baseline": -9, "Option A": 0, "Option B": 9, "5 servers": 0}   # points, avoids overlap

sims = {}
for i, (name, par) in enumerate(all_scen.items()):
    sims[name] = [simulate_times(lane_rate, par["mean_B"], par["sigma"], par["servers"],
                                 n_passengers_2b, seed=1000 * (i + 1) + r) for r in range(R)]

# stop at the earliest last arrival: after that no new passengers enter the model
t_end = min(arr[-1] for runs in sims.values() for arr, _, _ in runs)
grid = np.linspace(0, t_end, 600)

fig, ax = plt.subplots(figsize=(9, 5))
for name, runs in sims.items():
    # people in queue at time t = arrived by t - started service by t
    q = np.array([np.searchsorted(arr, grid, side="right") - np.searchsorted(start, grid, side="right")
                  for arr, start, _ in runs])
    mean_q = q.mean(axis=0)
    lo, hi = np.percentile(q, [2.5, 97.5], axis=0)
    ax.fill_between(grid / 60, lo, hi, color=style[name]["color"], alpha=0.15, linewidth=0)
    ax.plot(grid / 60, mean_q, linewidth=2, label=name, **style[name])
    ax.annotate(f"{name}: {mean_q[-1]:.0f}", xy=(grid[-1] / 60, mean_q[-1]),
                xytext=(6, label_offset[name]), textcoords="offset points", va="center", fontsize=9)

ax.set_xlim(0, t_end / 60 * 1.18)
ax.set_xlabel("Hours since the lane opened")
ax.set_ylabel("People waiting in the queue")
ax.set_title("Figure 3: queue length over the day (mean of 40 replications, band = 95%)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, "figure3_queue_length")


# ---------------------------------------------------------------------------
# Bonus data: Schiphol September traffic reports 2015-2019
# ---------------------------------------------------------------------------
def read_report(path):
    # column C has the labels, column D the September value; look values up by label
    ws = load_workbook(path, data_only=True).worksheets[0]
    values = {}
    for label, value in ws.iter_rows(min_col=3, max_col=4, values_only=True):
        if isinstance(label, str) and isinstance(value, (int, float)):
            values.setdefault(" ".join(label.split()), value)
    return values


rows = []
for path in glob.glob("Traffic*Transport_*_09_Eng.xlsx"):
    v = read_report(path)
    rows.append({"Year": int(re.search(r"_(\d{4})_09_", path).group(1)),
                 "Passengers": v["Passengers (*)"], "O&D": v["O & D"],
                 "Transfer": v["Transfer"], "Flights": v["Air Transport movements"]})
report = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
report["Pax per flight"] = (report["Passengers"] / report["Flights"]).round(1)
cagr = (report.iloc[-1] / report.iloc[0]) ** (1 / 4) - 1

orig_share = report["O&D"].mean() / 2 / report["Passengers"].mean()
shares = {"All departing passengers": 0.5,
          "Originating only (transfers screened separately)": orig_share}


# ---------------------------------------------------------------------------
# Figure 4: passengers and flights
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for col, color in [("O&D", "C1"), ("Transfer", "C7")]:
    y = report[col] / 1e6
    axes[0].plot(report["Year"], y, marker="o", linewidth=2, color=color, label=col)
    axes[0].annotate(f"{col}: {cagr[col]:+.1%}/yr", xy=(2019, y.iloc[-1]), xytext=(10, 0),
                     textcoords="offset points", va="center", fontsize=9)
axes[0].set_ylim(0, 4.8)
axes[0].set_xlim(2014.7, 2020.6)
axes[0].set_ylabel("Passengers (millions)")
axes[0].set_title("Passengers in September")
axes[0].legend(loc="lower left")

axes[1].plot(report["Year"], report["Flights"] / 1000, marker="o", linewidth=2, color="0.25")
for i, ha in [(0, "left"), (len(report) - 1, "right")]:
    axes[1].annotate(f"{report['Pax per flight'][i]:.0f} passengers per flight",
                     xy=(report["Year"][i], report["Flights"][i] / 1000), xytext=(0, -18),
                     textcoords="offset points", ha=ha, fontsize=9)
axes[1].set_ylim(0, 50)
axes[1].set_xlim(2014.7, 2019.3)
axes[1].set_ylabel("Flights (thousands)")
axes[1].set_title("Flights in September")

for ax in axes:
    ax.set_xticks(report["Year"])
    ax.set_xlabel("September of year")
    ax.grid(alpha=0.3)
fig.suptitle("Figure 4: passengers keep growing, flights stop growing after 2017")
plt.tight_layout()
save(fig, "figure4_schiphol_traffic")


# ---------------------------------------------------------------------------
# Figure 5: strategies under the corrected demand (one 16-hour day per lane)
# ---------------------------------------------------------------------------
strategies = {
    "Baseline":                     {"servers": 1, "sigma": 0.25, "mean_B": 1.0},
    "Option A: 2 servers":          {"servers": 2, "sigma": 0.25, "mean_B": 1.0},
    "Option B: sigma = 0.10":       {"servers": 1, "sigma": 0.10, "mean_B": 1.0},
    "Faster screening: E[B] = 0.7": {"servers": 1, "sigma": 0.25, "mean_B": 0.7},   # assumed value
}
demand_levels = list(shares.keys())

bonus = {}
for j, dname in enumerate(demand_levels):
    lam = lane_rate * shares[dname]
    for k, (sname, par) in enumerate(strategies.items()):
        bonus[(dname, sname)] = run_replications(R, lam, par["mean_B"], par["sigma"], par["servers"],
                                                 n_passengers=round(lam * 16 * 60),
                                                 seed_offset=20000 + 1000 * j + 100 * k)

# both demand levels on the same row; direct labels on the baseline row:
# "originating only" to the left of its marker, "all departing" to the right of its marker
demand_style = {
    "All departing passengers": {"color": "C0", "label": "all departing", "dx": 8, "ha": "left"},
    "Originating only (transfers screened separately)": {"color": "C1", "label": "originating only",
                                                         "dx": -8, "ha": "right"},
}
names = list(strategies.keys())

fig, ax = plt.subplots(figsize=(9, 4.5))
for dname, st in demand_style.items():
    lam = lane_rate * shares[dname]
    for k, sname in enumerate(names):
        data = bonus[(dname, sname)]
        low, high = confidence_interval(data, alpha)
        par = strategies[sname]
        stable = lam * par["mean_B"] / par["servers"] < 1
        # filled marker = stable lane (rho < 1), open marker = queue keeps growing (rho >= 1)
        ax.errorbar(data.mean(), k, xerr=[[data.mean() - low], [high - data.mean()]],
                    fmt="o", markersize=8, color=st["color"], markeredgewidth=1.5, capsize=3,
                    markerfacecolor=st["color"] if stable else "white")
        if sname == "Baseline":
            ax.annotate(st["label"], xy=(high if st["dx"] > 0 else low, k), xytext=(st["dx"], 0),
                        textcoords="offset points", ha=st["ha"], va="center", fontsize=9)

ax.set_xscale("log")
ax.set_xlim(0.1, 5000)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(["Baseline: 1 server" if n == "Baseline" else n for n in names])
ax.invert_yaxis()
ax.set_xlabel("Mean waiting time per passenger over one 16-hour day (min, log scale)")
ax.set_title("Figure 5: strategies with demand from the Schiphol statistics (R = 40, 95% CI)")
ax.grid(alpha=0.3, which="both", axis="x")

handles = [
    Line2D([], [], color="C0", marker="o", linestyle="", markersize=8,
           label=f"All departing passengers (lambda = {lane_rate * 0.5:.2f})"),
    Line2D([], [], color="C1", marker="o", linestyle="", markersize=8,
           label=f"Originating only (lambda = {lane_rate * orig_share:.2f})"),
    Line2D([], [], color="gray", marker="o", linestyle="", markersize=8, label="rho < 1: stable"),
    Line2D([], [], color="gray", marker="o", linestyle="", markersize=8, markerfacecolor="white",
           markeredgewidth=1.5, label="rho >= 1: queue keeps growing"),
]
# legend below the plot, so it never covers a marker
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9, frameon=False)
plt.tight_layout()
save(fig, "figure5_strategies")

pdf.close()
print("done, all figures are in", os.path.abspath(OUT_DIR))
