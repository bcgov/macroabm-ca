"""Inspect income distribution at initial state from a simulation HDF5 output.

Usage:
    uv run python inspect_income.py              # defaults to FRA
    uv run python inspect_income.py CAN          # Canada
    uv run python inspect_income.py CAN_BC       # British Columbia (region, auto PIT)
    uv run python inspect_income.py FRA output/custom.h5  # custom file
    uv run python inspect_income.py CAN_BC output/sim_can_bc.h5 BC_PIT_2014.csv
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def gini_coefficient(x):
    """Compute Gini coefficient from an array of values."""
    x = np.sort(x[x > 0])
    n = len(x)
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


REPO_ROOT = Path(__file__).resolve().parent.parent

# Default scale factor — each synthetic agent represents this many real people/households.
# Set by the `scale` parameter in `default_data_configuration()` (default: 10,000).
SCALE = 10_000

# --- Parse command line ---
COUNTRY = sys.argv[1] if len(sys.argv) > 1 else "FRA"
custom_h5 = sys.argv[2] if len(sys.argv) > 2 else None
pit_csv_arg = sys.argv[3] if len(sys.argv) > 3 else None

# Region codes (e.g. CAN_BC) are stored under the parent country key
# inside the HDF5 file.  Resolve the HDF5 group key accordingly.
HDF5_GROUP = COUNTRY.split("_")[0] if "_" in COUNTRY else COUNTRY

# Auto-activate PIT for CAN_BC
if COUNTRY == "CAN_BC" and pit_csv_arg is None:
    pit_csv_arg = "BC_PIT_2014.csv"

if custom_h5:
    H5_PATH = Path(custom_h5)
else:
    H5_PATH = REPO_ROOT / f"output/sim_{COUNTRY.lower()}.h5"

if not H5_PATH.exists():
    alt = REPO_ROOT / "output/sim_fra.h5"
    print(f"ERROR: {H5_PATH} not found.")
    if alt.exists():
        print(f"  (Found {alt} — try running with 'FRA' or pass a custom path as 2nd arg)")
    sys.exit(1)

if COUNTRY != HDF5_GROUP:
    print(f"Note: {COUNTRY} is a region — HDF5 group key is '{HDF5_GROUP}'")


# ── Load PIT schedule (if applicable) ───────────────────────────────
pit_thresholds = pit_rates = pit_lower = pit_quick = pit_basic_deduction = None
pit_label = ""
if pit_csv_arg:
    from macro_data.readers.taxation.personal_income_tax.pit_schedule import PITSchedule

    if "/" in pit_csv_arg or "\\" in pit_csv_arg:
        schedule = PITSchedule.from_csv(pit_csv_arg)
    else:
        schedule = PITSchedule.from_name(pit_csv_arg)
    pit_thresholds, pit_rates, pit_lower, pit_quick = schedule.get_brackets(
        tax_year=schedule.base_year
    )
    pit_basic_deduction = schedule.basic_deduction
    pit_label = f" | PIT: {pit_csv_arg} ({schedule.base_year})"
    if pit_basic_deduction is not None:
        pit_label += f" | basic_deduction=${pit_basic_deduction:,.0f}"
    print(f"PIT schedule loaded: {len(pit_thresholds)} brackets from {pit_csv_arg}")


with h5py.File(H5_PATH, "r") as f:
    if HDF5_GROUP not in f:
        available = list(f.keys())
        print(f"ERROR: group '{HDF5_GROUP}' not found in HDF5. Available: {available}")
        sys.exit(1)

    group = f[HDF5_GROUP]
    print(f"=== Groups in {HDF5_GROUP} (from {COUNTRY}){pit_label} ===")
    for key in group.keys():
        print(f"  {key}")

    ind = group["individuals"]
    hh = group["households"]

    # ---- INDIVIDUAL INCOME (timestep 0 = initial) ----
    ind_income = ind["income"][0]
    ind_emp_income = ind["employee_income"][0]

    print(f"\n===== INDIVIDUAL INCOME (initial state) =====")
    print(f"  NOTE: Each synthetic individual agent represents ~{SCALE:,} real people.")
    print(f"  All income values are aggregated at the synthetic-agent level, not per-capita.")
    print(f"  N = {len(ind_income):,} individuals")
    print(f"  Mean:   {ind_income.mean():,.6f}")
    print(f"  Median: {np.median(ind_income):,.6f}")
    print(f"  Std:    {ind_income.std():,.6f}")
    print(f"  Min:    {ind_income.min():,.6f}")
    print(f"  Max:    {ind_income.max():,.6f}")
    print(f"  Gini:   {gini_coefficient(ind_income):.4f}")
    print(f"\n  Deciles:")
    for p, v in zip(np.arange(0, 101, 10), np.percentile(ind_income, np.arange(0, 101, 10))):
        print(f"    {p:3d}%: {v:,.6f}")
    print(f"\n  Income composition:")
    print(f"    Employed (income>0): {(ind_income > 0).sum()} ({(ind_income > 0).mean()*100:.1f}%)")
    zero = (ind_income == 0).sum()
    print(f"    Zero income:         {zero} ({zero/len(ind_income)*100:.1f}%)")
    print(f"    Mean employee income (employed): {ind_emp_income[ind_emp_income > 0].mean():,.6f}")

    # ---- PIT DISTRIBUTION (if schedule loaded) ----
    if pit_thresholds is not None:
        from macro_data.readers.taxation.personal_income_tax.pit_schedule import compute_progressive_tax

        # Taxable base = employee income (same as in CentralGovernment.compute_taxes)
        pit_tax = compute_progressive_tax(ind_emp_income, pit_thresholds, pit_rates)

        # Apply non-refundable basic personal amount credit when configured.
        if pit_basic_deduction is not None and pit_basic_deduction > 0:
            credit = pit_basic_deduction * float(pit_rates[0])
            pit_tax = np.maximum(0.0, pit_tax - credit)

        total_tax = pit_tax.sum()
        total_taxable = ind_emp_income.sum()
        eff_rate = total_tax / total_taxable if total_taxable > 0 else 0.0

        print(f"\n===== PERSONAL INCOME TAX (initial state) =====")
        if pit_basic_deduction is not None:
            print(f"  Basic personal amount:   ${pit_basic_deduction:,.0f}  "
                  f"(credit @ {pit_rates[0]:.1%} = ${pit_basic_deduction * pit_rates[0]:,.0f})")
        print(f"  Total employee income:  {total_taxable:,.6f}")
        print(f"  Total PIT revenue:      {total_tax:,.6f}")
        print(f"  Effective rate:         {eff_rate:.4%}")
        print(f"  Mean tax per agent:     {pit_tax.mean():,.6f}")
        print(f"  Median tax per agent:   {np.median(pit_tax):,.6f}")
        print(f"  Max tax:                {pit_tax.max():,.6f}")
        print(f"  Zero-tax individuals:   {(pit_tax == 0).sum()} ({(pit_tax == 0).mean()*100:.1f}%)")
        print(f"\n  PIT by employee-income decile:")
        # Use quantile-based bins so each decile gets ~10% of individuals,
        # even when many share the same income value (e.g. zero).
        n = len(ind_emp_income)
        sorted_idx = np.argsort(ind_emp_income)
        sorted_emp = ind_emp_income[sorted_idx]
        sorted_tax = pit_tax[sorted_idx]
        for d in range(10):
            lo_idx = d * n // 10
            hi_idx = (d + 1) * n // 10
            mask = sorted_idx[lo_idx:hi_idx]
            n_dec = len(mask)
            inc_dec = ind_emp_income[mask].sum()
            tax_dec = pit_tax[mask].sum()
            lo_val = sorted_emp[lo_idx]
            hi_val = sorted_emp[min(hi_idx, n - 1)]
            rate_dec = tax_dec / inc_dec if inc_dec > 0 else 0.0
            share_dec = tax_dec / total_tax * 100 if total_tax > 0 else 0.0
            print(
                f"    Decile {d+1:2d} [{lo_val:>12,.0f} – {hi_val:>12,.0f}]: "
                f"n={n_dec:>5d}  total_inc={inc_dec:>14,.0f}  "
                f"tax={tax_dec:>12,.0f}  rate={rate_dec:.1%}  "
                f"share={share_dec:5.1f}%"
            )

    # ---- HOUSEHOLD INCOME (timestep 0 = initial) ----
    hh_income = hh["income"][0]

    print(f"\n===== HOUSEHOLD INCOME (initial state) =====")
    print(f"  NOTE: Each synthetic household represents ~{SCALE:,} real households.")
    print(f"  N = {len(hh_income):,} households")
    print(f"  Mean:   {hh_income.mean():,.6f}")
    print(f"  Median: {np.median(hh_income):,.6f}")
    print(f"  Std:    {hh_income.std():,.6f}")
    print(f"  Min:    {hh_income.min():,.6f}")
    print(f"  Max:    {hh_income.max():,.6f}")
    print(f"  Gini:   {gini_coefficient(hh_income):.4f}")
    print(f"\n  Deciles:")
    for p, v in zip(np.arange(0, 101, 10), np.percentile(hh_income, np.arange(0, 101, 10))):
        print(f"    {p:3d}%: {v:,.6f}")


# ---- Lorenz curve helper ----
def lorenz_curve(x):
    """Return cumulative population share and cumulative income share."""
    x = np.sort(x[x > 0])
    if len(x) == 0:
        return np.array([0, 1]), np.array([0, 1])
    cum_pop = np.linspace(0, 1, len(x))
    cum_inc = np.cumsum(x) / np.sum(x)
    return cum_pop, cum_inc


# ---- PIT helper ----
def pit_decile_breakdown(incomes, taxes, n_deciles=10):
    """Return decile stats using quantile-based bins (equal population per bin)."""
    n = len(incomes)
    sorted_idx = np.argsort(incomes)
    sorted_inc = incomes[sorted_idx]
    sorted_tax = taxes[sorted_idx]

    edges = np.zeros(n_deciles + 1)
    mean_tax = np.zeros(n_deciles)
    mean_rate = np.zeros(n_deciles)
    share = np.zeros(n_deciles)
    total_tax = taxes.sum()

    for d in range(n_deciles):
        lo = d * n // n_deciles
        hi = (d + 1) * n // n_deciles
        mask = sorted_idx[lo:hi]
        edges[d] = sorted_inc[lo]
        edges[d + 1] = sorted_inc[min(hi, n - 1)]
        if mask.size > 0:
            mean_tax[d] = taxes[mask].mean()
            inc_sum = incomes[mask].sum()
            mean_rate[d] = taxes[mask].sum() / inc_sum if inc_sum > 0 else 0
            share[d] = taxes[mask].sum() / total_tax * 100 if total_tax > 0 else 0

    return edges, mean_tax, mean_rate, share


# ---- Plots ----
has_pit = pit_thresholds is not None
n_rows = 3 if has_pit else 2
fig, axes = plt.subplots(n_rows, 3, figsize=(21, 7 * n_rows))
if n_rows == 2:
    axes = axes.reshape(2, 3)  # ensure 2D

# === ROW 1: INDIVIDUALS ===

# Column 1: Individual income histogram (log scale)
ax = axes[0, 0]
pos_income = ind_income[ind_income > 0]
log_bins = np.logspace(np.log10(pos_income.min()), np.log10(pos_income.max()), 50)
ax.hist(pos_income, bins=log_bins, edgecolor="white", alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel(f"Income — log scale (1 agent ≈ {SCALE:,} people)")
ax.set_ylabel("Number of Individuals")
ax.set_title(f"Individual Income — {COUNTRY}{pit_label} (initial state)")
ax.axvline(np.median(ind_income), color="red", linestyle="--", label=f"Median: {np.median(ind_income):.4f}")
ax.axvline(ind_income.mean(), color="green", linestyle="--", label=f"Mean: {ind_income.mean():.4f}")
ax.legend()

# Column 2: Individual decile bar chart
ax = axes[0, 1]
ind_dec = np.percentile(ind_income, np.arange(0, 101, 10))
dec_labels = [f"{d}–{d+10}" for d in range(0, 100, 10)]
dec_means_ind = []
for lo, hi in zip(ind_dec[:-1], ind_dec[1:]):
    mask = (ind_income >= lo) & (ind_income < hi)
    if hi == ind_dec[-1]:
        mask = (ind_income >= lo) & (ind_income <= hi)
    dec_means_ind.append(ind_income[mask].mean() if mask.any() else 0)
ax.bar(dec_labels, dec_means_ind, edgecolor="white", alpha=0.7)
ax.set_xlabel("Income Decile")
ax.set_ylabel("Mean Income")
ax.set_title(f"Individual Mean Income by Decile — {COUNTRY}")
ax.tick_params(axis="x", rotation=45)

# Column 3: Individual Lorenz curve
ax = axes[0, 2]
pop_i, inc_i = lorenz_curve(ind_income)
ax.plot(pop_i, inc_i, linewidth=2, label=f"Gini = {gini_coefficient(ind_income):.4f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect equality")
ax.fill_between(pop_i, pop_i, inc_i, alpha=0.1)
ax.set_xlabel("Cumulative population share")
ax.set_ylabel("Cumulative income share")
ax.set_title(f"Individual Lorenz Curve — {COUNTRY}")
ax.legend(loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# === ROW 2: HOUSEHOLDS ===

# Column 1: Household income histogram (log scale)
ax = axes[1, 0]
pos_hh = hh_income[hh_income > 0]
log_bins_h = np.logspace(np.log10(pos_hh.min()), np.log10(pos_hh.max()), 50)
ax.hist(pos_hh, bins=log_bins_h, edgecolor="white", alpha=0.7, color="orange")
ax.set_xscale("log")
ax.set_xlabel(f"Income — log scale (1 agent ≈ {SCALE:,} households)")
ax.set_ylabel("Number of Households")
ax.set_title(f"Household Income — {COUNTRY} (initial state)")
ax.axvline(np.median(hh_income), color="red", linestyle="--", label=f"Median: {np.median(hh_income):,.4f}")
ax.axvline(hh_income.mean(), color="green", linestyle="--", label=f"Mean: {hh_income.mean():,.4f}")
ax.legend()

# Column 2: Household decile bar chart
ax = axes[1, 1]
hh_dec = np.percentile(hh_income, np.arange(0, 101, 10))
dec_means_hh = []
for lo, hi in zip(hh_dec[:-1], hh_dec[1:]):
    mask = (hh_income >= lo) & (hh_income < hi)
    if hi == hh_dec[-1]:
        mask = (hh_income >= lo) & (hh_income <= hi)
    dec_means_hh.append(hh_income[mask].mean() if mask.any() else 0)
ax.bar(dec_labels, dec_means_hh, edgecolor="white", alpha=0.7, color="orange")
ax.set_xlabel("Income Decile")
ax.set_ylabel("Mean Income")
ax.set_title(f"Household Mean Income by Decile — {COUNTRY}")
ax.tick_params(axis="x", rotation=45)

# Column 3: Household Lorenz curve
ax = axes[1, 2]
pop_h, inc_h = lorenz_curve(hh_income)
ax.plot(pop_h, inc_h, linewidth=2, color="orange", label=f"Gini = {gini_coefficient(hh_income):.4f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect equality")
ax.fill_between(pop_h, pop_h, inc_h, alpha=0.1, color="orange")
ax.set_xlabel("Cumulative population share")
ax.set_ylabel("Cumulative income share")
ax.set_title(f"Household Lorenz Curve — {COUNTRY}")
ax.legend(loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# === ROW 3: PERSONAL INCOME TAX (if PIT schedule loaded) ===
if has_pit:
    # Column 1: PIT histogram
    ax = axes[2, 0]
    pos_tax = pit_tax[pit_tax > 0]
    if len(pos_tax) > 0:
        log_bins_t = np.logspace(np.log10(pos_tax.min()), np.log10(pos_tax.max()), 40)
        ax.hist(pos_tax, bins=log_bins_t, edgecolor="white", alpha=0.7, color="red")
        ax.set_xscale("log")
    ax.set_xlabel("Tax Paid — log scale")
    ax.set_ylabel("Number of Individuals")
    ax.set_title(f"PIT Distribution — {COUNTRY} (initial state)")
    ax.axvline(np.median(pit_tax), color="darkred", linestyle="--",
               label=f"Median tax: {np.median(pit_tax):,.4f}")
    ax.axvline(pit_tax.mean(), color="green", linestyle="--",
               label=f"Mean tax: {pit_tax.mean():,.4f}")
    ax.legend()

    # Column 2: Mean/median tax by employee-income decile (all individuals)
    ax = axes[2, 1]
    pit_edges, pit_mean, pit_rate, pit_share = pit_decile_breakdown(ind_emp_income, pit_tax)
    # Debug prints to inspect numeric values used for plotting (show rates as percent)
    print("\nPIT decile debug (all individuals):")
    print(f"  N individuals: {len(ind_emp_income)}")
    print(f"  pit_mean: {np.round(pit_mean, 6)}")
    print(f"  pit_rate (fraction): {np.round(pit_rate, 6)}")
    print(f"  pit_rate (%): {np.round(pit_rate * 100, 3)}")
    print(f"  pit_share: {np.round(pit_share, 3)}")
    x_pos = np.arange(10)
    width = 0.35
    ax.bar(x_pos - width / 2, pit_mean, width, label="Mean tax", edgecolor="white", alpha=0.85, color="red")
    ax.set_xlabel("Employee-Income Decile")
    ax.set_ylabel("Mean Tax Paid")
    ax.set_title(f"PIT by Employee-Income Decile — {COUNTRY}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dec_labels, rotation=45)
    ax.legend(loc="upper left")

    # Column 3: Effective tax rate by decile + revenue share
    ax = axes[2, 2]
    # Plot effective tax rate as percentage (0-100)
    bars = ax.bar(dec_labels, pit_rate * 100, edgecolor="white", alpha=0.85, color="darkred", label="Effective tax rate")
    ax.set_xlabel("Employee-Income Decile")
    ax.set_ylabel("Effective Tax Rate (%)")
    ax.set_title(f"PIT Effective Tax Rate & Revenue Share by Decile — {COUNTRY}")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, pit_rate.max() * 100 * 1.3 if pit_rate.max() > 0 else 1)

    # Revenue share on a secondary axis so it does not get conflated with the rate bars.
    ax2 = ax.twinx()
    ax2.plot(
        dec_labels,
        pit_share,
        color="steelblue",
        marker="o",
        linewidth=2,
        label="Revenue share",
    )
    ax2.set_ylabel("Revenue Share (%)")
    ax2.set_ylim(0, max(pit_share) * 1.3 if pit_share.max() > 0 else 1)

    # Combined legend from both axes.
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    # Annotate revenue share values on the secondary-axis line.
    for x, share_val in zip(dec_labels, pit_share):
        if share_val > 0.5:
            ax2.text(x, share_val + 0.4, f"{share_val:.1f}%", ha="center", va="bottom", fontsize=7, color="steelblue")

plt.tight_layout()
plot_path = REPO_ROOT / "output" / f"income_distribution_{COUNTRY.lower()}.png"
plt.savefig(plot_path, dpi=150)
print(f"\nPlot saved to {plot_path}")
plt.show()