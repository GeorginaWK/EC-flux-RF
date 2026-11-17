# -*- coding: utf-8 -*-
"""
NEE Flux Analysis - plotting the full dataset predictions and SHAP values
Uses full dataset prediction file from stage 2 (NEE_HPT_Za*.py)
@author: Isabella Rosenberg Jørgensen
"""

#25/07/2026 fik jeg endelig fikset figuren ift. shap og NEE. 
#mangler stadig at få fikset month labels og submitte ændringer til Github

#opdateret 04/11/2026 til at inkludere renaming af variable

from pathlib import Path

# Set matplotlib backend BEFORE importing pyplot to avoid Tkinter threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

SITE_IDS = ["ZaF", "ZaH"]
_SITE_DISPLAY_NAMES = {"ZaF": "GL-ZaF", "ZaH": "GL-ZaH"}

# source files holding the raw (unscaled) predictor values, since the
# full-dataset export only contains SHAP values and predictions, not the
# raw predictors themselves
_RAW_DATASET_PATHS = {
	"ZaF": Path(r"C:\Users\au800367\Projects\Python\ML_dataset\dataset_complete_output\ZaF_complete.csv"),
	"ZaH": Path(r"C:\Users\au800367\Projects\Python\ML_dataset\dataset_complete_output\ZaH_complete.csv"),
}

# hardcoded per-feature colors: two shades per related group
# (darker = base variable, lighter = its lagged variant)

_FEATURE_COLORS = {
	"SWC_10cm": "#2166ac",       # dark blue
	"SWC_lag21d": "#92c5de",     # light blue
	"TS_10cm": "#b2182b",        # dark warm (red)
	"TS_lag14d": "#f4a582",      # light warm (peach)
	"DSSM": "#4d4d4d",           # dark grey
	"Snow_Cover_Percentage": "#b3b3b3",  # light grey
	"RG": "#1b7837",             # dark green
	"RG_min": "#1b7837",
	"SWC_roll60d_mean_lag60d": "#ffd92f",  # yellow
	"TS_roll60d_mean_lag60d": "#ff7f00",   # orange
	"day_length": "#a6dba0",     # light green
	"precipitation_rate": "#762a83",  # purple, not part of a paired group
}

def _feature_color_map(feature_cols):
	color_map = {}
	for col in feature_cols:
		feature = col.replace("SHAP_", "")
		color_map[col] = _FEATURE_COLORS.get(feature, "#6c6c6c")
	return color_map


def _top_shap_cols(df, shap_cols, top_n=None):
	if top_n is None:
		return list(shap_cols)

	mean_abs = df[shap_cols].abs().mean().sort_values(ascending=False)
	return list(mean_abs.index[:top_n])


# fixed display order for stacking, coloring and the legend: grouped by
# color family (grey, green, blue, warm), base variable before its lag
_FEATURE_LEGEND_ORDER = [
	"DSSM", "Snow_Cover_Percentage",
	"RG", "RG_min", "day_length",
	"precipitation_rate",
	"SWC_10cm", "SWC_lag21d",
	"TS_10cm", "TS_lag14d",
	"SWC_roll60d_mean_lag60d", "TS_roll60d_mean_lag60d"
]


def _sort_shap_cols(cols):
	def _key(col):
		feature = col.replace("SHAP_", "")
		if feature in _FEATURE_LEGEND_ORDER:
			return (0, _FEATURE_LEGEND_ORDER.index(feature))
		return (1, feature)

	return sorted(cols, key=_key)


# rename features for plot legends only
feature_rename_map = {
	"SWC_10cm": "SWC",
	"SWC_lag21d": "SWC_21d",
	"TS_10cm": "TS",
	"TS_lag14d": "TS_14d",
	"DSSM": "DASM",
	"Snow_Cover_Percentage": "Snow Cover",
	"RG": "SW_IN",
	"RG_min": "SW_IN",
	"day_length": "Day length",
	"precipitation_rate": "P",
	"SWC_roll60d_mean_lag60d": "SWC_60d",
	"TS_roll60d_mean_lag60d": "TS_60d",
}

# column groups for the "below" legend: each inner list is kept together in
# one legend column so related variables stay vertically adjacent
_LEGEND_COLUMN_GROUPS = [
	["DASM", "Snow Cover"],
	["SW_IN", "Day length"],
	["P"],
	["SWC", "SWC_21d"],
	["TS", "TS_14d"],
	["SWC_60d", "TS_60d"],
]


def _legend_label(shap_col):
	feature = shap_col.replace("SHAP_", "")
	return feature_rename_map.get(feature, feature)


def _grouped_legend_handles_labels(handle_by_label):
	"""Lay out legend entries column-major so each group in _LEGEND_COLUMN_GROUPS
	fills its own column (plus a trailing column for any extra labels)."""
	groups = [[f for f in g if f in handle_by_label] for g in _LEGEND_COLUMN_GROUPS]
	groups = [g for g in groups if g]
	grouped_labels = {label for g in groups for label in g}
	extra = [label for label in handle_by_label if label not in grouped_labels]
	groups += [[label] for label in extra]

	max_rows = max(len(g) for g in groups)
	blank_handle = plt.Line2D([0], [0], color="none")

	handles, labels = [], []
	for group in groups:
		for row in range(max_rows):
			if row < len(group):
				label = group[row]
				handles.append(handle_by_label[label])
				labels.append(label)
			else:
				handles.append(blank_handle)
				labels.append("")
	return handles, labels, len(groups)


def plot_full_dataset_observed_predicted_line(df_full, plots_dir, site_id):
	series_df = df_full[["Date", "Observed_NEE", "Predicted_NEE"]].dropna().sort_values("Date")

	fig, ax = plt.subplots(figsize=(13, 6))
	ax.plot(series_df["Date"], series_df["Observed_NEE"], color="black", linewidth=1.0, label="Observed NEE")
	ax.plot(series_df["Date"], series_df["Predicted_NEE"], color="tab:blue", linewidth=1.0, alpha=0.9, label="Predicted NEE")
	ax.set_xlabel("Date")
	ax.set_ylabel("NEE")
	ax.set_title(f"Full Dataset ({_SITE_DISPLAY_NAMES[site_id]}): Observed vs Predicted NEE")
	ax.grid(True, alpha=0.25)
	ax.legend()

	plt.tight_layout()
	fig.savefig(plots_dir / f"plot_full_dataset_observed_predicted_line_{site_id}.png", dpi=600, bbox_inches="tight")
	plt.close(fig)


def plot_weekly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, site_ids, top_n=None, legend_position="below"):
	data_by_site = {}
	all_cols = set()
	master_periods = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		tmp = df.copy()
		tmp["WeekPeriod"] = tmp["Date"].dt.to_period("W")
		weekly_shap = tmp.groupby("WeekPeriod")[shap_cols].mean().sort_index()

		nee_col = "Predicted_NEE" if "Predicted_NEE" in tmp.columns else "Observed_NEE"
		weekly_nee = tmp.groupby("WeekPeriod")[nee_col].mean().sort_index()

		valid_periods = [
			p for p in weekly_shap.index
			if p.start_time.month in [6, 7, 8, 9]
		]

		weekly_shap = weekly_shap.loc[valid_periods]
		weekly_nee = weekly_nee.loc[valid_periods]
		avg_pred_nee = tmp[nee_col].mean()
		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")

		master_periods.update(weekly_shap.index.tolist())

		data_by_site[site] = {"shap": weekly_shap, "nee": weekly_nee, "avg_pred_nee": avg_pred_nee}

	all_cols = _sort_shap_cols(all_cols)
	master_periods = sorted(master_periods)
	if not master_periods:
		raise ValueError("No weekly periods found for full-dataset continuous plot.")

	bar_step = 2.6
	bar_width = 1.95
	year_gap_days = 22  # visual gap between years, expressed as equivalent days
	day_scale = bar_step / 7.0  # x-units per calendar day (~1 bar_step per week)
	year_span_days = 122  # Jun 1 -> Oct 1, constant every year (leap days fall in Feb)

	master_periods = [
		p for p in master_periods
		if p.start_time.month in [6, 7, 8, 9]
	]

	x_index = pd.Index(master_periods)
	x_years = np.array([period.year for period in master_periods])
	x_months = np.array([period.start_time.month for period in master_periods])

	sorted_years = sorted(set(x_years.tolist()))
	year_start_x = {
		year: i * (year_span_days + year_gap_days) * day_scale
		for i, year in enumerate(sorted_years)
	}

	# position each bar by its real calendar offset from June 1st of its year
	x_vals = np.array([
		year_start_x[period.year]
		+ (period.start_time - pd.Timestamp(year=period.year, month=6, day=1)).days * day_scale
		for period in master_periods
	])

	month_tick_positions = []
	month_tick_labels = []

	month_boundaries_days = [0, 30, 61, 92, 122]  # Jun 1, Jul 1, Aug 1, Sep 1, Oct 1 relative to Jun 1
	month_names = ["Jun", "Jul", "Aug", "Sep"]

	for year in sorted_years:
		for i, month_name in enumerate(month_names):
			# center each tick within its month so the Jun/Sep edge margins are symmetric
			day_offset = 0.5 * (month_boundaries_days[i] + month_boundaries_days[i + 1])
			month_tick_positions.append(year_start_x[year] + day_offset * day_scale)
			month_tick_labels.append(month_name)

	year_boundaries = [
		year_start_x[sorted_years[index - 1]] + year_span_days * day_scale + 0.5 * year_gap_days * day_scale
		for index in range(1, len(sorted_years))
	]

	year_label_positions = [
		(year_start_x[year] + 0.5 * year_span_days * day_scale, str(year))
		for year in sorted_years
	]

	color_map = _feature_color_map(all_cols)

	# legend on the right needs extra figure width; legend below needs extra height instead
	figsize = (14, 7.8) if legend_position == "right" else (12, 7.8)
	fig, axes = plt.subplots(len(site_ids), 1, figsize=figsize, sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	edge_pad = 0.5 * year_gap_days * day_scale  # same half-gap used between years, so all margins match
	# fixed grid edges (not derived from actual bar extents) keep the Jun/Sep margins identical across years
	plot_x_min = year_start_x[sorted_years[0]] - edge_pad
	plot_x_max = year_start_x[sorted_years[-1]] + year_span_days * day_scale + edge_pad

	ax_nee_list = []
	for ax, site in zip(axes, site_ids):
		shap_vals = data_by_site[site]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(x_index).fillna(0.0)
		nee_vals = data_by_site[site]["nee"].reindex(x_index)
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]
		ax_nee = ax.twinx()
		ax_nee_list.append(ax_nee)

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			values = shap_vals[col].values
			pos = np.clip(values, 0, None)
			neg = np.clip(values, None, 0)
			ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[col])
			ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[col])
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		nee_values = nee_vals.values
		for year in sorted(set(x_years.tolist())):
			year_mask = x_years == year
			ax_nee.plot(x_vals[year_mask], nee_values[year_mask], color="black", linewidth=1.2)
		ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
		ax.set_title(_SITE_DISPLAY_NAMES[site], y=1.10, pad=1)
		ax.set_ylabel("SHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax.grid(True, axis="y", alpha=0.2)

		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		nee_clean = nee_vals.dropna()
		nee_deviation = nee_clean - avg_pred_nee

		shap_max = pos_sum.max() if len(pos_sum) else 0.0
		shap_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_dev_max = nee_deviation.max() if len(nee_deviation) else 0.0
		nee_dev_min = nee_deviation.min() if len(nee_deviation) else 0.0

		# Keep a strict linear relationship between axes:
		# Predicted NEE (secondary) = SHAP (primary) + avg_pred_nee.
		shap_upper = max(shap_max, nee_dev_max)
		shap_lower = min(shap_min, nee_dev_min)
		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		ax.set_ylim(shap_y_min, shap_y_max)

		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee
		ax_nee.set_ylim(nee_y_min, nee_y_max)

		ax.set_xlim(plot_x_min, plot_x_max)
		for boundary in year_boundaries:
			ax.axvline(x=boundary, color="black", linewidth=0.8, alpha=0.6)

	axes[-1].set_xticks(month_tick_positions)
	axes[-1].set_xticklabels(
		month_tick_labels,
		rotation=0,
		ha="center",
		fontsize=10
	)
	for x_pos, year_label in year_label_positions:
		axes[0].text(x_pos, 1.03, year_label, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

	fig.align_ylabels(axes)
	fig.align_ylabels(ax_nee_list)

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.93, 0.5),
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.03),
			ncol=legend_ncol,
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.10, 1, 0.98])

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_weekly_stacked_shap_two_sites_full_dataset_continuous{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=600, bbox_inches="tight")
	plt.close(fig)


def plot_daily_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, site_ids, top_n=None, legend_position="below"):
	data_by_site = {}
	all_cols = set()
	master_dates = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		tmp = df.copy()
		tmp["Date"] = tmp["Date"].dt.normalize()
		nee_col = "Predicted_NEE" if "Predicted_NEE" in tmp.columns else "Observed_NEE"

		daily_shap_full = tmp.groupby("Date")[shap_cols].mean().sort_index()
		daily_nee_full = tmp.groupby("Date")[nee_col].mean().sort_index()

		# 3-day centered running mean smooths daily noise while keeping daily resolution
		daily_shap = daily_shap_full.rolling(window=3, center=True, min_periods=1).mean()
		daily_nee = daily_nee_full.rolling(window=3, center=True, min_periods=1).mean()

		valid_dates = [d for d in daily_shap.index if d.month in [6, 7, 8, 9]]

		daily_shap = daily_shap.loc[valid_dates]
		daily_nee = daily_nee.loc[valid_dates]
		avg_pred_nee = tmp[nee_col].mean()
		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")

		master_dates.update(daily_shap.index.tolist())

		data_by_site[site] = {"shap": daily_shap, "nee": daily_nee, "avg_pred_nee": avg_pred_nee}

	all_cols = _sort_shap_cols(all_cols)
	master_dates = sorted(master_dates)
	if not master_dates:
		raise ValueError("No daily dates found for full-dataset continuous plot.")

	bar_width = 0.85
	year_gap_days = 22  # visual gap between years, expressed as equivalent days
	day_scale = 1.0  # x-units per calendar day
	year_span_days = 122  # Jun 1 -> Oct 1, constant every year (leap days fall in Feb)

	master_dates = [d for d in master_dates if d.month in [6, 7, 8, 9]]

	x_index = pd.Index(master_dates)
	x_years = np.array([d.year for d in master_dates])

	sorted_years = sorted(set(x_years.tolist()))
	year_start_x = {
		year: i * (year_span_days + year_gap_days) * day_scale
		for i, year in enumerate(sorted_years)
	}

	# position each bar by its real calendar offset from June 1st of its year
	x_vals = np.array([
		year_start_x[d.year]
		+ (d - pd.Timestamp(year=d.year, month=6, day=1)).days * day_scale
		for d in master_dates
	])

	doy_ticks = [175, 215, 255]  # fixed DOY labels shown on the x-axis
	doy_tick_positions = []
	doy_tick_labels = []
	for year in sorted_years:
		june1 = pd.Timestamp(year=year, month=6, day=1)
		for doy in doy_ticks:
			tick_date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
			offset = (tick_date - june1).days
			doy_tick_positions.append(year_start_x[year] + offset * day_scale)
			doy_tick_labels.append(str(doy))

	year_boundaries = [
		year_start_x[sorted_years[index - 1]] + year_span_days * day_scale + 0.5 * year_gap_days * day_scale
		for index in range(1, len(sorted_years))
	]

	year_label_positions = [
		(year_start_x[year] + 0.5 * year_span_days * day_scale, str(year))
		for year in sorted_years
	]

	color_map = _feature_color_map(all_cols)

	# legend on the right needs extra figure width; legend below needs extra height instead
	figsize = (14, 7.8) if legend_position == "right" else (12, 7.8)
	fig, axes = plt.subplots(len(site_ids), 1, figsize=figsize, sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	edge_pad = 0.5 * year_gap_days * day_scale  # same half-gap used between years, so all margins match
	# fixed grid edges (not derived from actual bar extents) keep the Jun/Sep margins identical across years
	plot_x_min = year_start_x[sorted_years[0]] - edge_pad
	plot_x_max = year_start_x[sorted_years[-1]] + year_span_days * day_scale + edge_pad

	ax_nee_list = []
	for ax, site in zip(axes, site_ids):
		shap_vals = data_by_site[site]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(x_index).fillna(0.0)
		nee_vals = data_by_site[site]["nee"].reindex(x_index)
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]
		ax_nee = ax.twinx()
		ax_nee_list.append(ax_nee)

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			values = shap_vals[col].values
			pos = np.clip(values, 0, None)
			neg = np.clip(values, None, 0)
			ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[col])
			ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[col])
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		nee_values = nee_vals.values
		for year in sorted(set(x_years.tolist())):
			year_mask = x_years == year
			ax_nee.plot(x_vals[year_mask], nee_values[year_mask], color="black", linewidth=1.2)
		ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
		ax.set_title(_SITE_DISPLAY_NAMES[site], y=1.10, pad=1)
		ax.set_ylabel("SHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax.grid(True, axis="y", alpha=0.2)

		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		nee_clean = nee_vals.dropna()
		nee_deviation = nee_clean - avg_pred_nee

		shap_max = pos_sum.max() if len(pos_sum) else 0.0
		shap_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_dev_max = nee_deviation.max() if len(nee_deviation) else 0.0
		nee_dev_min = nee_deviation.min() if len(nee_deviation) else 0.0

		# Keep a strict linear relationship between axes:
		# Predicted NEE (secondary) = SHAP (primary) + avg_pred_nee.
		shap_upper = max(shap_max, nee_dev_max)
		shap_lower = min(shap_min, nee_dev_min)
		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		ax.set_ylim(shap_y_min, shap_y_max)

		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee
		ax_nee.set_ylim(nee_y_min, nee_y_max)

		ax.set_xlim(plot_x_min, plot_x_max)
		for boundary in year_boundaries:
			ax.axvline(x=boundary, color="black", linewidth=0.8, alpha=0.6)

	axes[-1].set_xticks(doy_tick_positions)
	axes[-1].set_xticklabels(
		doy_tick_labels,
		rotation=0,
		ha="center",
		fontsize=9
	)
	axes[-1].set_xlabel("DOY")
	for x_pos, year_label in year_label_positions:
		axes[0].text(x_pos, 1.03, year_label, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

	fig.align_ylabels(axes)
	fig.align_ylabels(ax_nee_list)

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.93, 0.5),
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.03),
			ncol=legend_ncol,
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.10, 1, 0.98])

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_daily_stacked_shap_two_sites_full_dataset_continuous{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=600, bbox_inches="tight")
	plt.close(fig)


def plot_daily_stacked_bars_two_sites_selected_years_and_mean(
	base_dir, plots_dir, site_ids,
	years_to_show=(2018, 2021, 2025), doy_start=152, doy_end=274,
	top_n=None, legend_position="below",
):
	data_by_site = {}
	all_cols = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		nee_col = "Predicted_NEE" if "Predicted_NEE" in df.columns else "Observed_NEE"
		avg_pred_nee = df[nee_col].mean()

		tmp = df.copy()
		tmp["Year"] = tmp["Date"].dt.year
		tmp["DOY"] = tmp["Date"].dt.dayofyear
		# keep a 1-day margin so the 3-day running mean has real neighboring-day
		# data at the DOY range edges
		tmp = tmp[(tmp["DOY"] >= doy_start - 1) & (tmp["DOY"] <= doy_end + 1)]

		series_by_key = {}
		for year in years_to_show:
			year_tmp = tmp[tmp["Year"] == year]
			shap_year = year_tmp.groupby("DOY")[shap_cols].mean().rolling(window=3, center=True, min_periods=1).mean()
			nee_year = year_tmp.groupby("DOY")[nee_col].mean().rolling(window=3, center=True, min_periods=1).mean()
			series_by_key[year] = {
				"shap": shap_year.loc[doy_start:doy_end],
				"nee": nee_year.loc[doy_start:doy_end],
			}

		# "All years" = mean across every year in the dataset for each DOY
		mean_shap = tmp.groupby("DOY")[shap_cols].mean().rolling(window=3, center=True, min_periods=1).mean()
		mean_nee = tmp.groupby("DOY")[nee_col].mean().rolling(window=3, center=True, min_periods=1).mean()
		series_by_key["All years"] = {
			"shap": mean_shap.loc[doy_start:doy_end],
			"nee": mean_nee.loc[doy_start:doy_end],
		}

		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")
		data_by_site[site] = {"series": series_by_key, "avg_pred_nee": avg_pred_nee}

	all_cols = _sort_shap_cols(all_cols)
	doy_index = pd.Index(range(doy_start, doy_end + 1), name="DOY")
	x_vals = doy_index.to_numpy()
	bar_width = 0.85

	column_keys = list(years_to_show) + ["All years"]
	column_titles = [str(key) for key in column_keys]

	color_map = _feature_color_map(all_cols)

	figsize = (4.6 * len(column_keys) + (2.0 if legend_position == "right" else 0), 8.0)
	fig, axes = plt.subplots(len(site_ids), len(column_keys), figsize=figsize, sharex=True, sharey="row")
	axes = np.atleast_2d(axes)

	for row, site in enumerate(site_ids):
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]

		# shared y-limits across this site's row keep the SHAP<->NEE axis relationship linear and comparable
		shap_upper, shap_lower = -np.inf, np.inf
		for key in column_keys:
			shap_vals = data_by_site[site]["series"][key]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(doy_index).fillna(0.0)
			nee_vals = data_by_site[site]["series"][key]["nee"].reindex(doy_index)
			pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
			neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
			nee_deviation = nee_vals.dropna() - avg_pred_nee
			shap_upper = max(shap_upper, pos_sum.max() if len(pos_sum) else 0.0, nee_deviation.max() if len(nee_deviation) else 0.0)
			shap_lower = min(shap_lower, neg_sum.min() if len(neg_sum) else 0.0, nee_deviation.min() if len(nee_deviation) else 0.0)

		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee

		for col, key in enumerate(column_keys):
			ax = axes[row, col]
			ax_nee = ax.twinx()

			shap_vals = data_by_site[site]["series"][key]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(doy_index).fillna(0.0)
			nee_vals = data_by_site[site]["series"][key]["nee"].reindex(doy_index)

			bottom_pos = np.zeros(len(shap_vals))
			bottom_neg = np.zeros(len(shap_vals))
			for feat_col in all_cols:
				values = shap_vals[feat_col].values
				pos = np.clip(values, 0, None)
				neg = np.clip(values, None, 0)
				ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[feat_col])
				ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[feat_col])
				bottom_pos = bottom_pos + pos
				bottom_neg = bottom_neg + neg

			ax_nee.plot(x_vals, nee_vals.values, color="black", linewidth=1.2)
			ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
			ax.grid(True, axis="y", alpha=0.2)
			ax.set_ylim(shap_y_min, shap_y_max)
			ax_nee.set_ylim(nee_y_min, nee_y_max)
			ax.set_xlim(doy_start - 0.5, doy_end + 0.5)

			if row == 0:
				ax.set_title(column_titles[col], fontsize=17)
			ax.tick_params(axis="both", labelsize=15)
			ax_nee.tick_params(axis="y", labelsize=15)
			if col == 0:
				ax.set_ylabel(_SITE_DISPLAY_NAMES[site] + "\nSHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=10, fontsize=16)
			# tick_params (not set_yticklabels) because y is shared across the row
			if col != len(column_keys) - 1:
				ax_nee.tick_params(axis="y", labelright=False)
			else:
				ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", 				labelpad=12, fontsize=16)
			if col != 0:
				ax.tick_params(axis="y", labelleft=False)

	for col in range(len(column_keys)):
		axes[-1, col].set_xlabel("DOY", fontsize=16)

	fig.align_ylabels(axes[:, 0])

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.995, 0.5),
			fontsize=14,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.9, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.05),
			ncol=legend_ncol,
			fontsize=14,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.12, 1, 0.96])

	# tighten the panels after tight_layout, which would otherwise reset the spacing
	fig.subplots_adjust(wspace=0.0)

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_daily_stacked_shap_two_sites_selected_years_and_mean{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=1000, bbox_inches="tight")
	plt.close(fig)


def plot_dssm_stacked_bars_two_sites_selected_years_and_mean(
	base_dir, plots_dir, site_ids,
	years_to_show=(2018, 2021, 2025), dssm_start=-30, dssm_end=90,
	top_n=None, legend_position="below",
):
	data_by_site = {}
	all_cols = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")
		raw_path = _RAW_DATASET_PATHS[site]
		if not raw_path.exists():
			raise FileNotFoundError(f"Missing raw predictor file for DSSM: {raw_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		nee_col = "Predicted_NEE" if "Predicted_NEE" in df.columns else "Observed_NEE"
		avg_pred_nee = df[nee_col].mean()

		# the full-dataset export only has SHAP values, so bring in the raw DSSM
		# predictor values from the original source dataset, matched by date
		raw_df = pd.read_csv(raw_path, usecols=["Date", "DSSM"], parse_dates=["Date"])
		tmp = df.merge(raw_df, on="Date", how="inner")
		tmp["Year"] = tmp["Date"].dt.year
		# keep a 1-unit margin so the 3-unit running mean has real neighboring
		# data at the DSSM range edges
		tmp = tmp[(tmp["DSSM"] >= dssm_start - 1) & (tmp["DSSM"] <= dssm_end + 1)]

		series_by_key = {}
		for year in years_to_show:
			year_tmp = tmp[tmp["Year"] == year]
			shap_year = year_tmp.groupby("DSSM")[shap_cols].mean().rolling(window=3, center=True, min_periods=1).mean()
			nee_year = year_tmp.groupby("DSSM")[nee_col].mean().rolling(window=3, center=True, min_periods=1).mean()
			series_by_key[year] = {
				"shap": shap_year.loc[dssm_start:dssm_end],
				"nee": nee_year.loc[dssm_start:dssm_end],
			}

		# "All years" = mean across every year in the dataset for each DSSM value
		mean_shap = tmp.groupby("DSSM")[shap_cols].mean().rolling(window=3, center=True, min_periods=1).mean()
		mean_nee = tmp.groupby("DSSM")[nee_col].mean().rolling(window=3, center=True, min_periods=1).mean()
		series_by_key["All years"] = {
			"shap": mean_shap.loc[dssm_start:dssm_end],
			"nee": mean_nee.loc[dssm_start:dssm_end],
		}

		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")
		data_by_site[site] = {"series": series_by_key, "avg_pred_nee": avg_pred_nee}

	all_cols = _sort_shap_cols(all_cols)
	dssm_index = pd.Index(range(dssm_start, dssm_end + 1), name="DSSM")
	x_vals = dssm_index.to_numpy()
	bar_width = 0.85

	column_keys = list(years_to_show) + ["All years"]
	column_titles = [str(key) for key in column_keys]

	color_map = _feature_color_map(all_cols)

	figsize = (4.6 * len(column_keys) + (2.0 if legend_position == "right" else 0), 8.0)
	fig, axes = plt.subplots(len(site_ids), len(column_keys), figsize=figsize, sharex=True, sharey="row")
	axes = np.atleast_2d(axes)

	for row, site in enumerate(site_ids):
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]

		# shared y-limits across this site's row keep the SHAP<->NEE axis relationship linear and comparable
		shap_upper, shap_lower = -np.inf, np.inf
		for key in column_keys:
			shap_vals = data_by_site[site]["series"][key]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(dssm_index).fillna(0.0)
			nee_vals = data_by_site[site]["series"][key]["nee"].reindex(dssm_index)
			pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
			neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
			nee_deviation = nee_vals.dropna() - avg_pred_nee
			shap_upper = max(shap_upper, pos_sum.max() if len(pos_sum) else 0.0, nee_deviation.max() if len(nee_deviation) else 0.0)
			shap_lower = min(shap_lower, neg_sum.min() if len(neg_sum) else 0.0, nee_deviation.min() if len(nee_deviation) else 0.0)

		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee

		for col, key in enumerate(column_keys):
			ax = axes[row, col]
			ax_nee = ax.twinx()

			shap_vals = data_by_site[site]["series"][key]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(dssm_index).fillna(0.0)
			nee_vals = data_by_site[site]["series"][key]["nee"].reindex(dssm_index)

			bottom_pos = np.zeros(len(shap_vals))
			bottom_neg = np.zeros(len(shap_vals))
			for feat_col in all_cols:
				values = shap_vals[feat_col].values
				pos = np.clip(values, 0, None)
				neg = np.clip(values, None, 0)
				ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[feat_col])
				ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[feat_col])
				bottom_pos = bottom_pos + pos
				bottom_neg = bottom_neg + neg

			ax_nee.plot(x_vals, nee_vals.values, color="black", linewidth=1.2)
			ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
			ax.grid(True, axis="y", alpha=0.2)
			ax.set_ylim(shap_y_min, shap_y_max)
			ax_nee.set_ylim(nee_y_min, nee_y_max)
			ax.set_xlim(dssm_start - 0.5, dssm_end + 0.5)

			if row == 0:
				ax.set_title(column_titles[col], fontsize=11)
			if col == 0:
				ax.set_ylabel(site + "\nSHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=10)
			if col != len(column_keys) - 1:
				ax_nee.set_yticklabels([])
			else:
				ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
			if col != 0:
				ax.set_yticklabels([])

	for col in range(len(column_keys)):
		axes[-1, col].set_xlabel("DSSM")

	fig.align_ylabels(axes[:, 0])

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.995, 0.5),
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.9, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.05),
			ncol=legend_ncol,
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.12, 1, 0.96])

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_dssm_stacked_shap_two_sites_selected_years_and_mean{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=600, bbox_inches="tight")
	plt.close(fig)


def plot_dssm_daily_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, site_ids, dssm_start=-30, dssm_end=90, top_n=None, legend_position="below"):
	data_by_site = {}
	all_cols = set()
	master_years = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")
		raw_path = _RAW_DATASET_PATHS[site]
		if not raw_path.exists():
			raise FileNotFoundError(f"Missing raw predictor file for DSSM: {raw_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		tmp = df.copy()
		tmp["Date"] = tmp["Date"].dt.normalize()
		nee_col = "Predicted_NEE" if "Predicted_NEE" in tmp.columns else "Observed_NEE"
		avg_pred_nee = tmp[nee_col].mean()

		daily_shap_full = tmp.groupby("Date")[shap_cols].mean().sort_index()
		daily_nee_full = tmp.groupby("Date")[nee_col].mean().sort_index()

		# 3-day centered running mean smooths daily noise while keeping daily resolution
		daily_shap = daily_shap_full.rolling(window=3, center=True, min_periods=1).mean()
		daily_nee = daily_nee_full.rolling(window=3, center=True, min_periods=1).mean()

		# the full-dataset export only has SHAP values, so bring in the raw DSSM
		# predictor values from the original source dataset, matched by date
		raw_df = pd.read_csv(raw_path, usecols=["Date", "DSSM"], parse_dates=["Date"])
		raw_df["Date"] = raw_df["Date"].dt.normalize()

		daily = daily_shap.join(daily_nee.rename(nee_col)).reset_index()
		daily = daily.merge(raw_df, on="Date", how="inner")
		daily["Year"] = daily["Date"].dt.year
		daily = daily[(daily["DSSM"] >= dssm_start) & (daily["DSSM"] <= dssm_end)]
		daily = daily.sort_values(["Year", "DSSM"])
		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")

		master_years.update(daily["Year"].unique().tolist())

		data_by_site[site] = {
			"data": daily.set_index(["Year", "DSSM"]),
			"nee_col": nee_col,
			"avg_pred_nee": avg_pred_nee,
		}

	all_cols = _sort_shap_cols(all_cols)
	sorted_years = sorted(master_years)
	if not sorted_years:
		raise ValueError("No DSSM values found for full-dataset continuous plot.")

	bar_width = 1.0  # bars touch edge-to-edge (no gaps) so stacks read as a continuous band
	year_gap_days = 15  # visual gap between years, expressed as equivalent DSSM units
	day_scale = 1.0  # x-units per DSSM unit
	dssm_span = dssm_end - dssm_start

	dssm_values = list(range(dssm_start, dssm_end + 1))
	x_index = pd.MultiIndex.from_product([sorted_years, dssm_values], names=["Year", "DSSM"])

	year_start_x = {
		year: i * (dssm_span + year_gap_days) * day_scale
		for i, year in enumerate(sorted_years)
	}

	x_vals = np.array([
		year_start_x[year] + (dssm - dssm_start) * day_scale
		for year, dssm in x_index
	])
	x_years = np.array([year for year, _ in x_index])

	dssm_ticks = [-10, 30, 70]  # fixed DSSM labels shown on the x-axis

	dssm_tick_positions = []
	dssm_tick_labels = []
	for year in sorted_years:
		for dssm in dssm_ticks:
			dssm_tick_positions.append(year_start_x[year] + (dssm - dssm_start) * day_scale)
			dssm_tick_labels.append(str(dssm))

	year_boundaries = [
		year_start_x[sorted_years[index - 1]] + dssm_span * day_scale + 0.5 * year_gap_days * day_scale
		for index in range(1, len(sorted_years))
	]

	year_label_positions = [
		(year_start_x[year] + 0.5 * dssm_span * day_scale, str(year))
		for year in sorted_years
	]

	color_map = _feature_color_map(all_cols)

	# legend on the right needs extra figure width; legend below needs extra height instead
	figsize = (14, 7.8) if legend_position == "right" else (12, 7.8)
	fig, axes = plt.subplots(len(site_ids), 1, figsize=figsize, sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	edge_pad = 0.5 * year_gap_days * day_scale  # same half-gap used between years, so all margins match
	# fixed grid edges (not derived from actual bar extents) keep the margins identical across years
	plot_x_min = year_start_x[sorted_years[0]] - edge_pad
	plot_x_max = year_start_x[sorted_years[-1]] + dssm_span * day_scale + edge_pad

	ax_nee_list = []
	for ax, site in zip(axes, site_ids):
		site_data = data_by_site[site]["data"]
		nee_col = data_by_site[site]["nee_col"]
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]

		shap_vals = site_data.reindex(x_index)[all_cols].fillna(0.0)
		nee_vals = site_data.reindex(x_index)[nee_col]
		ax_nee = ax.twinx()
		ax_nee_list.append(ax_nee)

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			values = shap_vals[col].values
			pos = np.clip(values, 0, None)
			neg = np.clip(values, None, 0)
			ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[col])
			ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[col])
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		nee_values = nee_vals.values
		for year in sorted_years:
			year_mask = x_years == year
			ax_nee.plot(x_vals[year_mask], nee_values[year_mask], color="black", linewidth=1.2)
		ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
		ax.set_title(_SITE_DISPLAY_NAMES[site], y=1.10, pad=1)
		ax.set_ylabel("SHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax.grid(True, axis="y", alpha=0.2)

		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		nee_clean = nee_vals.dropna()
		nee_deviation = nee_clean - avg_pred_nee

		shap_max = pos_sum.max() if len(pos_sum) else 0.0
		shap_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_dev_max = nee_deviation.max() if len(nee_deviation) else 0.0
		nee_dev_min = nee_deviation.min() if len(nee_deviation) else 0.0

		# Keep a strict linear relationship between axes:
		# Predicted NEE (secondary) = SHAP (primary) + avg_pred_nee.
		shap_upper = max(shap_max, nee_dev_max)
		shap_lower = min(shap_min, nee_dev_min)
		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		ax.set_ylim(shap_y_min, shap_y_max)

		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee
		ax_nee.set_ylim(nee_y_min, nee_y_max)

		ax.set_xlim(plot_x_min, plot_x_max)
		for boundary in year_boundaries:
			ax.axvline(x=boundary, color="black", linewidth=0.8, alpha=0.6)

	axes[-1].set_xticks(dssm_tick_positions)
	axes[-1].set_xticklabels(
		dssm_tick_labels,
		rotation=0,
		ha="center",
		fontsize=9
	)
	axes[-1].set_xlabel("DSSM")
	for x_pos, year_label in year_label_positions:
		axes[0].text(x_pos, 1.03, year_label, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

	fig.align_ylabels(axes)
	fig.align_ylabels(ax_nee_list)

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.93, 0.5),
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.03),
			ncol=legend_ncol,
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.10, 1, 0.98])

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_dssm_daily_stacked_shap_two_sites_full_dataset_continuous{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=600, bbox_inches="tight")
	plt.close(fig)


def plot_dssm_daily_stacked_bars_two_sites_selected_years_and_mean_continuous(
	base_dir, plots_dir, site_ids,
	years_to_show=(2018, 2021, 2025), dssm_start=-30, dssm_end=90,
	top_n=None, legend_position="below",
):
	data_by_site = {}
	all_cols = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")
		raw_path = _RAW_DATASET_PATHS[site]
		if not raw_path.exists():
			raise FileNotFoundError(f"Missing raw predictor file for DSSM: {raw_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		tmp = df.copy()
		tmp["Date"] = tmp["Date"].dt.normalize()
		nee_col = "Predicted_NEE" if "Predicted_NEE" in tmp.columns else "Observed_NEE"
		avg_pred_nee = tmp[nee_col].mean()

		daily_shap_full = tmp.groupby("Date")[shap_cols].mean().sort_index()
		daily_nee_full = tmp.groupby("Date")[nee_col].mean().sort_index()

		# 3-day centered running mean smooths daily noise while keeping daily resolution
		daily_shap = daily_shap_full.rolling(window=3, center=True, min_periods=1).mean()
		daily_nee = daily_nee_full.rolling(window=3, center=True, min_periods=1).mean()

		# the full-dataset export only has SHAP values, so bring in the raw DSSM
		# predictor values from the original source dataset, matched by date
		raw_df = pd.read_csv(raw_path, usecols=["Date", "DSSM"], parse_dates=["Date"])
		raw_df["Date"] = raw_df["Date"].dt.normalize()

		daily = daily_shap.join(daily_nee.rename(nee_col)).reset_index()
		daily = daily.merge(raw_df, on="Date", how="inner")
		daily["Year"] = daily["Date"].dt.year
		daily = daily[(daily["DSSM"] >= dssm_start) & (daily["DSSM"] <= dssm_end)]

		selected = daily[daily["Year"].isin(years_to_show)].copy()
		selected = selected.set_index(["Year", "DSSM"])[shap_cols + [nee_col]]

		# "All years" = mean across every year for each DSSM value, smoothed along the DSSM axis
		mean_shap = daily.groupby("DSSM")[shap_cols].mean().rolling(window=3, center=True, min_periods=1).mean()
		mean_nee = daily.groupby("DSSM")[nee_col].mean().rolling(window=3, center=True, min_periods=1).mean()
		mean_data = mean_shap.join(mean_nee)
		mean_data = pd.concat({"All years": mean_data}, names=["Year"])

		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")

		data_by_site[site] = {
			"data": pd.concat([selected, mean_data]),
			"nee_col": nee_col,
			"avg_pred_nee": avg_pred_nee,
		}

	all_cols = _sort_shap_cols(all_cols)
	group_keys = list(years_to_show) + ["All years"]

	bar_width = 1.0  # bars touch edge-to-edge (no gaps) so stacks read as a continuous band
	year_gap_days = 15  # visual gap between groups, expressed as equivalent DSSM units
	day_scale = 1.0  # x-units per DSSM unit
	dssm_span = dssm_end - dssm_start

	dssm_values = list(range(dssm_start, dssm_end + 1))
	x_index = pd.MultiIndex.from_product([group_keys, dssm_values], names=["Year", "DSSM"])

	group_start_x = {
		key: i * (dssm_span + year_gap_days) * day_scale
		for i, key in enumerate(group_keys)
	}

	x_vals = np.array([
		group_start_x[key] + (dssm - dssm_start) * day_scale
		for key, dssm in x_index
	])
	x_groups = np.array([key for key, _ in x_index])

	dssm_ticks = [-10, 30, 70]  # fixed DSSM labels shown on the x-axis

	dssm_tick_positions = []
	dssm_tick_labels = []
	for key in group_keys:
		for dssm in dssm_ticks:
			dssm_tick_positions.append(group_start_x[key] + (dssm - dssm_start) * day_scale)
			dssm_tick_labels.append(str(dssm))

	group_boundaries = [
		group_start_x[group_keys[index - 1]] + dssm_span * day_scale + 0.5 * year_gap_days * day_scale
		for index in range(1, len(group_keys))
	]

	group_label_positions = [
		(group_start_x[key] + 0.5 * dssm_span * day_scale, str(key))
		for key in group_keys
	]

	color_map = _feature_color_map(all_cols)

	# legend on the right needs extra figure width; legend below needs extra height instead
	figsize = (14, 7.8) if legend_position == "right" else (12, 7.8)
	fig, axes = plt.subplots(len(site_ids), 1, figsize=figsize, sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	edge_pad = 0.5 * year_gap_days * day_scale  # same half-gap used between groups, so all margins match
	# fixed grid edges (not derived from actual bar extents) keep the margins identical across groups
	plot_x_min = group_start_x[group_keys[0]] - edge_pad
	plot_x_max = group_start_x[group_keys[-1]] + dssm_span * day_scale + edge_pad

	ax_nee_list = []
	for ax, site in zip(axes, site_ids):
		site_data = data_by_site[site]["data"]
		nee_col = data_by_site[site]["nee_col"]
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]

		shap_vals = site_data.reindex(x_index)[all_cols].fillna(0.0)
		nee_vals = site_data.reindex(x_index)[nee_col]
		ax_nee = ax.twinx()
		ax_nee_list.append(ax_nee)

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			values = shap_vals[col].values
			pos = np.clip(values, 0, None)
			neg = np.clip(values, None, 0)
			ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[col])
			ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[col])
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		nee_values = nee_vals.values
		for key in group_keys:
			group_mask = x_groups == key
			ax_nee.plot(x_vals[group_mask], nee_values[group_mask], color="black", linewidth=1.2)
		ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
		ax.set_title(_SITE_DISPLAY_NAMES[site], y=1.10, pad=1)
		ax.set_ylabel("SHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax.grid(True, axis="y", alpha=0.2)

		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		nee_clean = nee_vals.dropna()
		nee_deviation = nee_clean - avg_pred_nee

		shap_max = pos_sum.max() if len(pos_sum) else 0.0
		shap_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_dev_max = nee_deviation.max() if len(nee_deviation) else 0.0
		nee_dev_min = nee_deviation.min() if len(nee_deviation) else 0.0

		# Keep a strict linear relationship between axes:
		# Predicted NEE (secondary) = SHAP (primary) + avg_pred_nee.
		shap_upper = max(shap_max, nee_dev_max)
		shap_lower = min(shap_min, nee_dev_min)
		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		ax.set_ylim(shap_y_min, shap_y_max)

		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee
		ax_nee.set_ylim(nee_y_min, nee_y_max)

		ax.set_xlim(plot_x_min, plot_x_max)
		for boundary in group_boundaries:
			ax.axvline(x=boundary, color="black", linewidth=0.8, alpha=0.6)

	axes[-1].set_xticks(dssm_tick_positions)
	axes[-1].set_xticklabels(
		dssm_tick_labels,
		rotation=0,
		ha="center",
		fontsize=9
	)
	axes[-1].set_xlabel("DSSM")
	for x_pos, group_label in group_label_positions:
		axes[0].text(x_pos, 1.03, group_label, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

	fig.align_ylabels(axes)
	fig.align_ylabels(ax_nee_list)

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.93, 0.5),
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.03),
			ncol=legend_ncol,
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.10, 1, 0.98])

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_dssm_daily_stacked_shap_two_sites_selected_years_and_mean_continuous{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=600, bbox_inches="tight")
	plt.close(fig)


def plot_monthly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, site_ids, top_n=None, legend_position="below"):
	data_by_site = {}
	all_cols = set()
	master_periods = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		tmp = df.copy()
		tmp["MonthPeriod"] = tmp["Date"].dt.to_period("M")
		monthly_shap = tmp.groupby("MonthPeriod")[shap_cols].mean().sort_index()

		nee_col = "Predicted_NEE" if "Predicted_NEE" in tmp.columns else "Observed_NEE"
		monthly_nee = tmp.groupby("MonthPeriod")[nee_col].mean().sort_index()

		valid_periods = [
			p for p in monthly_shap.index
			if p.start_time.month in [6, 7, 8, 9]
		]

		monthly_shap = monthly_shap.loc[valid_periods]
		monthly_nee = monthly_nee.loc[valid_periods]
		avg_pred_nee = tmp[nee_col].mean()
		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")

		master_periods.update(monthly_shap.index.tolist())

		data_by_site[site] = {"shap": monthly_shap, "nee": monthly_nee, "avg_pred_nee": avg_pred_nee}

	all_cols = _sort_shap_cols(all_cols)
	master_periods = sorted(master_periods)
	if not master_periods:
		raise ValueError("No monthly periods found for full-dataset continuous plot.")

	bar_width = 26.0
	year_gap_days = 22  # visual gap between years, expressed as equivalent days
	day_scale = 1.0  # x-units per calendar day
	year_span_days = 122  # Jun 1 -> Oct 1, constant every year (leap days fall in Feb)

	master_periods = [
		p for p in master_periods
		if p.start_time.month in [6, 7, 8, 9]
	]

	x_index = pd.Index(master_periods)
	x_years = np.array([period.year for period in master_periods])

	sorted_years = sorted(set(x_years.tolist()))
	year_start_x = {
		year: i * (year_span_days + year_gap_days) * day_scale
		for i, year in enumerate(sorted_years)
	}

	month_boundaries_days = [0, 30, 61, 92, 122]  # Jun 1, Jul 1, Aug 1, Sep 1, Oct 1 relative to Jun 1
	month_names = ["Jun", "Jul", "Aug", "Sep"]
	month_center_days = {
		m + 6: 0.5 * (month_boundaries_days[m] + month_boundaries_days[m + 1])
		for m in range(len(month_names))
	}

	# position each bar at the center of its calendar month
	x_vals = np.array([
		year_start_x[period.year] + month_center_days[period.start_time.month] * day_scale
		for period in master_periods
	])

	month_tick_positions = []
	month_tick_labels = []

	for year in sorted_years:
		for i, month_name in enumerate(month_names):
			day_offset = 0.5 * (month_boundaries_days[i] + month_boundaries_days[i + 1])
			month_tick_positions.append(year_start_x[year] + day_offset * day_scale)
			month_tick_labels.append(month_name)

	year_boundaries = [
		year_start_x[sorted_years[index - 1]] + year_span_days * day_scale + 0.5 * year_gap_days * day_scale
		for index in range(1, len(sorted_years))
	]

	year_label_positions = [
		(year_start_x[year] + 0.5 * year_span_days * day_scale, str(year))
		for year in sorted_years
	]

	color_map = _feature_color_map(all_cols)

	# legend on the right needs extra figure width; legend below needs extra height instead
	figsize = (14, 7.8) if legend_position == "right" else (12, 7.8)
	fig, axes = plt.subplots(len(site_ids), 1, figsize=figsize, sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	edge_pad = 0.5 * year_gap_days * day_scale  # same half-gap used between years, so all margins match
	# fixed grid edges (not derived from actual bar extents) keep the Jun/Sep margins identical across years
	plot_x_min = year_start_x[sorted_years[0]] - edge_pad
	plot_x_max = year_start_x[sorted_years[-1]] + year_span_days * day_scale + edge_pad

	ax_nee_list = []
	for ax, site in zip(axes, site_ids):
		shap_vals = data_by_site[site]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(x_index).fillna(0.0)
		nee_vals = data_by_site[site]["nee"].reindex(x_index)
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]
		ax_nee = ax.twinx()
		ax_nee_list.append(ax_nee)

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			values = shap_vals[col].values
			pos = np.clip(values, 0, None)
			neg = np.clip(values, None, 0)
			ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[col])
			ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[col])
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		nee_values = nee_vals.values
		for year in sorted(set(x_years.tolist())):
			year_mask = x_years == year
			ax_nee.plot(x_vals[year_mask], nee_values[year_mask], color="black", linewidth=1.2, marker="o", markersize=4)
		ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
		ax.set_title(_SITE_DISPLAY_NAMES[site], y=1.10, pad=1)
		ax.set_ylabel("SHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax.grid(True, axis="y", alpha=0.2)

		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		nee_clean = nee_vals.dropna()
		nee_deviation = nee_clean - avg_pred_nee

		shap_max = pos_sum.max() if len(pos_sum) else 0.0
		shap_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_dev_max = nee_deviation.max() if len(nee_deviation) else 0.0
		nee_dev_min = nee_deviation.min() if len(nee_deviation) else 0.0

		# Keep a strict linear relationship between axes:
		# Predicted NEE (secondary) = SHAP (primary) + avg_pred_nee.
		shap_upper = max(shap_max, nee_dev_max)
		shap_lower = min(shap_min, nee_dev_min)
		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		ax.set_ylim(shap_y_min, shap_y_max)

		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee
		ax_nee.set_ylim(nee_y_min, nee_y_max)

		ax.set_xlim(plot_x_min, plot_x_max)
		for boundary in year_boundaries:
			ax.axvline(x=boundary, color="black", linewidth=0.8, alpha=0.6)

	axes[-1].set_xticks(month_tick_positions)
	axes[-1].set_xticklabels(
		month_tick_labels,
		rotation=0,
		ha="center",
		fontsize=10
	)
	for x_pos, year_label in year_label_positions:
		axes[0].text(x_pos, 1.03, year_label, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)

	fig.align_ylabels(axes)
	fig.align_ylabels(ax_nee_list)

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.93, 0.5),
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.03),
			ncol=legend_ncol,
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.10, 1, 0.98])

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_monthly_stacked_shap_two_sites_full_dataset_continuous{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=600, bbox_inches="tight")
	plt.close(fig)


def plot_doy_stacked_bars_two_sites_mean_across_years(base_dir, plots_dir, site_ids, doy_start=160, doy_end=260, top_n=None, legend_position="below"):
	data_by_site = {}
	all_cols = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		nee_col = "Predicted_NEE" if "Predicted_NEE" in df.columns else "Observed_NEE"
		avg_pred_nee = df[nee_col].mean()

		tmp = df.copy()
		tmp["DOY"] = tmp["Date"].dt.dayofyear
		# keep a 1-day margin on each side so the 3-day running mean has real
		# neighboring-day data at the DOY range edges
		tmp = tmp[(tmp["DOY"] >= doy_start - 1) & (tmp["DOY"] <= doy_end + 1)]

		# mean across all years for each day-of-year, then smoothed with a 3-day running mean
		doy_shap = tmp.groupby("DOY")[shap_cols].mean().rolling(window=3, center=True, min_periods=1).mean()
		doy_nee = tmp.groupby("DOY")[nee_col].mean().rolling(window=3, center=True, min_periods=1).mean()
		doy_shap = doy_shap.loc[doy_start:doy_end]
		doy_nee = doy_nee.loc[doy_start:doy_end]
		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")

		data_by_site[site] = {"shap": doy_shap, "nee": doy_nee, "avg_pred_nee": avg_pred_nee}

	all_cols = _sort_shap_cols(all_cols)
	doy_index = pd.Index(range(doy_start, doy_end + 1), name="DOY")
	x_vals = doy_index.to_numpy()
	bar_width = 0.85

	color_map = _feature_color_map(all_cols)

	# legend on the right needs extra figure width; legend below needs extra height instead
	figsize = (14, 7.8) if legend_position == "right" else (12, 7.8)
	fig, axes = plt.subplots(len(site_ids), 1, figsize=figsize, sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	ax_nee_list = []
	for ax, site in zip(axes, site_ids):
		shap_vals = data_by_site[site]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(doy_index).fillna(0.0)
		nee_vals = data_by_site[site]["nee"].reindex(doy_index)
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]
		ax_nee = ax.twinx()
		ax_nee_list.append(ax_nee)

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			values = shap_vals[col].values
			pos = np.clip(values, 0, None)
			neg = np.clip(values, None, 0)
			ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[col])
			ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[col])
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		ax_nee.plot(x_vals, nee_vals.values, color="black", linewidth=1.2)
		ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
		ax.set_title(_SITE_DISPLAY_NAMES[site], y=1.05, pad=1)
		ax.set_ylabel("SHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax.grid(True, axis="y", alpha=0.2)

		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		nee_clean = nee_vals.dropna()
		nee_deviation = nee_clean - avg_pred_nee

		shap_max = pos_sum.max() if len(pos_sum) else 0.0
		shap_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_dev_max = nee_deviation.max() if len(nee_deviation) else 0.0
		nee_dev_min = nee_deviation.min() if len(nee_deviation) else 0.0

		# Keep a strict linear relationship between axes:
		# Predicted NEE (secondary) = SHAP (primary) + avg_pred_nee.
		shap_upper = max(shap_max, nee_dev_max)
		shap_lower = min(shap_min, nee_dev_min)
		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		ax.set_ylim(shap_y_min, shap_y_max)

		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee
		ax_nee.set_ylim(nee_y_min, nee_y_max)

		ax.set_xlim(doy_start - 0.5, doy_end + 0.5)

	axes[-1].set_xlabel("DOY")
	axes[-1].set_xticks(np.arange(doy_start, doy_end + 1, 5))

	fig.align_ylabels(axes)
	fig.align_ylabels(ax_nee_list)

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.93, 0.5),
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.03),
			ncol=legend_ncol,
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.10, 1, 0.98])

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_doy_stacked_shap_two_sites_mean_across_years{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=600, bbox_inches="tight")
	plt.close(fig)


_MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def plot_month_stacked_bars_two_sites_mean_across_years(base_dir, plots_dir, site_ids, doy_start=150, doy_end=275, top_n=None, legend_position="below"):
	data_by_site = {}
	all_cols = set()
	all_months = set()

	for site in site_ids:
		full_path = base_dir / "output" / site / f"NEE_HPT_{site}" / "full_dataset_predictions_shap_values.csv"
		if not full_path.exists():
			raise FileNotFoundError(f"Missing full-dataset file: {full_path}")

		df = pd.read_csv(full_path, parse_dates=["Date"])
		shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
		shap_cols = _top_shap_cols(df, shap_cols, top_n=top_n)
		all_cols.update(shap_cols)

		nee_col = "Predicted_NEE" if "Predicted_NEE" in df.columns else "Observed_NEE"
		avg_pred_nee = df[nee_col].mean()

		tmp = df.copy()
		tmp["DOY"] = tmp["Date"].dt.dayofyear
		tmp = tmp[(tmp["DOY"] >= doy_start) & (tmp["DOY"] <= doy_end)]
		tmp["Month"] = tmp["Date"].dt.month

		# mean across all years and all days within the window for each calendar month
		month_shap = tmp.groupby("Month")[shap_cols].mean()
		month_nee = tmp.groupby("Month")[nee_col].mean()
		print(f"[{site}] Average predicted NEE ({nee_col}): {avg_pred_nee:.6f}")

		all_months.update(month_shap.index.tolist())
		data_by_site[site] = {"shap": month_shap, "nee": month_nee, "avg_pred_nee": avg_pred_nee}

	all_cols = _sort_shap_cols(all_cols)
	month_index = pd.Index(sorted(all_months), name="Month")
	x_vals = np.arange(len(month_index))
	bar_width = 0.7

	color_map = _feature_color_map(all_cols)

	# legend on the right needs extra figure width; legend below needs extra height instead
	figsize = (14, 7.8) if legend_position == "right" else (12, 7.8)
	fig, axes = plt.subplots(len(site_ids), 1, figsize=figsize, sharex=True, sharey=False)
	if len(site_ids) == 1:
		axes = [axes]

	ax_nee_list = []
	for ax, site in zip(axes, site_ids):
		shap_vals = data_by_site[site]["shap"].reindex(columns=all_cols).fillna(0.0).reindex(month_index).fillna(0.0)
		nee_vals = data_by_site[site]["nee"].reindex(month_index)
		avg_pred_nee = data_by_site[site]["avg_pred_nee"]
		ax_nee = ax.twinx()
		ax_nee_list.append(ax_nee)

		bottom_pos = np.zeros(len(shap_vals))
		bottom_neg = np.zeros(len(shap_vals))
		for col in all_cols:
			values = shap_vals[col].values
			pos = np.clip(values, 0, None)
			neg = np.clip(values, None, 0)
			ax.bar(x_vals, pos, bottom=bottom_pos, width=bar_width, color=color_map[col])
			ax.bar(x_vals, neg, bottom=bottom_neg, width=bar_width, color=color_map[col])
			bottom_pos = bottom_pos + pos
			bottom_neg = bottom_neg + neg

		ax_nee.plot(x_vals, nee_vals.values, color="black", linewidth=1.2, marker="o", markersize=4)
		ax.axhline(y=0, color="gray", linewidth=0.9, alpha=0.7, linestyle="--")
		ax.set_title(_SITE_DISPLAY_NAMES[site], y=1.05, pad=1)
		ax.set_ylabel("SHAP value (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax_nee.set_ylabel("Predicted NEE (g C m$^{-2}$ day$^{-1}$)", labelpad=12)
		ax.grid(True, axis="y", alpha=0.2)

		pos_sum = np.clip(shap_vals.values, 0, None).sum(axis=1)
		neg_sum = np.clip(shap_vals.values, None, 0).sum(axis=1)
		nee_clean = nee_vals.dropna()
		nee_deviation = nee_clean - avg_pred_nee

		shap_max = pos_sum.max() if len(pos_sum) else 0.0
		shap_min = neg_sum.min() if len(neg_sum) else 0.0
		nee_dev_max = nee_deviation.max() if len(nee_deviation) else 0.0
		nee_dev_min = nee_deviation.min() if len(nee_deviation) else 0.0

		# Keep a strict linear relationship between axes:
		# Predicted NEE (secondary) = SHAP (primary) + avg_pred_nee.
		shap_upper = max(shap_max, nee_dev_max)
		shap_lower = min(shap_min, nee_dev_min)
		shap_span = shap_upper - shap_lower
		if np.isclose(shap_span, 0.0):
			shap_span = 1.0
		shap_padding = 0.06 * shap_span
		shap_y_min = shap_lower - shap_padding
		shap_y_max = shap_upper + shap_padding
		ax.set_ylim(shap_y_min, shap_y_max)

		nee_y_min = shap_y_min + avg_pred_nee
		nee_y_max = shap_y_max + avg_pred_nee
		ax_nee.set_ylim(nee_y_min, nee_y_max)

		ax.set_xlim(x_vals[0] - 0.5, x_vals[-1] + 0.5)

	axes[-1].set_xlabel("Month")
	axes[-1].set_xticks(x_vals)
	axes[-1].set_xticklabels([_MONTH_NAMES[m] for m in month_index])

	fig.align_ylabels(axes)
	fig.align_ylabels(ax_nee_list)

	legend_handles = [plt.Line2D([0], [0], color=color_map[col], lw=6) for col in all_cols]
	legend_labels = [_legend_label(col) for col in all_cols]
	legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.5))
	legend_labels.append("Predicted NEE")

	if legend_position == "right":
		fig.legend(
			legend_handles, legend_labels,
			loc="center left",
			bbox_to_anchor=(0.93, 0.5),
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0, 0.92, 0.96])
	else:
		handle_by_label = dict(zip(legend_labels, legend_handles))
		grouped_handles, grouped_labels, legend_ncol = _grouped_legend_handles_labels(handle_by_label)
		fig.legend(
			grouped_handles, grouped_labels,
			loc="lower center",
			bbox_to_anchor=(0.5, -0.03),
			ncol=legend_ncol,
			fontsize=8,
			frameon=False,
		)
		plt.tight_layout(rect=[0, 0.10, 1, 0.98])

	suffix = "" if top_n is None else f"_top{top_n}"
	suffix += "" if legend_position == "below" else f"_legend_{legend_position}"
	filename = f"plot_month_stacked_shap_two_sites_mean_across_years{suffix}.png"
	fig.savefig(plots_dir / filename, dpi=600, bbox_inches="tight")
	plt.close(fig)


def main():
	base_dir = Path(__file__).resolve().parent
	plots_dir = base_dir / "plots"
	plots_dir.mkdir(parents=True, exist_ok=True)

	for site_id in SITE_IDS:
		full_dataset_path = base_dir / "output" / site_id / f"NEE_HPT_{site_id}" / "full_dataset_predictions_shap_values.csv"
		if not full_dataset_path.exists():
			raise FileNotFoundError(f"Missing full-dataset SHAP + prediction file: {full_dataset_path}")
		df_full = pd.read_csv(full_dataset_path, parse_dates=["Date"])
		plot_full_dataset_observed_predicted_line(df_full, plots_dir, site_id)

	plot_weekly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_weekly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, legend_position="right")
	plot_daily_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_daily_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, legend_position="right")
	plot_daily_stacked_bars_two_sites_selected_years_and_mean(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_daily_stacked_bars_two_sites_selected_years_and_mean(base_dir, plots_dir, SITE_IDS, legend_position="right")
	plot_dssm_stacked_bars_two_sites_selected_years_and_mean(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_dssm_stacked_bars_two_sites_selected_years_and_mean(base_dir, plots_dir, SITE_IDS, legend_position="right")
	plot_dssm_daily_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_dssm_daily_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, legend_position="right")
	plot_dssm_daily_stacked_bars_two_sites_selected_years_and_mean_continuous(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_dssm_daily_stacked_bars_two_sites_selected_years_and_mean_continuous(base_dir, plots_dir, SITE_IDS, legend_position="right")
	plot_monthly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_monthly_stacked_bars_two_sites_full_dataset_continuous(base_dir, plots_dir, SITE_IDS, legend_position="right")
	plot_doy_stacked_bars_two_sites_mean_across_years(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_doy_stacked_bars_two_sites_mean_across_years(base_dir, plots_dir, SITE_IDS, legend_position="right")
	plot_month_stacked_bars_two_sites_mean_across_years(base_dir, plots_dir, SITE_IDS, legend_position="below")
	plot_month_stacked_bars_two_sites_mean_across_years(base_dir, plots_dir, SITE_IDS, legend_position="right")


if __name__ == "__main__":
	main()