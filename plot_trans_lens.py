# import matplotlib.pyplot as plt
# from collections import defaultdict
# from pathlib import Path
# import json
#
# import numpy as np


# def plot_transcript_stats(d, path: Path):
#     """Generate and save statistics plots from the given dictionary structure"""
#
#     # 1. Flatten all lengths for histogram
#     all_lengths = []
#     for moshaf_data in d.values():
#         for lengths in moshaf_data.values():
#             all_lengths.extend(lengths)
#
#     # Plot histogram
#     plt.figure(figsize=(10, 6))
#     plt.hist(all_lengths, bins=50, color="skyblue", edgecolor="black")
#     plt.title("Distribution of Part Lengths (All Moshafs & Suras)")
#     plt.xlabel("Part Length (characters)")
#     plt.ylabel("Frequency")
#     plt.grid(axis="y", alpha=0.5)
#     plt.savefig(path / "histogram_all_lengths.png", bbox_inches="tight")
#     plt.close()
#
#     # 2. Aggregate data by sura
#     sura_stats = defaultdict(list)
#     for moshaf_data in d.values():
#         for sura_id, lengths in moshaf_data.items():
#             sura_stats[sura_id].extend(lengths)
#
#     # Prepare sura data for plotting
#     sorted_suras = sorted(sura_stats.items(), key=lambda x: int(x[0]))
#     sura_ids = [s[0] for s in sorted_suras]
#     min_vals = [np.min(s[1]) for s in sorted_suras]
#     mean_vals = [np.mean(s[1]) for s in sorted_suras]
#     max_vals = [np.max(s[1]) for s in sorted_suras]
#
#     # Plot sura statistics (wide format)
#     fig, ax = plt.subplots(figsize=(25, 8))
#     x = np.arange(len(sura_ids))
#     width = 0.25
#
#     ax.bar(x - width, min_vals, width, label="Min", color="lightcoral")
#     ax.bar(x, mean_vals, width, label="Mean", color="skyblue")
#     ax.bar(x + width, max_vals, width, label="Max", color="lightgreen")
#
#     ax.set_title("Part Length Statistics by Sura")
#     ax.set_ylabel("Length (characters)")
#     ax.set_xlabel("Sura ID")
#     ax.set_xticks(x)
#     ax.set_xticklabels(sura_ids, rotation=90)
#     ax.legend()
#     ax.grid(axis="y", alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(path / "sura_statistics.png")
#     plt.close()
#
#     # 3. Aggregate data by moshaf
#     moshaf_stats = {}
#     for moshaf_id, moshaf_data in d.items():
#         all_lengths = []
#         for lengths in moshaf_data.values():
#             all_lengths.extend(lengths)
#         moshaf_stats[moshaf_id] = {
#             "min": np.min(all_lengths),
#             "mean": np.mean(all_lengths),
#             "max": np.max(all_lengths),
#         }
#
#     # Prepare moshaf data for plotting
#     moshaf_ids = list(moshaf_stats.keys())
#     min_vals = [moshaf_stats[id]["min"] for id in moshaf_ids]
#     mean_vals = [moshaf_stats[id]["mean"] for id in moshaf_ids]
#     max_vals = [moshaf_stats[id]["max"] for id in moshaf_ids]
#
#     # Plot moshaf statistics
#     fig, ax = plt.subplots(figsize=(12, 6))
#     x = np.arange(len(moshaf_ids))
#     width = 0.25
#
#     ax.bar(x - width, min_vals, width, label="Min", color="lightcoral")
#     ax.bar(x, mean_vals, width, label="Mean", color="skyblue")
#     ax.bar(x + width, max_vals, width, label="Max", color="lightgreen")
#
#     ax.set_title("Part Length Statistics by Moshaf")
#     ax.set_ylabel("Length (characters)")
#     ax.set_xlabel("Moshaf ID")
#     ax.set_xticks(x)
#     ax.set_xticklabels(moshaf_ids)
#     ax.legend()
#     ax.grid(axis="y", alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(path / "moshaf_statistics.png")
#     plt.close()
#
#
# if __name__ == "__main__":
#     # Create output directory
#     path = Path("./assets/moshaf_transcript_stats")
#     path.mkdir(exist_ok=True, parents=True)
#     with open("./assets/trans_lens.json", "r") as f:
#         d = json.load(f)
#
#     plot_transcript_stats(d, path)


import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
from collections import defaultdict


def plot_transcript_stats(d, path: Path, unit: str):
    """Generate and save interactive statistics plots"""

    # Create output directory
    path.mkdir(exist_ok=True, parents=True)

    # 1. Flatten all lengths for histogram
    all_lengths = []
    for moshaf_data in d.values():
        for sura_id, lengths in moshaf_data.items():
            all_lengths.extend(lengths)

    # Create interactive histogram
    fig_hist = go.Figure(
        data=[
            go.Histogram(
                x=all_lengths,
                nbinsx=50,
                marker_color="skyblue",
                hovertemplate="Length: %{x}<br>Count: %{y}<extra></extra>",
            )
        ]
    )
    fig_hist.update_layout(
        title="Distribution of Part Lengths (All Moshafs & Suras)",
        xaxis_title=f"Part Length ({unit})",
        yaxis_title="Frequency",
        bargap=0.1,
        template="plotly_white",
    )
    fig_hist.write_html(path / "histogram_all_lengths.html")

    # 2. Aggregate data by sura (across all moshafs)
    sura_stats = defaultdict(list)
    for moshaf_data in d.values():
        for sura_id, lengths in moshaf_data.items():
            sura_stats[sura_id].extend(lengths)

    # Prepare sura data for plotting
    # Sort numerically by converting to int
    sorted_suras = sorted(sura_stats.items(), key=lambda x: int(x[0]))
    sura_ids = [s[0] for s in sorted_suras]
    min_vals = [np.min(s[1]) for s in sorted_suras]
    mean_vals = [np.mean(s[1]) for s in sorted_suras]
    max_vals = [np.max(s[1]) for s in sorted_suras]

    # Create interactive sura statistics plot
    fig_sura = go.Figure()

    fig_sura.add_trace(
        go.Bar(
            x=sura_ids,
            y=min_vals,
            name="Min",
            marker_color="lightcoral",
            hovertemplate="Sura: %{x}<br>Min: %{y}<extra></extra>",
        )
    )

    fig_sura.add_trace(
        go.Bar(
            x=sura_ids,
            y=mean_vals,
            name="Mean",
            marker_color="skyblue",
            hovertemplate="Sura: %{x}<br>Mean: %{y}<extra></extra>",
        )
    )

    fig_sura.add_trace(
        go.Bar(
            x=sura_ids,
            y=max_vals,
            name="Max",
            marker_color="lightgreen",
            hovertemplate="Sura: %{x}<br>Max: %{y}<extra></extra>",
        )
    )

    fig_sura.update_layout(
        title="Part Length Statistics by Sura (All Moshafs)",
        xaxis_title="Sura ID",
        yaxis_title=f"Length ({unit})",
        barmode="group",
        template="plotly_white",
        height=600,
        width=2000,  # Extra wide to accommodate all suras
    )

    # Add dropdown for better navigation
    fig_sura.update_layout(
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=1.0,
                y=1.1,
                buttons=list(
                    [
                        dict(
                            args=[{"visible": [True, True, True]}],
                            label="All",
                            method="update",
                        ),
                        dict(
                            args=[{"visible": [True, False, False]}],
                            label="Min Only",
                            method="update",
                        ),
                        dict(
                            args=[{"visible": [False, True, False]}],
                            label="Mean Only",
                            method="update",
                        ),
                        dict(
                            args=[{"visible": [False, False, True]}],
                            label="Max Only",
                            method="update",
                        ),
                    ]
                ),
            )
        ]
    )

    fig_sura.write_html(path / "sura_statistics.html")

    # 3. Aggregate data by moshaf (overall for the moshaf)
    moshaf_stats = {}
    for moshaf_id, moshaf_data in d.items():
        all_lengths = []
        for lengths in moshaf_data.values():
            all_lengths.extend(lengths)
        moshaf_stats[moshaf_id] = {
            "min": np.min(all_lengths) if all_lengths else 0,
            "mean": np.mean(all_lengths) if all_lengths else 0,
            "max": np.max(all_lengths) if all_lengths else 0,
        }

    # Prepare moshaf data for plotting
    moshaf_ids = list(moshaf_stats.keys())
    min_vals = [moshaf_stats[id]["min"] for id in moshaf_ids]
    mean_vals = [moshaf_stats[id]["mean"] for id in moshaf_ids]
    max_vals = [moshaf_stats[id]["max"] for id in moshaf_ids]

    # Create interactive moshaf statistics plot
    fig_moshaf = go.Figure()

    fig_moshaf.add_trace(
        go.Bar(
            x=moshaf_ids,
            y=min_vals,
            name="Min",
            marker_color="lightcoral",
            hovertemplate="Moshaf: %{x}<br>Min: %{y}<extra></extra>",
        )
    )

    fig_moshaf.add_trace(
        go.Bar(
            x=moshaf_ids,
            y=mean_vals,
            name="Mean",
            marker_color="skyblue",
            hovertemplate="Moshaf: %{x}<br>Mean: %{y}<extra></extra>",
        )
    )

    fig_moshaf.add_trace(
        go.Bar(
            x=moshaf_ids,
            y=max_vals,
            name="Max",
            marker_color="lightgreen",
            hovertemplate="Moshaf: %{x}<br>Max: %{y}<extra></extra>",
        )
    )

    fig_moshaf.update_layout(
        title="Part Length Statistics by Moshaf",
        xaxis_title="Moshaf ID",
        yaxis_title=f"Length ({unit})",
        barmode="group",
        template="plotly_white",
        height=600,
    )

    fig_moshaf.write_html(path / "moshaf_statistics.html")

    # 4. Create an interactive plot with dropdown for per-moshaf sura statistics
    # Get sorted sura_ids (numerically)
    # Collect all unique sura IDs and sort numerically
    all_sura_ids = set()
    for moshaf_data in d.values():
        all_sura_ids.update(moshaf_data.keys())
    sura_ids_all = sorted(all_sura_ids, key=lambda x: int(x))

    # Prepare data for each moshaf
    moshaf_data_dict = {}
    for moshaf_id, moshaf_data in d.items():
        min_vals = []
        mean_vals = []
        max_vals = []
        for sura_id in sura_ids_all:
            lengths = moshaf_data.get(sura_id, [])
            if lengths:
                min_vals.append(np.min(lengths))
                mean_vals.append(np.mean(lengths))
                max_vals.append(np.max(lengths))
            else:
                # Use NaN for missing data so it doesn't plot
                min_vals.append(float("nan"))
                mean_vals.append(float("nan"))
                max_vals.append(float("nan"))
        moshaf_data_dict[moshaf_id] = {
            "min": min_vals,
            "mean": mean_vals,
            "max": max_vals,
        }

    # Prepare aggregate data (All Moshafs) for the same suras
    agg_min_vals = []
    agg_mean_vals = []
    agg_max_vals = []
    for sura_id in sura_ids_all:
        lengths = sura_stats.get(sura_id, [])
        if lengths:
            agg_min_vals.append(np.min(lengths))
            agg_mean_vals.append(np.mean(lengths))
            agg_max_vals.append(np.max(lengths))
        else:
            agg_min_vals.append(float("nan"))
            agg_mean_vals.append(float("nan"))
            agg_max_vals.append(float("nan"))

    # Create the figure
    fig_combined = go.Figure()

    # Add traces for the first moshaf
    first_moshaf = list(d.keys())[0]
    fig_combined.add_trace(
        go.Bar(
            x=sura_ids_all,
            y=moshaf_data_dict[first_moshaf]["min"],
            name="Min",
            marker_color="lightcoral",
            hovertemplate="Sura: %{x}<br>Min: %{y}<extra></extra>",
            visible=True,
        )
    )

    fig_combined.add_trace(
        go.Bar(
            x=sura_ids_all,
            y=moshaf_data_dict[first_moshaf]["mean"],
            name="Mean",
            marker_color="skyblue",
            hovertemplate="Sura: %{x}<br>Mean: %{y}<extra></extra>",
            visible=True,
        )
    )

    fig_combined.add_trace(
        go.Bar(
            x=sura_ids_all,
            y=moshaf_data_dict[first_moshaf]["max"],
            name="Max",
            marker_color="lightgreen",
            hovertemplate="Sura: %{x}<br>Max: %{y}<extra></extra>",
            visible=True,
        )
    )

    # Create dropdown menu options
    dropdown_options = []

    # Add "All" option first
    dropdown_options.append(
        {
            "label": "All Moshafs",
            "method": "update",
            "args": [
                {
                    "y": [agg_min_vals, agg_mean_vals, agg_max_vals],
                    "name": ["Min", "Mean", "Max"],
                },
                {"title": "Part Length Statistics by Sura (All Moshafs)"},
            ],
        }
    )

    # Add options for each moshaf
    for moshaf_id in d.keys():
        dropdown_options.append(
            {
                "label": moshaf_id,
                "method": "update",
                "args": [
                    {
                        "y": [
                            moshaf_data_dict[moshaf_id]["min"],
                            moshaf_data_dict[moshaf_id]["mean"],
                            moshaf_data_dict[moshaf_id]["max"],
                        ],
                        "name": [
                            f"Min ({moshaf_id})",
                            f"Mean ({moshaf_id})",
                            f"Max ({moshaf_id})",
                        ],
                    },
                    {"title": f"Part Length Statistics by Sura - {moshaf_id}"},
                ],
            }
        )

    # Create trace visibility buttons
    visibility_buttons = [
        {"args": [{"visible": [True, True, True]}], "label": "All", "method": "update"},
        {
            "args": [{"visible": [True, False, False]}],
            "label": "Min Only",
            "method": "update",
        },
        {
            "args": [{"visible": [False, True, False]}],
            "label": "Mean Only",
            "method": "update",
        },
        {
            "args": [{"visible": [False, False, True]}],
            "label": "Max Only",
            "method": "update",
        },
    ]

    # Update layout with BOTH dropdowns
    fig_combined.update_layout(
        updatemenus=[
            # Moshaf selection dropdown
            {
                "buttons": dropdown_options,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "font": {"size": 12},
            },
            # Trace visibility dropdown
            {
                "buttons": visibility_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.3,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
            },
        ],
        title="Part Length Statistics by Sura - " + first_moshaf,
        xaxis_title="Sura ID",
        yaxis_title=f"Length ({unit})",
        barmode="group",
        template="plotly_white",
        height=600,
        width=2000,
        showlegend=True,
    )

    fig_combined.write_html(path / "sura_statistics_per_moshaf.html")


if __name__ == "__main__":
    # Create output directory
    # path = Path("./assets/moshaf_transcript_stats")
    path = Path("./assets/moshaf_audio_stats")
    with open("./assets/audio_lens.json", "r") as f:
        d = json.load(f)

    d = {
        m_id: {s_id: [val for val in d[m_id][s_id] if val != 0] for s_id in d[m_id]}
        for m_id in d
    }

    # plot_transcript_stats(d, path, unit="characters")
    plot_transcript_stats(d, path, unit="seconds")
