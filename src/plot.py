from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILE_PATH = Path("Sensor_Corruption.xlsx")
OUTPUT_DIR = Path("plots")
MODELS = ["KNN", "Logistic", "SVM", "LSTM"]
METRICS = ["Accuracy", "Macro-F1"]


def safe_filename(name):
    name = name.replace("<", "_lt_").replace(">", "_gt_")
    return re.sub(r'[:"/\\|?*]', "_", name)


def save_figure(fig, name):
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = safe_filename(name.replace(" ", "_"))
    fig.savefig(OUTPUT_DIR / f"{filename}.png", dpi=300)
    plt.close(fig)


def parse_baseline(sheet_df):
    cols = list(sheet_df.columns)
    cols[0] = ("Condition", "")
    sheet_df.columns = pd.MultiIndex.from_tuples(cols)

    records = []
    for _, row in sheet_df.iterrows():
        condition = str(row[("Condition", "")])
        for model in MODELS:
            for metric in METRICS:
                val = row.get((model, metric))
                if pd.notna(val):
                    records.append(
                        {
                            "Condition": condition,
                            "Model": model,
                            "Metric": metric,
                            "Score": float(val),
                        }
                    )
    return pd.DataFrame(records)


def parse_corruption(sheet_name, sheet_df):
    cols = list(sheet_df.columns)
    cols[0] = ("Severity", "")
    sheet_df.columns = pd.MultiIndex.from_tuples(cols)

    records = []
    for _, row in sheet_df.iterrows():
        severity = pd.to_numeric(row[("Severity", "")], errors="coerce")
        for model in MODELS:
            for subcol in ["GYRO", "GYRO.1", "ACCL", "ACCL.1"]:
                val = row.get((model, subcol))
                val_num = pd.to_numeric(val, errors="coerce")
                if pd.isna(val_num):
                    continue

                sensor = subcol.replace(".1", "")
                metric = "Macro-F1" if subcol.endswith(".1") else "Accuracy"
                records.append(
                    {
                        "Corruption": sheet_name,
                        "Severity": float(severity) if pd.notna(severity) else np.nan,
                        "Sensor": sensor,
                        "Model": model,
                        "Metric": metric,
                        "Score": float(val_num),
                    }
                )
    return pd.DataFrame(records)


def plot_baseline_comparison(baseline_df):
    for metric in METRICS:
        subset = baseline_df[baseline_df["Metric"] == metric]
        if subset.empty:
            continue

        pivot = subset.pivot(index="Model", columns="Condition", values="Score")
        pivot = pivot.reindex(MODELS)
        conditions = list(pivot.columns)
        x = np.arange(len(pivot.index))
        width = 0.8 / max(len(conditions), 1)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, condition in enumerate(conditions):
            ax.bar(x + i * width, pivot[condition], width=width, label=condition)

        ax.set_xticks(x + width * (len(conditions) - 1) / 2)
        ax.set_xticklabels(pivot.index)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric)
        ax.set_title(f"Baseline {metric}: Raw vs Processed")
        ax.legend(title="Condition")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        save_figure(fig, f"baseline_{metric}_by_model")


def plot_severity_trends(corruption_df):
    for corruption in sorted(corruption_df["Corruption"].unique()):
        cdata = corruption_df[corruption_df["Corruption"] == corruption]
        for metric in METRICS:
            mdata = cdata[cdata["Metric"] == metric]
            if mdata.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
            sensors = ["GYRO", "ACCL"]

            for ax, sensor in zip(axes, sensors):
                sdata = mdata[mdata["Sensor"] == sensor]
                for model in MODELS:
                    line = sdata[sdata["Model"] == model].sort_values("Severity")
                    if line.empty:
                        continue
                    ax.plot(
                        line["Severity"],
                        line["Score"],
                        marker="o",
                        linewidth=2,
                        label=model,
                    )

                ax.set_title(sensor)
                ax.set_xlabel("Severity")
                ax.grid(alpha=0.25)

            axes[0].set_ylabel(metric)
            axes[0].set_ylim(0.0, 1.0)
            handles, labels = axes[1].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.965),
                    ncol=4,
                    frameon=False,
                    columnspacing=1.6,
                    handlelength=2.2,
                    handletextpad=0.6,
                    borderaxespad=0.4,
                )

            fig.suptitle(f"{corruption}: {metric} vs Severity")
            fig.tight_layout(rect=[0, 0, 1, 0.9])
            save_figure(fig, f"trend_{corruption}_{metric}")


def plot_mean_heatmaps(corruption_df):
    for sensor in ["GYRO", "ACCL"]:
        for metric in METRICS:
            subset = corruption_df[
                (corruption_df["Sensor"] == sensor) & (corruption_df["Metric"] == metric)
            ]
            if subset.empty:
                continue

            summary = (
                subset.groupby(["Corruption", "Model"], as_index=False)["Score"]
                .mean()
                .pivot(index="Model", columns="Corruption", values="Score")
            )
            summary = summary.reindex(MODELS)

            fig, ax = plt.subplots(figsize=(10, 4.5))
            im = ax.imshow(summary.values, aspect="auto", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(summary.columns)))
            ax.set_xticklabels(summary.columns, rotation=40, ha="right")
            ax.set_yticks(range(len(summary.index)))
            ax.set_yticklabels(summary.index)
            ax.set_title(f"Mean {metric} by Corruption ({sensor})")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Score")
            fig.tight_layout()

            save_figure(fig, f"mean_heatmap_{sensor}_{metric}")


def main():
    xls = pd.ExcelFile(FILE_PATH)
    baseline_df = pd.DataFrame()
    corruption_frames = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(FILE_PATH, sheet_name=sheet, header=[0, 1])
        if sheet.lower() == "baseline":
            baseline_df = parse_baseline(df)
        else:
            corruption_frames.append(parse_corruption(sheet, df))

    corruption_df = pd.concat(corruption_frames, ignore_index=True)

    plot_baseline_comparison(baseline_df)
    plot_severity_trends(corruption_df)
    plot_mean_heatmaps(corruption_df)

    print(f"Plots written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()