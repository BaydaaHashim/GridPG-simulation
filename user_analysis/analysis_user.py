"""User-supplied analysis script, adapted to the final results:
- protocol names updated to the final ones (GridPG family)
- N = 100 (main experiment) added to the density plot
Run from this folder: python analysis_user.py"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# 1. قراءة الملف وتنظيف النصوص
df = pd.read_csv(os.path.join("..", "results", "summary.csv"))
df.columns = df.columns.str.strip()
for col in ["experiment", "param", "proto"]:
    df[col] = df[col].astype(str).str.strip()

metrics = [
    "FND", "HND", "LND", "PDR", "latency_ms",
    "hops", "energy_per_reading_mJ", "delivered",
    "opt_time_ms", "gt_sweeps"
]

# 2. حفظ الجداول الإحصائية لجميع التجارب في ملف Excel واحد
with pd.ExcelWriter("all_experiments_summary.xlsx") as writer:
    main_df = df[df["experiment"] == "main"]
    main_df.groupby("proto")[metrics].mean().round(4).to_excel(writer, sheet_name="Main_Mean")
    main_df.groupby("proto")[metrics].std().round(4).to_excel(writer, sheet_name="Main_Std")
    main_df.groupby("proto")[metrics].median().round(4).to_excel(writer, sheet_name="Main_Median")
    for exp in ["density", "der", "grid", "bs_center"]:
        sub = df[df["experiment"] == exp]
        sub.groupby(["value", "proto"])[metrics[:8]].mean().round(4).to_excel(writer, sheet_name=f"{exp}_Mean")
        sub.groupby(["value", "proto"])[metrics[:8]].std().round(4).to_excel(writer, sheet_name=f"{exp}_Std")
print("تم حفظ ملخص الجداول في: all_experiments_summary.xlsx")

# 3. الرسوم البيانية للتجربة الرئيسية
sns.set_theme(style="whitegrid", font_scale=1.05)
proto_order = ["LEACH", "S-LEACH", "LEACH-C", "PSO-C",
               "GridPG-Direct", "GridPG-3x3", "GridPG-GWO", "GridPG"]

# الشكل 1: عمر الشبكة
lifetime_df = main_df.melt(id_vars="proto", value_vars=["FND", "HND", "LND"],
                           var_name="Metric", value_name="Rounds")
plt.figure(figsize=(12, 5))
sns.barplot(data=lifetime_df, x="proto", y="Rounds", hue="Metric",
            order=proto_order, errorbar="sd", capsize=0.08, palette="viridis")
plt.title("Network Lifetime Comparison (Main Experiment)", fontweight="bold")
plt.xlabel("Protocol"); plt.ylabel("Rounds"); plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig("main_lifetime_comparison.png", dpi=300); plt.close()

# الشكل 2: مؤشرات الأداء الأربعة
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
specs = [("PDR", "Packet Delivery Ratio (PDR)", "PDR"),
         ("latency_ms", "Average End-to-End Latency", "Latency (ms)"),
         ("energy_per_reading_mJ", "Energy Consumption per Reading", "Energy (mJ)"),
         ("delivered", "Total Delivered Packets", "Delivered Packets")]
for ax, (col, title, ylabel) in zip(axes.flat, specs):
    sns.barplot(data=main_df, x="proto", y=col, order=proto_order, errorbar="sd", capsize=0.08,
                palette="Set2", hue="proto", hue_order=proto_order, legend=False, ax=ax)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig("main_performance_metrics.png", dpi=300); plt.close()

# الشكل 3: قابلية التوسع (density + N=100 من التجربة الرئيسية)
density_df = df[df["experiment"] == "density"].copy()
density_df["value"] = pd.to_numeric(density_df["value"])
n100 = main_df[main_df["proto"].isin(density_df["proto"].unique())].copy()
n100["value"] = 100
density_df = pd.concat([density_df, n100])
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (col, title, ylabel) in zip(axes, [
        ("FND", "FND vs. Node Density (N)", "FND (Rounds)"),
        ("energy_per_reading_mJ", "Energy/Reading vs. Node Density (N)", "Energy (mJ)"),
        ("delivered", "Delivered Packets vs. Node Density (N)", "Delivered Packets")]):
    sns.lineplot(data=density_df, x="value", y=col, hue="proto", marker="o", linewidth=2, ax=ax)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Number of Nodes (N)"); ax.set_ylabel(ylabel)
    ax.set_xticks([50, 100, 150, 200])
handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    ax.get_legend().remove()
fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig("density_scalability.png", dpi=300); plt.close()
print("تم حفظ الأشكال الثلاثة")
