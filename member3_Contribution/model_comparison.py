
import pandas as pd
import matplotlib.pyplot as plt

# Model comparison data
data = {
    "Model": [
        "Gradient Boosting",
        "XGBoost (Calibrated)"
    ],
    "ROC_AUC": [0.8684, 0.8675],
    "PR_AUC": [0.4034, 0.4028]
}

df = pd.DataFrame(data)

# Save comparison table
df.to_csv("model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(df.to_string(index=False))

# Create comparison graph
ax = df.set_index("Model").plot(
    kind="bar",
    figsize=(9, 5),
    rot=0
)

plt.title("Gradient Boosting vs XGBoost")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig("model_comparison.png", dpi=300)
plt.close()

print("\nModel comparison completed successfully!")
