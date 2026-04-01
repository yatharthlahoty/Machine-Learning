# ============================================================
# Question 1
# Student Performance Prediction using Linear Regression
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── 1. Generate synthetic student dataset ──────────────────
np.random.seed(42)
n = 200

study_hours      = np.random.uniform(1, 10, n)
attendance       = np.random.uniform(50, 100, n)
previous_scores  = np.random.uniform(40, 100, n)
sleep_hours      = np.random.uniform(4, 9, n)
extracurricular  = np.random.randint(0, 2, n)          # 0 or 1

# Target: final score (linear combination + noise)
final_score = (
    3.5 * study_hours
    + 0.3 * attendance
    + 0.4 * previous_scores
    + 1.2 * sleep_hours
    + 2.0 * extracurricular
    + np.random.normal(0, 5, n)
)
final_score = np.clip(final_score, 0, 100)

df = pd.DataFrame({
    "Study_Hours":      study_hours,
    "Attendance":       attendance,
    "Previous_Scores":  previous_scores,
    "Sleep_Hours":      sleep_hours,
    "Extracurricular":  extracurricular,
    "Final_Score":      final_score,
})

print("=" * 55)
print("  STUDENT PERFORMANCE PREDICTION - Linear Regression")
print("=" * 55)
print("\n── Dataset Overview ──")
print(df.describe().round(2))

# ── 2. Prepare features & target ───────────────────────────
X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 3. Train model ─────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── 4. Evaluate ────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n── Model Evaluation Metrics ──")
print(f"  Mean Absolute Error  (MAE)  : {mae:.4f}")
print(f"  Mean Squared Error   (MSE)  : {mse:.4f}")
print(f"  Root MSE             (RMSE) : {rmse:.4f}")
print(f"  R² Score                    : {r2:.4f}")

print("\n── Feature Coefficients ──")
for feat, coef in zip(df.columns[:-1], model.coef_):
    print(f"  {feat:<20}: {coef:.4f}")
print(f"  {'Intercept':<20}: {model.intercept_:.4f}")

# ── 5. Visualizations ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Student Performance Prediction - Linear Regression", fontsize=14)

# Plot 1 – Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="k", linewidths=0.3)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[0].set_xlabel("Actual Score")
axes[0].set_ylabel("Predicted Score")
axes[0].set_title("Actual vs Predicted")

# Plot 2 – Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color="coral", edgecolors="k", linewidths=0.3)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set_xlabel("Predicted Score")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")

# Plot 3 – Feature Coefficients
feat_names = list(df.columns[:-1])
axes[2].barh(feat_names, model.coef_, color=["steelblue" if c > 0 else "coral" for c in model.coef_])
axes[2].axvline(0, color="black", linestyle="--")
axes[2].set_title("Feature Coefficients")
axes[2].set_xlabel("Coefficient Value")

plt.tight_layout()
plt.savefig("student_performance_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("=" * 55)
