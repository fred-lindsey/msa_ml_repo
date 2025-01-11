
# %% [markdown]
# #### HW3
# - variable selection, with regularized regression setting in CPH
# - checking assumptions: linearity (martingale resids), schoenfeld for PH, 
# - then create the long dataset
# - then check the p-value for the new consechours variable
# - hour level: counting process (to remove immortal bias)
# - CPH on the hourly
# 
# - filter for failure type = 2
# - use reason = 2 as my event column, don't chop the dataset

# %%
import pandas as pd

# %% [markdown]
# Get Data - base df, to check for variable significance

# %%
import pandas as pd
from lifelines import CoxPHFitter

# Load your dataset (replace with your data)
# Assume `df` is a DataFrame with time, event, and covariates
# 'time' = duration, 'event' = binary outcome (1 = event, 0 = censored)
df = pd.read_csv("hurricane.csv")


# %%
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# Load the data
df = pd.read_csv("hurricane.csv")

# Remove columns matching h1 to h48, and other specified columns
cols_to_remove = [f"h{i}" for i in range(1, 49)] + ["reason2", "survive", "trashrack"]
df2 = df.drop(columns=[col for col in cols_to_remove if col in df.columns])


# %%
# Create a new binary column where 1 indicates reason == 2
df2['reason_binary'] = (df2['reason'] == 2).astype(int)

# Cox Proportional Hazards Model - Backward selection (approximation in Python)
cph = CoxPHFitter()
cph.fit(df2, duration_col="hour", event_col="reason_binary")

# %%

# For backward selection, there isn't an automatic backward selection in Python like R.
# It must be implemented manually or use a library like `statsmodels` for stepwise regression.
# Here, assume we manually identify variables to keep.

# Variables to keep based on backward selection
selected_vars = ["servo", "age", "slope", "backup"]

# Check linearity
# For linearity checks, you'd need to visualize residuals or use other statistical methods.
# CoxPHFitter doesn't have a direct ggcoxfunctional equivalent, so you would create custom visualizations.

# Transform `slope` and `age` into categories based on knots
df3 = df2.copy()
df3["slope"] = np.select(
    [df3["slope"] <= 3, df3["slope"] <= 5, df3["slope"] >= 10],
    [0, 1, 2],
    default=np.nan
)
df3["age"] = np.select(
    [df3["age"] < 7, df3["age"] < 9, df3["age"] >= 9],
    [0, 1, 2],
    default=np.nan
)

df3.head()


# %%
df3.shape

# %%
# Drop rows with missing values in the relevant columns for the Cox model
df3_cleaned = df3.dropna(subset=["hour", "reason_binary", "servo", "age", "slope"])

# Verify shape consistency
print(f"Shape of df3 before cleaning: {df3.shape}")
print(f"Shape of df3_cleaned: {df3_cleaned.shape}")

# Fit Cox model with transformed variables
cox_model = CoxPHFitter()
cox_model.fit(df3_cleaned, duration_col="hour", event_col="reason_binary", formula="servo + age + slope")

# Check time dependency using the cleaned data
results = proportional_hazard_test(cox_model, df3_cleaned, time_transform="rank")
print(results.summary)


# %%

# Transform data into long format
df["id"] = range(1, len(df) + 1)

# Convert wide to long format for h1 to h48
hurricane = df.melt(
    id_vars=["id", "hour", "reason", "servo", "slope", "age"],
    value_vars=[f"h{i}" for i in range(1, 49)],
    var_name="stop",
    value_name="pumpon"
)


# %%
hurricane.isna().mean()

# %%
hurricane[["servo", "age", "slope", "reason", "pumpon"]] = hurricane[["servo", "age", "slope", "reason", "pumpon"]].fillna(0).replace([np.inf, -np.inf], 0).astype(int)

# %%
hurricane.isna().mean()

# %%
hurricane.head()

# %%

# Additional transformations
hurricane["stop"] = hurricane["stop"].astype(str).str.extract("(\d+)").astype(int)
hurricane["start"] = hurricane["stop"] - 1
hurricane["twelve_hours"] = (
    hurricane.groupby("id")["pumpon"]
    .apply(lambda x: x.rolling(window=12, min_periods=12).sum() == 12)
    .reset_index(drop=True)
    .astype(int)
)
hurricane = hurricane[hurricane["stop"] <= hurricane["hour"]]
hurricane["reason"] = np.where(hurricane["hour"] == hurricane["stop"], hurricane["reason"], 0)
hurricane["slope"] = np.select(
    [hurricane["slope"] <= 3, hurricane["slope"] <= 5, hurricane["slope"] >= 10],
    [0, 1, 2],
    default=np.nan
)
hurricane["age"] = np.select(
    [hurricane["age"] < 7, hurricane["age"] < 9, hurricane["age"] >= 9],
    [0, 1, 2],
    default=np.nan
)
hurricane[["servo", "age", "slope", "reason", "pumpon", "twelve_hours"]] = hurricane[["servo", "age", "slope", "reason", "pumpon", "twelve_hours"]].fillna(0).replace([np.inf, -np.inf], 0).astype(int)

hurricane[["servo", "age", "slope", "reason", "pumpon", "twelve_hours"]] = hurricane[["servo", "age", "slope", "reason", "pumpon", "twelve_hours"]].astype(int)

hurricane["slope_tt"] = hurricane["slope"] * hurricane["start"]


# %%
hurricane.dropna(subset=["start", "reason", "servo", "age", "slope", "twelve_hours", "slope_tt"], inplace=True)

# %%
hurricane.isna().mean()

# %%
hurricane.tail(25)

# %%
hurricane.drop(columns={'slope'}, inplace=True)
hurricane.columns

# %%


# Fit a Cox model for the long-format data
hurricane_cph = CoxPHFitter()
hurricane_cph.fit(
    hurricane,
    duration_col="start",
    event_col="reason",
    formula="servo + twelve_hours"
)

# %%
hurricane_cph.summary

# %%
hurricane_cph.plot_partial_effects_on_outcome(covariates='twelve_hours', values=[0,1], cmap='coolwarm')

# %%



