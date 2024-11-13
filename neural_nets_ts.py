# %% [markdown]
# ## Time Series 2 HW 1
# - Your team is tasked with forecasting the electric load for American Electric Power Co. (AEP), one of the major electricity suppliers in mid-Atlantic. 
# 
# - Specifically, you will be forecasting hourly energy load for the Appalachian Power Territory of AEP. Electricity load by a power supplier need forecasts to better meet the needs of customers and prepare for operational expenses.
# 
# - You are to forecast hourly energy usage for October 25, 2024 – October 31, 2024. Every week you will be sent the updated data with the latest week’s energy data through October 24, 2024.

# %% [markdown]
# ### Phases:
# - Phase 1 – Exponential smoothing and seasonal ARIMA models
# 
# - Phase 2 – Prophet and neural network models
# 
# - Final Phase – Additional model as well as overall comparison of all modeling
# approaches used.

# %% [markdown]
# ### Phase 1:
# - The data contains monthly information with an annual seasonal component: 
# >>Feel free to try different approaches to account for the seasonality. 
# 
# >>Explain which approach you use and why.
# 
# - Build an appropriate Exponential Smoothing Model.
# >> Forecast this model for your validation set only.
# 
# >> Calculate the MAE and MAPE for the validation set.
# 
# - Build a seasonal ARIMA model.
# >>Describe the approach you used to select the lags of the model. 
# 
# >>Forecast this model for your validation set only.
# 
# >>Calculate the MAE and MAPE for the validation set.

# %% [markdown]
# #### Takeaway: dropped 6 rows
# - makes sense, we have 6 years of data, and there are 6 instances of 1 added hour each year
# - next we need to add in the missing hours. Let's check for those missing data points

# %%
# final code block
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# read in the CSV
ts_df = pd.read_csv("hrl_load_metered.csv")

#drop unused columns 
ts_df_rdx = ts_df.drop(columns=['is_verified', 'nerc_region', 'mkt_region','zone','load_area'])

# convert time column to date-time 
ts_df_rdx['datetime_beginning_ept'] = pd.to_datetime(ts_df_rdx['datetime_beginning_ept'], format='mixed')

# set the index as the date time column
ts_df_rdx.set_index(ts_df_rdx.datetime_beginning_ept, inplace=True)

#drop the index title
ts_df_rdx.index.name = None

ts_df_rdx.drop(columns='datetime_beginning_ept', inplace=True)

# drop duplicates due to daylight savings
deduped = ts_df_rdx.reset_index()
deduped.drop_duplicates(subset=['index'], keep='first', inplace=True)
deduped.set_index('index', inplace=True)

# remove the index title
deduped.index.name = None

# check for missing data points
full_date_range = pd.date_range(start=deduped.index.min(), end=deduped.index.max(), freq='H')

gaps = full_date_range.difference(deduped.index)

if not gaps.empty:
    print(f"gaps: {gaps}")
else:
    print("No gaps")

# now add in the missing hourly data, using df.asfreq()
deduped_and_filled = deduped
deduped_and_filled = deduped_and_filled.asfreq('H')

# next we can fill the missing values, using a spline interpolation
deduped_and_filled.head()

# now rerun the gap analysis
full_date_range = pd.date_range(start=deduped_and_filled.index.min(), end=deduped_and_filled.index.max(), freq='H')

gaps = full_date_range.difference(deduped_and_filled.index)

if not gaps.empty:
    print(f"gaps: {gaps}")
else:
    print("No gaps")

# Drop rows where all values are NaN before interpolation
df_valid = deduped_and_filled.dropna(subset=['mw'])

# Perform spline interpolation
spline_interpolator = interpolate.InterpolatedUnivariateSpline(df_valid.index.astype(int), df_valid['mw'], k=2)

# Fill missing values in the original df using the spline interpolator
deduped_and_filled['mw'] = deduped_and_filled['mw'].combine_first(pd.Series(spline_interpolator(deduped_and_filled.index.astype(int)), index=deduped_and_filled.index))

# 5. Plot the original data with the interpolated values
plt.figure(figsize=(10, 6))
plt.plot(df_valid.index, df_valid['mw'], label='Original Data', marker='o', linestyle='--', color='blue')
plt.plot(deduped_and_filled.index, deduped_and_filled['mw'], label='Spline Interpolated Data', color='orange')
plt.legend()
plt.show()

# Check for missing values again to ensure they were filled
print(deduped_and_filled.isna().sum())

#check the dataframe with the index of the previously empty values
print(deduped_and_filled[deduped_and_filled.index =='2020-03-08 02:00:00'])

print(deduped_and_filled.index.min())
print(deduped_and_filled.index.max())

# %%
energy_train = deduped_and_filled

# %%
energy_train.to_csv("energy_train.csv")

# %% [markdown]
# #### Do the same for the test data

# %%
# final code block
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# read in the CSV
ts_df = pd.read_csv("hrl_load_metered - test1.csv")

#drop unused columns 
ts_df_rdx = ts_df.drop(columns=['is_verified', 'nerc_region', 'mkt_region','zone','load_area'])

# convert time column to date-time 
ts_df_rdx['datetime_beginning_ept'] = pd.to_datetime(ts_df_rdx['datetime_beginning_ept'], format='mixed')

# set the index as the date time column
ts_df_rdx.set_index(ts_df_rdx.datetime_beginning_ept, inplace=True)

#drop the index title
ts_df_rdx.index.name = None

ts_df_rdx.drop(columns='datetime_beginning_ept', inplace=True)

# drop duplicates due to daylight savings
deduped = ts_df_rdx.reset_index()
deduped.drop_duplicates(subset=['index'], keep='first', inplace=True)
deduped.set_index('index', inplace=True)

# remove the index title
deduped.index.name = None

# check for missing data points
full_date_range = pd.date_range(start=deduped.index.min(), end=deduped.index.max(), freq='H')

gaps = full_date_range.difference(deduped.index)

if not gaps.empty:
    print(f"gaps: {gaps}")
else:
    print("No gaps")

# now add in the missing hourly data, using df.asfreq()
deduped_and_filled2 = deduped
deduped_and_filled2 = deduped_and_filled2.asfreq('H')

# next we can fill the missing values, using a spline interpolation
deduped_and_filled2.head()

# now rerun the gap analysis
full_date_range = pd.date_range(start=deduped_and_filled2.index.min(), end=deduped_and_filled2.index.max(), freq='H')

gaps = full_date_range.difference(deduped_and_filled2.index)

if not gaps.empty:
    print(f"gaps: {gaps}")
else:
    print("No gaps")

# Drop rows where all values are NaN before interpolation
df_valid = deduped_and_filled2.dropna(subset=['mw'])

# Perform spline interpolation
spline_interpolator = interpolate.InterpolatedUnivariateSpline(df_valid.index.astype(int), df_valid['mw'], k=2)

# Fill missing values in the original df using the spline interpolator
deduped_and_filled2['mw'] = deduped_and_filled2['mw'].combine_first(pd.Series(spline_interpolator(deduped_and_filled2.index.astype(int)), index=deduped_and_filled2.index))

# 5. Plot the original data with the interpolated values
plt.figure(figsize=(10, 6))
plt.plot(df_valid.index, df_valid['mw'], label='Original Data', marker='o', linestyle='--', color='blue')
plt.plot(deduped_and_filled2.index, deduped_and_filled2['mw'], label='Spline Interpolated Data', color='orange')
plt.legend()
plt.show()

# Check for missing values again to ensure they were filled
print(deduped_and_filled2.isna().sum())

#check the dataframe with the index of the previously empty values
print(deduped_and_filled2[deduped_and_filled2.index =='2020-03-08 02:00:00'])

print(deduped_and_filled2.index.min())
print(deduped_and_filled2.index.max())

# %%
energy_validate = deduped_and_filled2

# %% [markdown]
# ### Building an ESM Model

# %% [markdown]
# ##### Step 1: Plot the Data and explore the trends

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# first step: plot the data for a visual overview
plt.plot(energy_train["mw"])
plt.xlabel("Date")
plt.ylabel("MW")
plt.title("Hourly Electric Load")
plt.show()

# %%
# plot by hour
energy_train['Hour'] = energy_train.index.hour
energy_train['Year'] = energy_train.index.year

# %%
energy_train['Month'] = energy_train.index.month

# %%
palette = sns.cubehelix_palette(len(energy_train['Year'].unique()), start=0.5, rot=-0.75, dark=0.2, light=0.8)


plt.figure(figsize=(12,6))
sns.lineplot(data=energy_train, x='Hour', y='mw', hue='Year', palette=palette)

plt.title('Plot of Hourly Energy Use, Average Day by Year')
plt.xlabel('Hour of the Day')
plt.ylabel('MW')
plt.show()

# %% [markdown]
# ##### Takeaways:
# - the trends generally show a sharp increase in demand starting around 0500 in the morning and lasting until roughly 0900, where it plateaus.
# - from 0900 to 1900 it is sligthly positive but overall flat.
# - after 2000 it sharply decreases, and reaches a low point generally at 0300.

# %%
palette = sns.cubehelix_palette(len(energy_train['Month'].unique()), start=0.5, rot=-0.75, dark=0.2, light=0.8)


plt.figure(figsize=(12,6))
sns.lineplot(data=energy_train, x='Hour', y='mw', hue='Month', palette=palette)

plt.title('Plot of Hourly Energy Use, Average Day by Month')
plt.xlabel('Hour of the Day')
plt.ylabel('MW')
plt.show()

# %% [markdown]
# ##### Takeaways:
# - the trends are sharply different between hotter and cooler months. Cooler months show high peaks IVO 0900 and 2000
# - Hotter months show peaks during the hottest part of the day, reaching peaks IVO 1600/1700.
# - Cooler months have bimodal peaks, hotter months have unimodal peaks.

# %% [markdown]
# ##### Step Two: Seasonal Decomposition

# %% [markdown]
# ##### Daily Seasonality

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA
from statsforecast.utils import generate_series

# %%
# 'ds' is the date column, and 'y' is the value column.

# format for the STL Decomp
d = {'unique_id': 1, 'ds': energy_train.index, 'y': energy_train['mw']}
energy_sf = pd.DataFrame(data = d)

# %%
d2 = {'unique_id': 1, 'ds': energy_validate.index, 'y': energy_validate['mw']}
energy_valid_sf = pd.DataFrame(data = d2)

# %%
energy_sf.head(5)

# %% [markdown]
# #### Daily Seasonality

# %%
# the season here is a Day, so the the season length will be 24, for our hourly data
dcmp = StatsForecast(models = [MSTL(season_length=24)], freq = 'H')
dcmp.fit(df=energy_sf)

# %%
# decompose and view the decomposed values
result = dcmp.fitted_[0,0].model_
result

# %%
# plot the decomp
result.plot(subplots=True, grid=True)
## array([<Axes: >, <Axes: >, <Axes: >, <Axes: >], dtype=object)

plt.tight_layout()
plt.show()

# %%
# get the relative trend strenghts for trend and season

# Extract components
trend = result.trend
seasonal = result.seasonal
residual = result.remainder
data = result.data  # Original data
t_and_r = trend + residual

# Ensure all series are aligned
if not trend.empty and not residual.empty and not seasonal.empty:
    # Calculate Trend Strength
    var_residual = residual.var()
    var_t_plus_r = t_and_r.var()
    var_detrended = (data - trend).var()

    if var_detrended == 0:
        trend_strength = 0
        print("Trend Strength: 0 (Detrended variance is zero)")
    else:
        trend_strength = 1 - (var_residual / var_t_plus_r)
        trend_strength = max(0, trend_strength)  # Avoid negative strength
        print(f"Trend Strength: {trend_strength:.2f}")

    # Calculate Seasonal Strength
    var_seasonal = seasonal.var()

    if var_detrended == 0:
        seasonal_strength = 0
        print("Seasonal Strength: 0 (Detrended variance is zero)")
    else:
        seasonal_strength = var_seasonal / var_detrended
        seasonal_strength = max(0, seasonal_strength)  # Avoid negative strength
        print(f"Seasonal Strength: {seasonal_strength:.2f}")
else:
    print("Trend, Seasonal, or Residual series is empty or misaligned.")


# %% [markdown]
# #### Weekly Seasonality

# %%
# the season here is a Week, so the the season length will be 24x7=168, for our hourly data
dcmp = StatsForecast(models = [MSTL(season_length=[168])], freq = 'H')
dcmp.fit(df=energy_sf)

# %%
# decompose and view the decomposed values
result2 = dcmp.fitted_[0,0].model_
result2

# %%
# plot the decomp
result2.plot(subplots=True, grid=True)
## array([<Axes: >, <Axes: >, <Axes: >, <Axes: >], dtype=object)

plt.tight_layout()
plt.show()

# %%
# get the relative trend strenghts for trend and season

# Extract components
trend = result2.trend
seasonal = result2.seasonal
residual = result2.remainder
data = result2.data  # Original data
t_and_r = trend + residual

# Ensure all series are aligned
if not trend.empty and not residual.empty and not seasonal.empty:
    # Calculate Trend Strength
    var_residual = residual.var()
    var_t_plus_r = t_and_r.var()
    var_detrended = (data - trend).var()

    if var_detrended == 0:
        trend_strength = 0
        print("Trend Strength: 0 (Detrended variance is zero)")
    else:
        trend_strength = 1 - (var_residual / var_t_plus_r)
        trend_strength = max(0, trend_strength)  # Avoid negative strength
        print(f"Trend Strength: {trend_strength:.2f}")

    # Calculate Seasonal Strength
    var_seasonal = seasonal.var()

    if var_detrended == 0:
        seasonal_strength = 0
        print("Seasonal Strength: 0 (Detrended variance is zero)")
    else:
        seasonal_strength = var_seasonal / var_detrended
        seasonal_strength = max(0, seasonal_strength)  # Avoid negative strength
        print(f"Seasonal Strength: {seasonal_strength:.2f}")
else:
    print("Trend, Seasonal, or Residual series is empty or misaligned.")


# %% [markdown]
# ##### Monthly Seasonality

# %%
# the season here is a Week, so the the season length will be 24x7=168, for our hourly data
dcmp = StatsForecast(models = [MSTL(season_length=[730])], freq = 'H')
dcmp.fit(df=energy_sf)

# %%
# decompose and view the decomposed values
result3 = dcmp.fitted_[0,0].model_
result3

# %%
# plot the decomp
result3.plot(subplots=True, grid=True)
## array([<Axes: >, <Axes: >, <Axes: >, <Axes: >], dtype=object)

plt.tight_layout()
plt.show()

# %%
# get the relative trend strenghts for trend and season

# Extract components
trend = result3.trend
seasonal = result3.seasonal
residual = result3.remainder
data = result3.data  # Original data
t_and_r = trend + residual

# Ensure all series are aligned
if not trend.empty and not residual.empty and not seasonal.empty:
    # Calculate Trend Strength
    var_residual = residual.var()
    var_t_plus_r = t_and_r.var()
    var_detrended = (data - trend).var()

    if var_detrended == 0:
        trend_strength = 0
        print("Trend Strength: 0 (Detrended variance is zero)")
    else:
        trend_strength = 1 - (var_residual / var_t_plus_r)
        trend_strength = max(0, trend_strength)  # Avoid negative strength
        print(f"Trend Strength: {trend_strength:.2f}")

    # Calculate Seasonal Strength
    var_seasonal = seasonal.var()

    if var_detrended == 0:
        seasonal_strength = 0
        print("Seasonal Strength: 0 (Detrended variance is zero)")
    else:
        seasonal_strength = var_seasonal / var_detrended
        seasonal_strength = max(0, seasonal_strength)  # Avoid negative strength
        print(f"Seasonal Strength: {seasonal_strength:.2f}")
else:
    print("Trend, Seasonal, or Residual series is empty or misaligned.")


# %% [markdown]
# ##### Daily/Weekly/Monthly Seasonality decomp with MSTL

# %%
# the season here is a Week, so the the season length will be 24x7=168, for our hourly data
dcmp = StatsForecast(models = [MSTL(season_length=[24,168,730])], freq = 'H')
dcmp.fit(df=energy_sf)

# %%
# decompose and view the decomposed values
result4 = dcmp.fitted_[0,0].model_
result4

# %%
# plot the decomp
result4.plot(subplots=True, grid=True)
## array([<Axes: >, <Axes: >, <Axes: >, <Axes: >], dtype=object)

plt.tight_layout()
plt.show()

# %%
# get the relative trend strengths for trend and season

# Extract components
trend = result4.trend
seasonal = result4.seasonal730
residual = result4.remainder
data = result4.data  # Original data
t_and_r = trend + residual

# Ensure all series are aligned
if not trend.empty and not residual.empty and not seasonal.empty:
    # Calculate Trend Strength
    var_residual = residual.var()
    var_t_plus_r = t_and_r.var()
    var_detrended = (data - trend).var()

    if var_detrended == 0:
        trend_strength = 0
        print("Trend Strength: 0 (Detrended variance is zero)")
    else:
        trend_strength = 1 - (var_residual / var_t_plus_r)
        trend_strength = max(0, trend_strength)  # Avoid negative strength
        print(f"Trend Strength: {trend_strength:.2f}")

    # Calculate Seasonal Strength
    var_seasonal = seasonal.var()

    if var_detrended == 0:
        seasonal_strength = 0
        print("Seasonal Strength: 0 (Detrended variance is zero)")
    else:
        seasonal_strength = var_seasonal / var_detrended
        seasonal_strength = max(0, seasonal_strength)  # Avoid negative strength
        print(f"Seasonal Strength: {seasonal_strength:.2f}")
else:
    print("Trend, Seasonal, or Residual series is empty or misaligned.")


# %% [markdown]
# ##### Takeaways: using a weekly or monthly season reduces the seasonal strength. Likely will use that in the prediction model

# %% [markdown]
# ##### Step Three: Predict

# %% [markdown]
# ##### 1. Exponential Smoothing Model

# %% [markdown]
# ##### A. With Weekly Seasonality

# %%
from statsforecast.models import HoltWinters

# set the season length to 168 hrs, and make sure frequency is hourly. This code builds both additive/multiplicative models

model_HW = StatsForecast(models = [HoltWinters(season_length = 168, error_type="M", alias = "HWM"), HoltWinters(season_length = 168, error_type="A", alias = "HWA")], freq = 'H')
model_HW.fit(df = energy_sf)

# %%
print(model_HW.fitted_[0,0].model_.keys())

# %%
result_HWM = model_HW.fitted_[0,0].model_
print("Mult. AIC =", result_HWM['aic'], "\nMult. AICc =", result_HWM['aicc'], "\nMult. BIC =", result_HWM['bic'])

# %%
result_HWA = model_HW.fitted_[0,1].model_
print("Add. AIC =", result_HWA['aic'], "\nAdd. AICc =", result_HWA['aicc'], "\nAdd. BIC =", result_HWA['bic'])

# %%
# predict one week in the future
model_HW_for = model_HW.forecast(df = energy_sf, h = 168, level = [95])

#plot the predictions against the actual holdout week of data
model_HW.plot(energy_valid_sf, model_HW_for)

# %%
# error for the Multiplicative Model
error = np.array(energy_valid_sf['y']) - np.array(model_HW_for['HWM'])
MAPE = np.mean(abs(error)/energy_valid_sf['y'])*100
print("Mult. MAE =", np.mean(abs(error)), "\nMult. MAPE =", MAPE)

# %%
# error for the Additive Model
error = np.array(energy_valid_sf['y']) - np.array(model_HW_for['HWA'])
MAPE = np.mean(abs(error)/energy_valid_sf['y'])*100
print("Add. MAE =", np.mean(abs(error)), "\nAdd. MAPE =", MAPE)

# %% [markdown]
# #### B. With Daily Seasonality

# %%
# set the season length to 730 hrs, and make sure frequency is hourly. This code builds both additive/multiplicative models
model_HW2 = StatsForecast(models = [HoltWinters(season_length = 24, error_type="M", alias = "HWM"), HoltWinters(season_length = 24, error_type="A", alias = "HWA")], freq = 'H')
model_HW2.fit(df = energy_sf)

# %%
print(model_HW2.fitted_[0,0].model_.keys())

# %%
result_HWM2 = model_HW2.fitted_[0,0].model_
print("Mult. AIC =", result_HWM2['aic'], "\nMult. AICc =", result_HWM2['aicc'], "\nMult. BIC =", result_HWM2['bic'])

# %%
result_HWA2 = model_HW2.fitted_[0,1].model_
print("Add. AIC =", result_HWA2['aic'], "\nAdd. AICc =", result_HWA2['aicc'], "\nAdd. BIC =", result_HWA2['bic'])

# %%
# predict one week in the future
model_HW_for2 = model_HW2.forecast(df = energy_sf, h = 168, level = [95])

#plot the predictions against the actual holdout week of data
model_HW.plot(energy_valid_sf, model_HW_for2)

# %%
# error for the Multiplicative Model
error = np.array(energy_valid_sf['y']) - np.array(model_HW_for2['HWM'])
MAPE = np.mean(abs(error)/energy_valid_sf['y'])*100
print("Mult. MAE =", np.mean(abs(error)), "\nMult. MAPE =", MAPE)

# %%
# error for the Additive Model
error = np.array(energy_valid_sf['y']) - np.array(model_HW_for2['HWA'])
MAPE = np.mean(abs(error)/energy_valid_sf['y'])*100
print("Add. MAE =", np.mean(abs(error)), "\nAdd. MAPE =", MAPE)

# %% [markdown]
# ##### Takeaways:
# - the best performing model on the out of sample data was the Holt Winters Multiplicative ESM, with Daily Seasonality:
# - Model performance stats: Mult. AIC = 1295841.902689142, Mult. AICc = 1295841.9323324007 , Mult. BIC = 1296102.3421740618
# - Model Evaluation Stats: MAE: 144.48, MAPE: 3.67

# %% [markdown]
# ### Building a Seasonal ARIMA Model

# %% [markdown]
# ##### Step One: Creating a Deterministic Solution
# - the best performing ESM model, the Holt Winter Multiplicative, with Daily Seasonality, had a daily seasonal component
# - that means that it's unrealistic to create dummy variables
# - will need to use a Fourier Transformation, with k up to 12 (24/2)

# %%
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.feature_engineering import fourier
import pandas as pd

# %%
# the dataset is too large/complex - let's take the last two years of data, instead of the last 6, and run the fourier transform on that
# additionally we may just need to try it with weekly seasonality instead

energy_sf_rdx = energy_sf.loc['2023-01-01 00:00:00': '2024-09-12 23:00:00']

# %%
## Daily Seasonality

# Define possible values for k
k_values = [2, 6, 12]  # Adjust this list based on what range you want to explore

# Initialize an array to store AIC values
aic = np.array([])

# Loop through the different values of k
for k in k_values:
    print(k)
    # Generate Fourier terms for the current value of k
    energy_sf_f, _ = fourier(energy_sf_rdx, freq='H', season_length=24, k=k, h=168)
    
    # Fit the model using the Fourier-transformed dataset
    model_F_ARIMA = StatsForecast(models=[AutoARIMA(season_length=24, D=0)], freq='H')
    
    # The dataset now includes the original series and the Fourier terms
    model_F_ARIMA.fit(df=energy_sf_f)
    
    # Get the AIC value from the model's internal ARIMA object
    aic_value = model_F_ARIMA.fitted_[0][0].model_.get("aic")
    print(aic_value)
    
    # Append the AIC value for the current k
    aic = np.append(aic, aic_value)

# Print AIC values for each k
for i, k in enumerate(k_values):
    print(f'k = {k}, AIC = {aic[i]}')


# %% [markdown]
# ##### Approach 2: Stochastic (take Differences)
# - check if we need to take differences, with the ADF

# %%
energy_sf_rdx.head()

# %%
# import Dickey Fuller test
from statsmodels.tsa.stattools import adfuller

# %%
# compare this hour to itself, a year ago
energy_sf_rdx["energy_daily_diff"] = energy_sf_rdx["y"] - energy_sf_rdx["y"].shift(24) #number of hrs in a yr

# %%
energy_sf_rdx.head(25)

# %%
# run ADF with daily differencing
result_d = adfuller(energy_sf_rdx.iloc[365:]["energy_daily_diff"])
result_d

# %%
# first step: plot the data for a visual overview
plt.figure(figsize=(12,6))
plt.plot(energy_sf_rdx["y"].iloc[25:])
plt.xlabel("Date")
plt.ylabel("MW")
plt.title("Hourly Electric Load")
plt.show()

# %%
# first step: plot the data for a visual overview
plt.figure(figsize=(12,6))
plt.plot(energy_sf_rdx["energy_daily_diff"].iloc[25:])
plt.xlabel("Date")
plt.ylabel("MW")
plt.title("Differenced Hourly Electric Load, Daily Diffs")
plt.show()

# %% [markdown]
# ##### Takeaways:
# - to evaluate the ADF test, look at the ADF statistic (first line), and compare it to the Critical Values (next to %s)
# - if p_value is below significance threshold, and the ADF is less than the critical values, then there is stationarity
# - in this case, the data is evaluated as stationary. Due to p-value, we will use Daily Diffs

# %% [markdown]
# ##### Step Three B.: Build the sARIMA
# - evaluate the ACF and PACF plots

# %%
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# %%
energy_sf_rdx2 = energy_sf_rdx.iloc[25:]
energy_sf_rdx2.head()

# %%
plot_acf(energy_sf_rdx2["energy_daily_diff"],lags=72)
pyplot.show()

# %% [markdown]
# ### ACF Takeaways:
# 1. Use the ACF plot set MA (moving average) terms. These are q and Q
# 2. For q: this plot is exponenetially decreasing in the 1st lag period (24 hours), so q=0. Outside of the 1st season, there are spikes at lags 2 and 3. Q=3

# %%
plot_pacf(energy_sf_rdx2["energy_daily_diff"],lags=72)
pyplot.show()

# %% [markdown]
# ### PACF Takeaways:
# 1. Use the pACF plot set AR (auto-regressive) terms. These are p and P
# 2. For p: this plot spikes at 1,2,3 in the 1st lag period (24 hours), so p=3. Outside of the 1st season, there are spikes at lags 1,2 and 3. P=3

# %% [markdown]
# #### Notes on Building Our Model:
# 1. We have derived potential p,P,q, and Q terms. We also know that we needed to take a difference for Trend initially. Here's our model in correct notation:
# 2. ARIMA(3,1,0)(3,0,2) subscript 24. The subscript 24 indicates the season length, which is 24 hours.

# %% [markdown]
# #### Building the SARIMA - use our hand calculated model, and use one Auto Search model

# %%
from statsforecast.models import AutoARIMA, ARIMA
import statsmodels.api as sm

# %%
energy_sf_rdx.drop(columns=['energy_daily_diff'], inplace=True)

# %%
model_SARIMA = StatsForecast(models = [ARIMA(order = (3, 1, 2), season_length = 24, seasonal_order = (1,0,1), include_drift = True), AutoARIMA(season_length = 24, allowdrift = True)], freq = 'H')
model_SARIMA.fit(df = energy_sf_rdx)

# %%
#get the ARMA terms from our model
model_SARIMA.fitted_[0][0].model_.get("arma")

# %%
# get the coefficients from our model
model_SARIMA.fitted_[0][0].model_.get("coef")

# %%
model_SARIMA.fitted_[0][0].model_.get("aicc")

# %%
# get the Ljung-Box statistic from our model
sm.stats.acorr_ljungbox(model_SARIMA.fitted_[0][0].model_.get("residuals"), lags = [72], model_df = 9)

# %% [markdown]
# ##### Get the Model Parameters for the Auto Search model:

# %%
auto_arma = model_SARIMA.fitted_[0][1].model_.get("arma")
auto_coeff = model_SARIMA.fitted_[0][1].model_.get("coef")
auto_aicc = model_SARIMA.fitted_[0][1].model_.get("aicc")
auto_lb = sm.stats.acorr_ljungbox(model_SARIMA.fitted_[0][1].model_.get("residuals"), lags = [72], model_df = 31)

print(f"Auto arma: {auto_arma}\nAuto coefficients: {auto_coeff}\nAuto AICc: {auto_aicc}\n{auto_lb}")

# %% [markdown]
# #### Plot the forecasts against the Validation Data, and check Performance

# %%
model_SARIMA_for = model_SARIMA.forecast(df = energy_sf_rdx, h = 168, level = [95])

model_SARIMA.plot(energy_valid_sf, model_SARIMA_for)

# %%
error = np.array(energy_valid_sf['y']) - np.array(model_SARIMA_for['AutoARIMA'])
MAPE = np.mean(abs(error)/energy_valid_sf['y'])*100
print("Auto. MAE =", np.mean(abs(error)), "\nAuto. MAPE =", MAPE)

# %%
error = np.array(energy_valid_sf['y']) - np.array(model_SARIMA_for['ARIMA'])
MAPE = np.mean(abs(error)/energy_valid_sf['y'])*100
print("Hand MAE =", np.mean(abs(error)), "\nHand MAPE =", MAPE)

# %% [markdown]
# #### Part 2: Prophet and TS Neural Nets

# %% [markdown]
# ##### Scheme of Maneuver
# - combine the first 3 test sets into a Validation set
# - reserve the 4th test set as actual test set
# - run all through the same cleaning script
# - run the prophet model, maybe with two seasons, and maybe with an intervention for hurricane Helene
# - run the Neural Net
# - run the best Prophet model and best NN model on validation
# - run the best performer across train and validate on test (final week of data)

# %%
# basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# library for interpolation
from scipy import interpolate
# libraries for ARIMA and Neural Nets
import sklearn
from sklearn.neural_network import MLPRegressor

# %% [markdown]
# ##### Data Cleaning and Transformation

# %%
# train OG
df1 = pd.read_csv("hrl_load_metered.csv")
df_train_og = df1

# %%
df_train_og.tail(2)

# %%
# combine these three into a super-validate. This will be three weeks of data.
df2 = pd.read_csv("hrl_load_metered - test1.csv")
df3 = pd.read_csv("hrl_load_metered - test2.csv")
df4 = pd.read_csv("hrl_load_metered - test3.csv")

df_validate_final = pd.concat([df2,df3,df4], ignore_index=True)

# %%
df_validate_final.tail(2)

# %%
# create a third df, which is train + validations, we will retrain the best performing models on these
df_train_final = pd.concat([df_train_og, df_validate_final], ignore_index=True)

# %%
df_train_final.tail(2)

# %%
# test - hold out data from the most recent week
df5 = pd.read_csv("hrl_load_metered - test4.csv")
df_test = df5

# %%
df_test.tail(2)

# %%
df_train_og_rdx
df_validate_final_rdx
df_train_final_rdx
df_test_rdx

# %%
## Create cleaned, interpolated version of all the final dfs: 
# df_train_og, 
# df_validate_final, 
# df_train_final, 
# df_test

#drop unused columns 
df_train_og_rdx = df_train_og.drop(columns=['is_verified', 'nerc_region', 'mkt_region','zone','load_area'])
df_validate_final_rdx = df_validate_final.drop(columns=['is_verified', 'nerc_region', 'mkt_region','zone','load_area'])
df_train_final_rdx = df_train_final.drop(columns=['is_verified', 'nerc_region', 'mkt_region','zone','load_area'])
df_test_rdx = df_test.drop(columns=['is_verified', 'nerc_region', 'mkt_region','zone','load_area'])

# convert time column to date-time 
# ex: ts_df_rdx['datetime_beginning_ept'] = pd.to_datetime(ts_df_rdx['datetime_beginning_ept'], format='mixed')
df_train_og_rdx['datetime_beginning_ept'] = pd.to_datetime(df_train_og_rdx['datetime_beginning_ept'], format='mixed')
df_validate_final_rdx['datetime_beginning_ept'] = pd.to_datetime(df_validate_final_rdx['datetime_beginning_ept'], format='mixed')
df_train_final_rdx['datetime_beginning_ept'] = pd.to_datetime(df_train_final_rdx['datetime_beginning_ept'], format='mixed')
df_test_rdx['datetime_beginning_ept'] = pd.to_datetime(df_test_rdx['datetime_beginning_ept'], format='mixed')


# set the index as the date time column
# ex: ts_df_rdx.set_index(ts_df_rdx.datetime_beginning_ept, inplace=True)
df_train_og_rdx.set_index(df_train_og_rdx.datetime_beginning_ept, inplace=True)
df_validate_final_rdx.set_index(df_validate_final_rdx.datetime_beginning_ept, inplace=True)
df_train_final_rdx.set_index(df_train_final_rdx.datetime_beginning_ept, inplace=True)
df_test_rdx.set_index(df_test_rdx.datetime_beginning_ept, inplace=True)

#drop the index title
# ex: ts_df_rdx.index.name = None
df_train_og_rdx.index.name = None
df_validate_final_rdx.index.name = None
df_train_final_rdx.index.name = None
df_test_rdx.index.name = None
# ex: ts_df_rdx.drop(columns='datetime_beginning_ept', inplace=True)
df_train_og_rdx.drop(columns='datetime_beginning_ept', inplace=True)
df_validate_final_rdx.drop(columns='datetime_beginning_ept', inplace=True)
df_train_final_rdx.drop(columns='datetime_beginning_ept', inplace=True)
df_test_rdx.drop(columns='datetime_beginning_ept', inplace=True)

# drop duplicates due to daylight savings
# ex: deduped = ts_df_rdx.reset_index()
df_train_og_rdx = df_train_og_rdx.reset_index()
df_validate_final_rdx = df_validate_final_rdx.reset_index()
df_train_final_rdx = df_train_final_rdx.reset_index()
df_test_rdx = df_test_rdx.reset_index()
# ex: deduped.drop_duplicates(subset=['index'], keep='first', inplace=True)
df_train_og_rdx.drop_duplicates(subset=['index'], keep='first', inplace=True)
df_validate_final_rdx.drop_duplicates(subset=['index'], keep='first', inplace=True)
df_train_final_rdx.drop_duplicates(subset=['index'], keep='first', inplace=True)
df_test_rdx.drop_duplicates(subset=['index'], keep='first', inplace=True)
# ex: deduped.set_index('index', inplace=True)
df_train_og_rdx.set_index('index', inplace=True)
df_validate_final_rdx.set_index('index', inplace=True)
df_train_final_rdx.set_index('index', inplace=True)
df_test_rdx.set_index('index', inplace=True)

# remove the index title
# deduped.index.name = None
df_train_og_rdx.index.name = None
df_validate_final_rdx.index.name = None
df_train_final_rdx.index.name = None
df_test_rdx.index.name = None

# check for missing data points
full_date_range_train_og = pd.date_range(start=df_train_og_rdx.index.min(), end=df_train_og_rdx.index.max(), freq='H')
full_date_range_validate_final = pd.date_range(start=df_validate_final_rdx.index.min(), end=df_validate_final_rdx.index.max(), freq='H')
full_date_range_train_final = pd.date_range(start=df_train_final_rdx.index.min(), end=df_train_final_rdx.index.max(), freq='H')
full_date_range_test = pd.date_range(start=df_test_rdx.index.min(), end=df_test_rdx.index.max(), freq='H')

gaps1 = full_date_range_train_og.difference(df_train_og_rdx.index)
gaps2 = full_date_range_validate_final.difference(df_validate_final_rdx.index)
gaps3 = full_date_range_train_final.difference(df_train_final_rdx.index)
gaps4 = full_date_range_test.difference(df_test_rdx.index)

gaps_list = [gaps1, gaps2, gaps3, gaps4]

for gap in gaps_list:
    if not gap.empty:
        print(f"gaps: {gap}")
    else:
        print("No gaps")


# %%
# need to fill gaps only for df_train_og and df_train_final

# now add in the missing hourly data, using df.asfreq()
# ex: deduped_and_filled = deduped_and_filled.asfreq('H')
df_train_og_rdx = df_train_og_rdx.asfreq('H')
df_validate_final_rdx = df_validate_final_rdx.asfreq('H')
df_train_final_rdx = df_train_final_rdx.asfreq('H')
df_test_rdx = df_test_rdx.asfreq('H')

# Drop rows where all values are NaN before interpolation
# ex: df_valid = deduped_and_filled.dropna(subset=['mw'])
df_train_og_rdx = df_train_og_rdx.dropna(subset=['mw'])
df_validate_final_rdx = df_validate_final_rdx.dropna(subset=['mw'])
df_train_final_rdx = df_train_final_rdx.dropna(subset=['mw'])
df_test_rdx = df_test_rdx.dropna(subset=['mw'])

# Perform spline interpolation
# ex: spline_interpolator = interpolate.InterpolatedUnivariateSpline(df_valid.index.astype(int), df_valid['mw'], k=2)
spline_interpolator1 = interpolate.InterpolatedUnivariateSpline(df_train_og_rdx.index.astype(int), df_train_og_rdx['mw'], k=2)
spline_interpolator2 = interpolate.InterpolatedUnivariateSpline(df_validate_final_rdx.index.astype(int), df_validate_final_rdx['mw'], k=2)
spline_interpolator3 = interpolate.InterpolatedUnivariateSpline(df_train_final_rdx.index.astype(int), df_train_final_rdx['mw'], k=2)
spline_interpolator4 = interpolate.InterpolatedUnivariateSpline(df_test_rdx.index.astype(int), df_test_rdx['mw'], k=2)

# Fill missing values in the original df using the spline interpolator
# ex: deduped_and_filled['mw'] = deduped_and_filled['mw'].combine_first(pd.Series(spline_interpolator(deduped_and_filled.index.astype(int)), index=deduped_and_filled.index))
df_train_og_rdx['mw'] = df_train_og_rdx['mw'].combine_first(pd.Series(spline_interpolator1(df_train_og_rdx.index.astype(int)), index=df_train_og_rdx.index))
df_validate_final_rdx['mw'] = df_validate_final_rdx['mw'].combine_first(pd.Series(spline_interpolator2(df_validate_final_rdx.index.astype(int)), index=df_validate_final_rdx.index))
df_train_final_rdx['mw'] = df_train_final_rdx['mw'].combine_first(pd.Series(spline_interpolator3(df_train_final_rdx.index.astype(int)), index=df_train_final_rdx.index))
df_test_rdx['mw'] = df_test_rdx['mw'].combine_first(pd.Series(spline_interpolator4(df_test_rdx.index.astype(int)), index=df_test_rdx.index))

# Check for missing values again to ensure they were filled
print(df_train_og_rdx.isna().sum())
print(df_validate_final_rdx.isna().sum())
print(df_train_final_rdx.isna().sum())
print(df_test_rdx.isna().sum())


#check the dataframe with the index of the previously empty values
print(df_train_og_rdx[df_train_og_rdx.index =='2020-03-08 02:00:00'])
print(df_train_final_rdx[df_train_final_rdx.index =='2020-03-08 02:00:00'])

print(df_train_final_rdx.index.min())
print(df_train_final_rdx.index.max())

# %% [markdown]
# ##### Create final named dataframes

# %%
train_og = df_train_og_rdx
validate_combined = df_validate_final_rdx
train_final = df_train_final_rdx
test = df_test_rdx

# %%
train_og['hh'] = 0 
validate_combined['hh'] = 0 # set HH dates for this one
train_final['hh'] = 0 # set HH dates for this one
test['hh'] = 0 

# %% [markdown]
# #### Train Neural Net Model on Train

# %%
train_final['hh']['2024-09-24 00:00:00']

# %%
train_og.head()

# %%
# building a neural net with one hidden layer, start with custom suffix for our nn df
train_final_nn = train_final

# we are going to set the season as 1 day, creating a difference from the data point 24 hours prior. We are doing this to produce stationary data, which the NN needs
train_final_nn['diff'] = train_final_nn['mw'] - train_final_nn['mw'].shift(24)

# define the lags
train_final_nn['L1'] = train_final_nn['diff'].shift()
train_final_nn['L24'] = train_final_nn['diff'].shift(24)
train_final_nn['L48'] = train_final_nn['diff'].shift(48)
train_final_nn['L72'] = train_final_nn['diff'].shift(72)

validate_nn = train_final_nn.tail(504)
train_nn = train_final_nn.iloc[:-504]

# %%
plt.figure(figsize=(12,6))
plt.plot(train_nn["diff"].dropna())
plt.xlabel("Date")
plt.ylabel("MW")
plt.title("Seasonal Difference in MW")
plt.show()

# %%
train_nn.head()

# %%
# we are going to use only the index number, and the lagged columns for predictions. So we will reformat the dataframes to reflect that. We will also only use data from 2024
# Note, we will lose the first 24 observations, becasue differences begin on the 25th observation, which is the first observation where there was an observation 24 hours prior, which could produce a diff

X = train_nn.drop(['mw', 'diff'], axis = 1)
X = X.loc['2024-01-01':]
X.head()


# %%
# now drop the index, and we will have the df correctly formatted for the NN
X = X.reset_index(drop=True)
X.head()


# %%
y = train_nn['diff']
y = y.loc['2024-01-01':]
y = y.reset_index(drop=True)


# %%
y.head(2)

# %%

model_nnet = MLPRegressor(hidden_layer_sizes = (3,), random_state=123, max_iter = 100000).fit(X, y)

# %%
# use combined_validate as the test for this portion
X_test = validate_nn.drop(['mw', 'diff'], axis = 1)

# %%
X_test = X_test.reset_index(drop=True)
X_test.head(5)

# %%
len(X_test)

# %%
# the last 504 observations in the training data are the last 3 weeks of data. We are going to add the predictions we have for the differences from this NN for 3 weeks of data, to the last 3 weeks
# of observations in the data, so that our predictions are not differences but instead they are actual MW unit predictions.

training_tail = train_nn.tail(504)
training_tail


# %%

training_tail['NN_pred'] = model_nnet.predict(X_test)
training_tail['NN_pred']


# %%

nn_pred = training_tail['NN_pred'] + training_tail['mw']

nn_pred.head()


# %%
nn_pred.index = pd.to_datetime(pd.date_range(start='2024-09-13 00:00:00', end='2024-10-03 23:00:00', freq='H'))

validate_nn['nn_pred'] = nn_pred
validate_nn.head()


# %%
plt.cla()
fcast = validate_nn['nn_pred']
ax = train_nn["mw"].plot(figsize=(12,8))
fcast.plot(ax = ax,color = "orange")
plt.show()

# %%
error = validate_nn['mw'] - validate_nn['nn_pred']
MAPE = np.mean(abs(error)/validate_nn['mw'])*100
print("NN MAE =", np.mean(abs(error)), "\nNN MAPE =", MAPE)

# %% [markdown]
# #### Takeaways:
# - pretty rough model accuracy. But Hurricane Helene likely impacted real world energy use during this time.
# - may try this again with an intervention variable. SEPT 24 2024 - OCT 01 2023

# %% [markdown]
# #### Prophet Model

# %%
# train_og
# validate_combined
# train_final
# test

# %%
from prophet import Prophet

# %%
d_X = {'unique_id': 1, 'ds': train_og.index, 'HurrHel': train_og.hh}
X_sf = pd.DataFrame(data = d_X)

train_sf_X = train_sf.merge(X_sf, how = 'left', on = ['unique_id', 'ds']) 

from prophet import Prophet

m = Prophet(seasonality_mode='multiplicative')
m.add_regressor('HurrHel')

# %%



