# %%
import pandas as pd

# %%
ames = pd.read_csv("ames.csv")
ames.head()

# %%
ames.shape

# %%
ames.columns

# %%
cols_to_keep = ['Sale_Price', 'Street', 'Central_Air', 'Bedroom_AbvGr', 'Year_Built', 'Mo_Sold', 'Lot_Area',
       'First_Flr_SF', 'Second_Flr_SF', 'Fireplaces', 'Garage_Area', 'Gr_Liv_Area']

# %%
ames_update = ames[cols_to_keep]
ames_update.head()

# %%
ames_update.columns

# %%
X_train = ames_update[['Street', 'Central_Air', 'Bedroom_AbvGr', 'Year_Built',
       'Mo_Sold', 'Lot_Area', 'First_Flr_SF', 'Second_Flr_SF', 'Fireplaces',
       'Garage_Area', 'Gr_Liv_Area']]

y_train = ames_update[['Sale_Price']]

# %%
X_train = pd.get_dummies(X_train, columns=['Street','Central_Air']).astype(int)
X_train.head()

# %% [markdown]
# #### Feature Selection

# %%
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

lm = LinearRegression()

model_cv = SequentialFeatureSelector(lm, n_features_to_select = "auto", direction = "backward", scoring = "neg_mean_squared_error", cv = 10)

model_cv.fit(X_train, y_train)

# %%
model_cv.get_feature_names_out()

# %% [markdown]
# ##### Classical Model with this Variable Selection

# %%
import statsmodels.formula.api as smf

final_model1 = smf.ols("Sale_Price ~ Bedroom_AbvGr + Year_Built + Mo_Sold + Lot_Area + First_Flr_SF + Second_Flr_SF + Fireplaces + Garage_Area", data = ames_update).fit()

final_model1.summary()

# %% [markdown]
# ##### ML Approach

# %%
final_model2 = SequentialFeatureSelector(lm, n_features_to_select = 8, direction = "backward", scoring = "neg_mean_squared_error")

final_model2.fit(X_train, y_train)

# %%
final_model2.get_feature_names_out()

# %% [markdown]
# ##### Regularization with Elastic Net

# %%
from sklearn.linear_model import ElasticNetCV

en_model = ElasticNetCV(cv = 10, l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], n_alphas = 100)

en_model.fit(X_train, y_train)

# %% [markdown]
# ##### Get the optimal Lambda and Alpha levels, to optimize balance between Ridge and Lasso Regressions, and the penalty terms
# - 0 = Ridge
# - 1 = Lasso

# %% [markdown]
# Lambda Value - set by looking at min SSE for many different values of lambda, then picking the optimal value

# %%
en_model.alpha_

# %% [markdown]
# Alpha Value

# %%
en_model.l1_ratio_

# %% [markdown]
# Build the plot, to Visualize the selection process

# %%
import numpy as np
from sklearn.linear_model import ElasticNet
from matplotlib import pyplot as plt
import seaborn as sns

n_alphas = 100
alphas3 = np.logspace(1, 8, n_alphas)

coefs3 = []
for a in alphas3:
    en = ElasticNet(alpha = a, l1_ratio=0.9)
    en.fit(X_train, y_train)
    coefs3.append(en.coef_)

# %%
    
plt.cla()
ax3 = plt.gca()

ax3.plot(alphas3, coefs3)
ax3.set_xscale("log")
plt.xlabel("alpha")
plt.ylabel("weights")
plt.title("ElasticNet coefficients as a function of the regularization")
plt.axis("tight")

# %% [markdown]
# ### i.Linear and Additive Models

# %% [markdown]
# #### 1.General Additive Model
# - can be used for CAT and CONT target variables
# - used to add multiple non-linear functions together

# %% [markdown]
# ##### A. Piecewise Linear Regression

# %%
cement = pd.read_csv("cement.csv")
cement.head()

# %%
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import seaborn as sns

cement_lm = smf.ols("STRENGTH ~ RATIO + X2STAR", data = cement).fit()

cement_lm.summary()

# %% [markdown]
# Visual for the Piecewise Linear Regression

# %%
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x = 'RATIO', y='STRENGTH', data = cement, ax = ax)

x_pred = cement['RATIO']
y_pred = cement_lm.fittedvalues

sns.lineplot(x = x_pred, y = y_pred, ax = ax)

plt.show()

# %% [markdown]
# Visual for a Piece-wise, Discontinuous Extension

# %%
cement_lm = smf.ols("STRENGTH ~ RATIO + X2STAR + X2", data = cement).fit()

cement_lm.summary()

# %%
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x = 'RATIO', y='STRENGTH', data = cement, ax = ax)

x_pred = cement['RATIO']
y_pred = cement_lm.fittedvalues

sns.lineplot(x = x_pred, y = y_pred, ax = ax)

plt.show()

# %% [markdown]
# ##### B. MARS
# - this has to be done with an R wrapper in Python, Python library has not been updated since 2017

# %%


# %% [markdown]
# ##### C. Smoothing Splines

# %%
### code in ML_hw nb

# %% [markdown]
# #### 2.Random Forest
# - can be used for CAT and CONT target variables
# - voting ensemble of many, many trees
# - dfs: amesupdate, X_train, y_train

# %%
X_train.head()

# %%
y_train.head()

# %%
from sklearn.ensemble import RandomForestRegressor

rf_ames = RandomForestRegressor(n_estimators = 100,
                                  random_state = 12345,
                                  oob_score = True)

rf_ames.fit(X_train, y_train.iloc[:,0])

# %%
rf_ames.oob_score_

# %% [markdown]
# Variable Importance

# %%
from matplotlib import pyplot as plt
import seaborn as sns

forest_importances = pd.Series(rf_ames.feature_importances_, index = rf_ames.feature_names_in_)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax = ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()

# %% [markdown]
# Parameter Tuning

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True], ##enable this
    'max_features': [3, 4, 5, 6, 7], ##number of features
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800] ##number of trees
}

rf = RandomForestRegressor(random_state = 12345)

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 10)

grid_search.fit(X_train, y_train.iloc[:,0])

# %%
grid_search.best_params_

# %% [markdown]
# Variable Selection

# %%
import numpy as np

X_train_r = X_train

X_train_r['random'] = np.random.normal(0, 1, 2930)

# %% [markdown]
# Compare variable against a random number. If random number outperforms

# %% [markdown]
# ### ii.Gradient Boosting Models

# %% [markdown]
# ##### 1.XGBoost

# %%
pip install xgboost

# %% [markdown]
# Initialization

# %%
from xgboost import XGBRegressor

xgb_ames = XGBRegressor(n_estimators = 50, #number of trees,
                        subsample = 0.5, #stochastic gradient descent,
                        random_state = 12345)

xgb_ames.fit(X_train, y_train)

# %% [markdown]
# Parameter Tuning

# %%
param_grid = {
    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'eta': [0.1, 0.15, 0.2, 0.25, 0.3],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'subsample': [0.25, 0.5, 0.75, 1]
}

xgb = XGBRegressor()

grid_search = GridSearchCV(estimator = xgb, param_grid = param_grid, cv = 10)

grid_search.fit(X_train, y_train)

# %%
grid_search.best_params_

# %% [markdown]
# Variable Importance and Selection

# %%
xgb_ames = XGBRegressor(n_estimators = 50,
                        subsample = 0.75,
                        eta = 0.15,
                        max_depth = 5,
                        random_state = 12345)

xgb_ames.fit(X_train, y_train)

# %% [markdown]
# Visual 1 - Var Importance

# %%
forest_importances = pd.Series(xgb_ames.feature_importances_, index = xgb_ames.feature_names_in_)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax = ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()

# %% [markdown]
# Visual 2 - Var Importance

# %%
import xgboost

xgboost.plot_importance(xgb_ames, importance_type = 'cover')

plt.show()

# %% [markdown]
# Adding the Random Variable to Baseline variable importance

# %%
import numpy as np

X_train_r = X_train

X_train_r['random'] = np.random.normal(0, 1, 2930)

# %%
xgb_ames = XGBRegressor(n_estimators = 50,
                        subsample = 0.75,
                        eta = 0.15,
                        max_depth = 5,
                        random_state = 12345)

xgb_ames.fit(X_train_r, y_train)

# %%
forest_importances = pd.Series(xgb_ames.feature_importances_, index = xgb_ames.feature_names_in_)
forest_importances = forest_importances.sort_values(ascending=False)

colors = ['red' if feature == 'random' else 'blue' for feature in forest_importances.index]

fig, ax = plt.subplots()
forest_importances.plot.bar(ax = ax, color=colors)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

plt.show()

# %%
import xgboost as xgb
import pandas as pd

# Assuming `xgb_ins` is your trained XGBoost model instance

booster = xgb_ames.get_booster()

# Get feature importance by gain, coverage, and frequency
importance_types = ['weight', 'gain', 'cover']  # weight: frequency, gain, cover: coverage

# Collect each importance type into a dictionary
importance_dict = {imp_type: booster.get_score(importance_type=imp_type) for imp_type in importance_types}

# Convert dictionaries to DataFrames, then merge them to get all values in a single DataFrame
importance_df = pd.DataFrame.from_dict(importance_dict)

# Rename columns for clarity
importance_df.columns = ['Frequency', 'Gain', 'Coverage']

# Fill NaN values with 0 (in case some features have missing importance metrics)
importance_df = importance_df.fillna(0)

# Sort by 'Gain' or another metric if desired
importance_df = importance_df.sort_values(by='Gain', ascending=False)

print(importance_df)


# %% [markdown]
# Partial Dependence Plots

# %%
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

features = [forest_importances.index[0], forest_importances.index[1]]
PartialDependenceDisplay.from_estimator(xgb_ames, X_train, features=features, kind="average")
plt.show()


# %% [markdown]
# ##### 2.EBM: Explainable Boosting Machines
# - build the model one variable at a time
# - build additional models on the residual errors

# %%
pip install interpret

# %%
import interpret
from interpret.glassbox import ExplainableBoostingRegressor
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.blackbox import PartialDependence
from interpret import show

# %%


# Fit a glassbox model, like Explainable Boosting Machine (EBM), for demonstration
ebm = ExplainableBoostingClassifier(interactions=5)
ebm.fit(X_train, y_train)

# Use Partial Dependence for blackbox models, like an XGBoost or other complex models
pdp_global = ebm.explain_global(X_train)

# Visualize the partial dependence
show(pdp_global)


# %% [markdown]
# ### iii.Neural Networks

# %%
# Data needs to be scaled
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  
scaler.fit(X_train)

# %%
X_train_s = scaler.transform(X_train) 

# %%
pip install

# %%
y_train_flat = y_train.values.ravel()

# %%
from sklearn.neural_network import MLPRegressor

nn_ames = MLPRegressor(solver='lbfgs', alpha = 1e-5,
                       hidden_layer_sizes = (5,),
                       random_state = 12345)
                       
nn_ames.fit(X_train_s, y_train_flat)

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [3, 4, 5, 6, 7],
    'alpha': [0.00005, 0.0005],
    'solver': ['lbfgs']
}

nn = MLPRegressor(max_iter = 5000, random_state = 12345)

grid_search = GridSearchCV(estimator = nn, param_grid = param_grid, cv = 10)

grid_search.fit(X_train_s, y_train_flat)

# %%
grid_search.best_params_

# %%


# %% [markdown]
# ### iv.Naive Bayes

# %%
import pandas as pd

train_dummy = pd.get_dummies(ames_update, columns = ['Street', 'Central_Air'])

print(train_dummy)

# %%
train_dummy['Bonus'] = (train_dummy['Sale_Price']>175000).astype(int)

# %%
y_train_c = train_dummy['Bonus']

X_train_c = train_dummy.loc[:, train_dummy.columns != 'Bonus']

# %%
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train_c, y_train_c)

# %% [markdown]
# ### v. Model Agnostic Evaluation

# %% [markdown]
# ##### i. Permutation Importance - Global

# %%
pip install scikit-learn matplotlib pandas


# %%
from sklearn.ensemble import RandomForestRegressor

rf_ames = RandomForestRegressor(n_estimators = 100,
                                  random_state = 12345,
                                  max_features = 7,
                                  oob_score = True)

rf_ames.fit(X_train, y_train.values.flatten())

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

feature_names = [f"Feature {i}" for i in range(1, 11)]

# Permutation importance
result = permutation_importance(rf_ames, X_train, y_train, n_repeats=10, random_state=42, scoring='r2')

# Get importances and feature indices
importances = result.importances_mean
indices = np.argsort(importances)  # Sort feature indices by importance
sorted_feature_names = X_train.columns[indices]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, importances[indices], xerr=result.importances_std[indices], align='center', color='skyblue')
plt.xlabel("Mean Permutation Importance")
plt.title("Feature Importances via Permutation")
plt.tight_layout()
plt.show()

# %% [markdown]
# ##### i. Individual Conditional Expectation (ECF) - Local method

# %%
pip install scikit-explain

# %%
import skexplain

# %%



