# %% [markdown]
# #### Clustering Lab:
# 

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np
import seaborn as sns

# %%
import rpy2.robjects as robjects
from rpy2.robjects import r

# Load the RData file
r['load']("TeenSNS4.RData")

# List the objects in the R environment to check what's loaded
print(r['ls']())

# %%
import pandas as pd
from rpy2.robjects import pandas2ri

# Activate the automatic conversion between R and pandas
pandas2ri.activate()

# Assuming 'my_data' is the name of an R dataframe stored in the RData file
r_dataframe = robjects.r['teens4']

# Convert the R dataframe to a pandas dataframe
py_dataframe = pandas2ri.rpy2py(r_dataframe)

df = py_dataframe

# Now 'py_dataframe' is a pandas DataFrame in Python
df.head()

# %%
df.shape

# %%
df.describe()

# %%
df.nunique().sort_values(ascending=False)

# %%
df.friends.max()

# %%
df.head(25)

# %%
# check for nulls
df.isna().mean().sort_values(ascending=False)

# %%
df['gender'].unique()

# %%
len(df[df['gender']==-2147483648])

# %%
df.columns

# %%
df.gender.dtype

# %%
df['gender'] = df['gender'].replace({-2147483648 : 3})

# %%
df['gender'].unique()

# %%
plt.hist(df.drugs)

# %%
fact_cols = ['gradyear', 'gender', 'age']
content_cols = ['friends', 'basketball', 'football',
       'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading',
       'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot', 'kissed',
       'dance', 'band', 'marching', 'music', 'rock', 'god', 'church', 'jesus',
       'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes',
       'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']

for category in fact_cols:
    for feature in content_cols:
        sns.scatterplot(x=category, y=feature, data=df)
        plt.show()

# %% [markdown]
# #### Age and Gender EDA

# %%
fact_cols = ['gradyear', 'gender', 'age']
content_cols = ['friends', 'basketball', 'football',
       'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading',
       'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot', 'kissed',
       'dance', 'band', 'marching', 'music', 'rock', 'god', 'church', 'jesus',
       'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes',
       'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']

for feature in content_cols:
    sns.scatterplot(x='age', y=feature, hue='gender', data=df)
    plt.show()

# %%
fact_cols = ['gradyear', 'gender', 'age']
content_cols = ['friends', 'basketball', 'football',
       'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading',
       'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot', 'kissed',
       'dance', 'band', 'marching', 'music', 'rock', 'god', 'church', 'jesus',
       'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes',
       'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']

for feature in content_cols:
    sns.scatterplot(x='age', y=feature, hue='gradyear', data=df)
    plt.show()

# %%
fact_cols = ['gradyear', 'gender', 'age']
content_cols = ['friends', 'basketball', 'football',
       'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading',
       'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot', 'kissed',
       'dance', 'band', 'marching', 'music', 'rock', 'god', 'church', 'jesus',
       'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes',
       'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']

for column in fact_cols:
    plt.hist(x=column, data=df)
    plt.show()

# %%
df.head()

# %% [markdown]
# #### Takeaways from EDA:
# - there are some large outliers in a few categories (church, god, rock, music, and more), so we will use a scaling method that is robust in outlier handling: RobustScaler in Python
# - age is continuous, to three decimal places, so let's round it and then get dummies for the categorical variables.
# - distribution across genders is lopsided. 1=~80%, 2=~15%, 3=~5%

# %%
df.age.head()

# %%
df.age.head()

# %% [markdown]
# #### Preprocessing and Clustering:
# 1. One hot encode categorical variables
# 2. Scale continuous variables
# 3. PCA for feature selection/reduction
# 4. Cluster with K-Means, one hierarchical, and DBScan

# %%
# One hot encoding for categorical vars
df_encoded = pd.get_dummies(df, columns=['gender', 'gradyear']).astype(int)
df_encoded

# %%
# Scaling
from sklearn.preprocessing import RobustScaler
import pandas as pd

columns_to_scale = ['age','friends', 'basketball', 'football',
       'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading',
       'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot', 'kissed',
       'dance', 'band', 'marching', 'music', 'rock', 'god', 'church', 'jesus',
       'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes',
       'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']

# Apply RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df_encoded[columns_to_scale])

# Convert back to DataFrame for easier interpretation
df_scaled = pd.DataFrame(df_scaled, columns=[columns_to_scale])

# %%
df_encoded.columns

# %%
df_cat = df_encoded[['gender_1','gender_2', 'gender_3', 'gradyear_2006', 'gradyear_2007','gradyear_2008', 'gradyear_2009']]

# %%
df_cat

# %%
df_scaled

# %%
df_cat

# %%
df_scaled = df_scaled.reset_index(drop=True)
df_cat = df_cat.reset_index(drop=True)

df_combined = pd.concat([df_scaled, df_cat], axis=1)

df_combined

# %% [markdown]
# #### Preprocessing is complete - let's take a shot a feature engineering/reduction with Factor Analysis for Mixed Data (FAMD)
# - this is because our data is not all continuous and normalyu distributed
# - this will help blunt the distortions the binary variables would have on the PCA

# %%
df_combined.columns

# %%
df_combined.columns = df_combined.columns.astype(str)

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Fit PCA
pca = PCA(n_components=2)  # Reduce to 2 components for simplicity
pca.fit(df_combined)

# Get the loading scores (component weights)
loading_scores = pca.components_

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

print("Loading scores (components):")
print(loading_scores)


# %%
print("\nExplained variance ratio for each component:")
print(explained_variance_ratio)


# %%

# Feature importance for the first principal component
pc1_loading_scores = pd.Series(loading_scores[0], index=df_combined.columns)
print("\nFeature importance for the first component:")
print(pc1_loading_scores.sort_values(ascending=False))

# %%

# Feature importance for the second principal component
pc2_loading_scores = pd.Series(loading_scores[1], index=df_combined.columns)
print("\nFeature importance for the second component:")
print(pc2_loading_scores.sort_values(ascending=False))

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Assume df_combined is your dataframe

# Fit PCA
pca = PCA(n_components=4)  # Reduce to 2 components for simplicity
pca.fit(df_combined)

# Get the loading scores (component weights)
loading_scores = pca.components_

# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Print loading scores and explained variance ratio
print("Loading scores (components):")
print(loading_scores)

# Display explained variance ratio in percentage
print("\nExplained variance ratio (percent):")
print(explained_variance_ratio * 100)


# %% [markdown]
# #### Percent of variation explained
# - how many principal components? that will explain majority of variation
# - the Principal Components captured farily little of the variation. We are going to take a different route

# %% [markdown]
# #### Takeaway from PCA:
# - both PCA clusters ranked 'blonde' the highest. Upon investigation, blonde had a very significant outlier. I think the PCAs are indicating features with the largest variance, which are very skewed by a few outliers.

# %% [markdown]
# #### 1. K Means

# %%

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

clus_py=KMeans(n_clusters=6, random_state=5687, n_init=25).fit(df_combined)

clus_py.labels_

# %%
clus_py

# %%
inertias = []
silhouette_coefficients = []
   
K=range(2,10)
for k in K:
    kmean1 = KMeans(n_clusters=k).fit(df_combined)
    kmean1.fit(df_combined)
    inertias.append(kmean1.inertia_)
    score = silhouette_score(df_combined, kmean1.labels_)
    silhouette_coefficients.append(score)

# %%
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

# %%
plt.plot(K, silhouette_coefficients, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette Coefficient')
plt.title('The Silhouette Method')
plt.show()
  

# %% [markdown]
# ##### Using 6 Clusters with K Means, add those cluster labels back to the rows of the Dataframe

# %%
from sklearn.cluster import KMeans
import pandas as pd

# Assume df is your original dataset
# Example: Performing K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=6)
kmeans.fit(df_combined)

# Get the cluster labels
cluster_labels = kmeans.labels_


# %% [markdown]
# ##### Sum of Squares for the Cluster

# %%
sum_of_squares = kmeans.inertia_
sum_of_squares

# %%
df_combined.head()

# %% [markdown]
# #### Takeaways from K-Means:
# - optimal number of clusters from Elbow Plot looks to be in the 4-8 range
# - optimal number from the Silhouette method looks like 2 or 6
# - going with 6, b/c that value is indicated from both plots.

# %% [markdown]
# #### 2. Hierarchical Clustering
# - going with Euclidean distance as our distance metric

# %%
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

linkage_data = linkage(df_combined, method='ward', metric='euclidean')
dendrogram(linkage_data)

# %%
# Get cluster labels using a threshold, for example by defining the number of clusters (e.g., 4)
# criterion='maxclust' means to form the desired number of clusters
cluster_labels = fcluster(linkage_data, t=4, criterion='maxclust')

# Add the cluster labels back to the original dataset
df['hc_cluster'] = cluster_labels

# Display the updated DataFrame
df.head()

# %% [markdown]
# #### 3. DBScan
# - we will use 4 as our number of density reaachable points here.

# %%
from sklearn.cluster import DBSCAN
from collections import Counter


db_py = DBSCAN(eps=1.2,min_samples=42).fit(df_combined)

db_py.labels_ 

# %%
set(db_py.labels_)

# %%
Counter(db_py.labels_)

# %%
df_combined.columns

# %% [markdown]
# #### Takeaways:
# - not sure what this output means. Need a visual for interpretability, or a table.

# %% [markdown]
# #### Now that the three clusters are added to the original DataFrame, let's do some exploration of commonalities within each cluster

# %%
df['db_scan'] = db_py.labels_
df['k_means_cluster'] = cluster_labels

# %%
df.head()

# %%
df.columns

# %%
cats = ['gradyear', 'gender','hc_cluster', 'db_scan', 'k_means_cluster']

quants = ['age', 'friends', 'basketball', 'football',
       'soccer', 'softball', 'volleyball', 'swimming', 'cheerleading',
       'baseball', 'tennis', 'sports', 'cute', 'sex', 'sexy', 'hot', 'kissed',
       'dance', 'band', 'marching', 'music', 'rock', 'god', 'church', 'jesus',
       'bible', 'hair', 'dress', 'blonde', 'mall', 'shopping', 'clothes',
       'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs']

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import matplotlib.pyplot as plt

pivoted_kmeans_df = df.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'sum')
plt.figure(figsize=(12,6))

pivoted_kmeans_df.T.plot(kind='bar', stacked=True)

# %%
import matplotlib.pyplot as plt

pivoted_kmeans_df = df.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'mean')

plt.figure(figsize=(12,6))

pivoted_kmeans_df.T.plot(kind='bar', stacked=True)


# %% [markdown]
# ##### Cluster 1

# %%
# get the cluster counts
len(df[df.k_means_cluster == 1])

# %%
cluster_1 = df[df.k_means_cluster == 1]
cluster_1.head()

# %%
pivoted_kmeans_df_1 = cluster_1.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'sum')
plt.figure(figsize=(12,6))

pivoted_kmeans_df_1.T.plot(kind='bar')

# %%
pivoted_kmeans_df_1 = cluster_1.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'mean')
plt.figure(figsize=(12,6))

pivoted_kmeans_df_1.T.plot(kind='bar')

# %%
pivoted_kmeans_df_1_drop = pivoted_kmeans_df_1.drop(columns = ['age','friends'])
pivoted_kmeans_df_1_drop.columns

# %%
plt.figure(figsize=(12,6))

pivoted_kmeans_df_1_drop.T.plot(kind='bar')

# %%
plt.hist(cluster_1.gender)

# %%
sns.violinplot(x='k_means_cluster', y='age', data=cluster_1, inner="quartile")

# %% [markdown]
# ##### Takeaways for Cluster 1:
# - 'music' is the number one interest by far
# - shopping, god, and dance are all roughly tied for second
# - this cluster is the largest. roughly 90% of the dataset, 27k
# - gender split: same as underlying data
# - age split: concentrated between 16-19 years
# - cluster title: Omnivores

# %% [markdown]
# ##### Cluster 2

# %%
len(df[df.k_means_cluster == 2])

# %%
cluster_2 = df[df.k_means_cluster == 2]

pivoted_kmeans_df_2 = cluster_2.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'sum')
plt.figure(figsize=(12,6))

pivoted_kmeans_df_2.T.plot(kind='bar')

# %%
pivoted_kmeans_df_2 = cluster_2.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'mean')
plt.figure(figsize=(12,6))

pivoted_kmeans_df_2.T.plot(kind='bar')

# %%
pivoted_kmeans_df_2_drop = pivoted_kmeans_df_2.drop(columns = ['age','friends'])
pivoted_kmeans_df_2_drop.columns

# %%
plt.figure(figsize=(12,6))

pivoted_kmeans_df_2_drop.T.plot(kind='bar')

# %%
plt.hist(cluster_2.gender)

# %%
sns.violinplot(x='k_means_cluster', y='age', data=cluster_2, inner="quartile")

# %% [markdown]
# ##### Takeaways for Cluster 2:
# - god, music and church are all roughly tied for second
# - this cluster is small, with 612 members, or about 2% of the data.
# - gender split: same as underlying data
# - age split: concentrated between 17-19 years
# - cluster title: Holy Rollers

# %% [markdown]
# ##### Cluster 3

# %%
len(df[df.k_means_cluster == 3])

# %%
cluster_3 = df[df.k_means_cluster == 3]

pivoted_kmeans_df_3 = cluster_3.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'sum')
plt.figure(figsize=(12,6))

pivoted_kmeans_df_3.T.plot(kind='bar')

# %%
pivoted_kmeans_df_3 = cluster_3.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'mean')
plt.figure(figsize=(12,6))

pivoted_kmeans_df_3.T.plot(kind='bar')

# %%
pivoted_kmeans_df_3_drop = pivoted_kmeans_df_3.drop(columns = ['age','friends'])
pivoted_kmeans_df_3_drop.columns

# %%
plt.figure(figsize=(12,6))

pivoted_kmeans_df_3_drop.T.plot(kind='bar')

# %%
plt.hist(cluster_2.gender)

# %%
sns.violinplot(x='k_means_cluster', y='age', data=cluster_3, inner="quartile")

# %% [markdown]
# ##### Takeaways for Cluster 3:
# - music, hair, dance, band are the most popular terms
# - this cluster is small, with 2143 members, or about 7% of the data.
# - gender split: same as underlying data
# - age split: concentrated between 16-19 years
# - cluster title: Rockers and Ravers

# %% [markdown]
# #### Cluster 4

# %%
len(df[df.k_means_cluster == 4])

# %%
cluster_4 = df[df.k_means_cluster == 4]

# %%
pivoted_kmeans_df_4 = cluster_4.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'sum')
plt.figure(figsize=(12,6))

pivoted_kmeans_df_4.T.plot(kind='bar')

# %%
pivoted_kmeans_df_4 = cluster_4.pivot_table(index = 'k_means_cluster', values = quants, aggfunc = 'mean')
plt.figure(figsize=(12,6))

pivoted_kmeans_df_4.T.plot(kind='bar')

# %%
pivoted_kmeans_df_4_drop = pivoted_kmeans_df_4.drop(columns = ['age','friends'])
pivoted_kmeans_df_4_drop.columns

# %%
plt.figure(figsize=(12,6))

pivoted_kmeans_df_4_drop.T.plot(kind='bar')

# %%
plt.hist(cluster_4.gender)

# %%
sns.violinplot(x='k_means_cluster', y='age', data=cluster_4, inner="quartile")

# %%



