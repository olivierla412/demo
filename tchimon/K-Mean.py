import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

# Load the data
raw_data = pd.read_csv('C:/Users/THONIEL/Downloads/[FreeCourseSite.com] Udemy - The Data Science Course 2019 Complete Data Science Bootcamp/38. Advanced Statistical Methods - K-Means Clustering/3.2 Countries_exercise.csv.csv')

data = raw_data.copy()

plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()

# features
x = data.iloc[:,1:3]
x

# clustering
kmeans = KMeans(7)
kmeans.fit(x)

# results
identified_clusters = kmeans.fit_predict(x)
identified_clusters

# ADD result to table
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters

# plot
plt.scatter(data['Longitude'], data['Latitude'],c=data_with_clusters['Cluster'], cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()
