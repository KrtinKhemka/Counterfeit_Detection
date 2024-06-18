import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt

dfmain = pd.read_csv('final_dataset.csv', encoding='latin1')
#Step 1: cluster
dfmain['date'] = pd.to_datetime(dfmain['date'], errors='coerce')
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
J = vectorizer.fit_transform(dfmain['review'])


wcss = []
for i in range(1,16):
    kmeans = MiniBatchKMeans(n_clusters=i, random_state=42, batch_size=100)
    kmeans.fit(J)
    wcss.append(kmeans.inertia_)


#Plot elblow curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, 16), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()