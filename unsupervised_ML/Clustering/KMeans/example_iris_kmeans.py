import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans as SklearnKM

# Path setup to root for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from unsupervised_ML.Clustering.KMeans.k_means import KMeans

def run_kmeans_iris():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    X = df[['petal length (cm)', 'petal width (cm)']].values
    
    # 2. Preprocessing
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("--- Iris Petal Clustering (KMeans) ---")
    
    # 3. Custom Model Training & Elbow Plot
    k_rng = range(1, 10)
    sse = []
    
    for k in k_rng:
        km = KMeans(n_clusters=k, max_iters=100, random_state=42)
        km.fit(X_scaled)
        sse.append(km.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(k_rng, sse, marker='o', color='blue')
    plt.xlabel("Number of Clusters (K)"); plt.ylabel("Sum of Squared Error"); plt.title("Elbow Plot For Iris Petal Features")
    plt.grid(True)
    plt.savefig(os.path.join(current_dir, "elbow_plot.png"))
    
    # 4. Form optimal clusters (K=3)
    model = KMeans(n_clusters=3, max_iters=200, random_state=42)
    model.fit(X_scaled)
    y_predicted = model.labels_
    centroids = model.centroids
    
    print(f"Custom model optimal inertia (K=3): {model.inertia_:.4f}")
    
    # 5. Validation with Sklearn
    sk_model = SklearnKM(n_clusters=3, max_iter=200, random_state=42, n_init='auto').fit(X_scaled)
    print(f"\n--- Sklearn Validation ---")
    print(f"Sklearn Inertia for K=3: {sk_model.inertia_:.4f}")
    
    if np.isclose(model.inertia_, sk_model.inertia_, rtol=1e-01):
        print("✅ Success! Custom model matches Sklearn results closely.")
        
    # 6. Save Visualization
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'blue']
    for i in range(3):
        cluster_points = X_scaled[y_predicted == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}')
        
    plt.scatter(centroids[:, 0], centroids[:, 1], color='purple', marker='*', s=250, label='Centroids')
    plt.xlabel("Petal Length (scaled)"); plt.ylabel("Petal Width (scaled)"); plt.title("K-Means Clustering on Iris (K=3)")
    plt.legend(); plt.grid(True)
    
    plt.savefig(os.path.join(current_dir, "cluster_plot.png"))
    print(f"Visualizations saved to: {current_dir}")

if __name__ == '__main__':
    run_kmeans_iris()
