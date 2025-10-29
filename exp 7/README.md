# Experiment 7: Implementation of K-Means Clustering from Scratch

## Objective
The objective of this experiment is to implement the K-Means clustering algorithm **from scratch using NumPy**, apply it to a **customer segmentation dataset**, and compare its performance and results with the implementation from **scikit-learn**.

---

## Dataset and Preprocessing
The dataset used is **Mall_Customers**, containing the following features:
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

### Preprocessing Steps
1. Checked for and handled missing values.
2. Standardized numerical features using `StandardScaler`.
3. Generated descriptive statistics and visualizations:
   - Histograms for distribution
   - Pairplots for relationships among features

---

## Implementation Details

### Steps Implemented
1. **Initialization**
   - Random centroid selection.
   - K-Means++ initialization (for improved convergence).

2. **Cluster Assignment**
   - Computed Euclidean distances from each data point to all centroids.
   - Assigned each point to its nearest centroid.

3. **Centroid Update**
   - Recomputed centroids as the mean of all points belonging to each cluster.

4. **Convergence Criteria**
   - Algorithm stopped when centroid shifts were below a defined tolerance or the maximum number of iterations was reached.

5. **Performance Metrics**
   - Inertia (sum of squared distances to centroids)
   - Number of iterations

---

## Determining Optimal Number of Clusters

### Elbow Method
- The **Elbow Method** was used to plot inertia vs. number of clusters (k).
- The “elbow point” where the curve bends sharply indicates the optimal k.

### Silhouette Analysis
- The **Silhouette Score** was computed for each k to validate the optimal number of clusters.
- The value of **k** with the highest silhouette score was chosen as the most suitable.

---

## Results and Visualization
- Implemented clustering using the optimal number of clusters.
- Displayed:
  - Final centroids
  - Cluster sizes
  - 2D and 3D visualizations (Age, Annual Income, Spending Score)
- Computed cluster-wise averages of each feature for business interpretation:
  - Example:  
    *Cluster 0: Young, high-income, high-spending customers*  
    *Cluster 2: Older, low-income, conservative spenders*

---

## Comparison with scikit-learn Implementation
Both implementations were run on the same dataset with identical parameters.

| Metric | Custom Implementation | scikit-learn |
|:-------|:----------------------|:--------------|
| Inertia | Comparable | Reference |
| Iterations | Slightly higher | Optimized |
| Runtime | Slower (Python/NumPy) | Faster (C-optimized) |
| Centroids | Nearly identical | Reference result |

Visual comparisons of both clustering results confirmed consistent cluster boundaries and behavior.

---

## Conclusion
- The custom NumPy-based K-Means implementation successfully replicated the logic and performance of scikit-learn’s version.
- K-Means++ initialization improved convergence speed and reduced inertia compared to random initialization.
- The experiment demonstrated effective **unsupervised learning** for **customer segmentation**, providing interpretable insights for business applications.

---

## Dependencies
- Python 3.x  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- scikit-learn  
- kneed (for automated elbow detection)

---

## Author
**Yug Bathla**  
3rd Year, B.Tech (CSE)  
Delhi Technological University
