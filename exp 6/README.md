# Experiment 6: K-Nearest Neighbors (KNN) Algorithm Implementation

## Overview
This experiment implements the K-Nearest Neighbors (KNN) classification algorithm from scratch using Python and NumPy. The implementation is tested on two datasets from the UCI Machine Learning Repository: the Iris dataset and the Wine dataset.


## Objective
- Implement the K-Nearest Neighbors algorithm from scratch without using sklearn's KNN classifier
- Understand the mathematics and mechanics behind the KNN algorithm
- Perform comprehensive exploratory data analysis (EDA) on datasets
- Evaluate model performance with different hyperparameters (k values)
- Compare performance across multiple datasets

## Datasets

### 1. Iris Dataset
- **Source**: UCI Machine Learning Repository (ID: 53)
- **Samples**: 150
- **Features**: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Characteristics**: Balanced, well-separated classes

### 2. Wine Dataset
- **Source**: UCI Machine Learning Repository (ID: 109)
- **Samples**: 178
- **Features**: 13 (chemical analysis results)
- **Classes**: 3 (Class 1, Class 2, Class 3)
- **Characteristics**: Slightly imbalanced, more complex feature space

## Implementation Details

### Core Algorithm Components

#### 1. **Euclidean Distance Function**
```python
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
```
Calculates the distance between two data points in n-dimensional space.

#### 2. **KNN Classifier Class**
The main classifier with three key methods:
- `fit(X_train, y_train)`: Stores training data
- `_predict(x)`: Predicts class for a single sample
- `predict(X_test)`: Predicts classes for multiple samples

**Algorithm Steps**:
1. Calculate distances from test point to all training points
2. Sort distances and select k nearest neighbors
3. Use majority voting among k neighbors to determine class
4. Return the most common class label

#### 3. **Train-Test Split**
Custom implementation that randomly splits data into training and testing sets with configurable test size and random seed.

#### 4. **Accuracy Calculation**
Simple metric to evaluate model performance:
```python
accuracy = (correct predictions) / (total predictions)
```

## Key Components

### Data Loading and Preprocessing
- Fetch datasets directly from UCI repository using `ucimlrepo`
- Convert to NumPy arrays for efficient computation
- Clean target labels (remove prefixes)
- Validate data integrity (check for missing values, data types)

### Exploratory Data Analysis (EDA)
Comprehensive analysis including:
- Dataset structure and shape
- Class distribution (bar plots, pie charts)
- Statistical summaries (mean, std, min, max)
- Feature distributions by class (histograms)
- Box plots for outlier detection
- Pairwise scatter plots for feature relationships
- Correlation heatmaps
- Feature separability analysis (between-class vs within-class variance)

### Model Evaluation
- **Basic Evaluation**: Train-test split (80-20)
- **Hyperparameter Tuning**: Testing k values [1, 3, 5, 7, 9, 11, 15]
- **Performance Metrics**: Accuracy, confusion matrices
- **Visualization**: Accuracy vs k-value plots

### Confusion Matrices
Confusion matrices are generated for both datasets showing:
- True Positives, False Positives
- True Negatives, False Negatives
- Per-class performance

## Visualizations

The notebook generates comprehensive visualizations:

1. **Class Distribution Plots**
   - Bar charts showing sample counts per class
   - Pie charts showing class proportions

2. **Feature Distribution Histograms**
   - Overlaid histograms for each feature by class
   - Helps identify discriminative features

3. **Box Plots**
   - Identify outliers and data spread
   - Compare feature ranges across classes

4. **Pairwise Scatter Plots**
   - Visualize feature relationships
   - Show class separability in 2D projections

5. **Correlation Heatmaps**
   - Identify correlated features
   - Helps understand feature redundancy

6. **Accuracy vs k-Value Plots**
   - Line plots showing how accuracy changes with k
   - Helps select optimal hyperparameter

7. **Confusion Matrices**
   - Visual representation of classification results
   - Identify which classes are confused

## Dependencies

```python
numpy              # Numerical computations
pandas             # Data manipulation
matplotlib         # Plotting and visualization
collections        # Counter for majority voting
ucimlrepo          # UCI dataset fetching
sklearn.metrics    # Confusion matrix (for visualization only)
```

### Using the KNN Classifier
```python
# Initialize classifier
knn = KNNClassifier(k=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = calculate_accuracy(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```


### Algorithm Insights
1. **Distance Metric**: Euclidean distance works well for numeric features
2. **k-Value Selection**: 
   - Low k (1-3): More sensitive to noise, less stable
   - Medium k (5-9): Good balance, often optimal
   - High k (>11): Over-smoothing, may miss local patterns
3. **Scalability**: Algorithm stores all training data (memory-intensive)

### Dataset-Specific Observations

#### Iris Dataset
- **Separability**: Classes are well-separated (especially Setosa)
- **Performance**: Consistently high accuracy (>96%)
- **Best Features**: Petal Length and Petal Width are most discriminative
- **Stability**: Performance stable across different k values

#### Wine Dataset
- **Complexity**: 13 features create complex decision space
- **Performance**: Moderate accuracy (~75%)
- **Key Features**: OD280/OD315 of diluted wines, Flavanoids, Proline show high separability
- **Sensitivity**: More sensitive to k-value choice
- **Improvement Potential**: Feature scaling/normalization could improve results

### Advantages of KNN
- ✅ Simple to understand and implement
- ✅ No training phase (lazy learner)
- ✅ Naturally handles multi-class problems
- ✅ Non-parametric (no assumptions about data distribution)

### Limitations of KNN
- ❌ Computationally expensive for large datasets (O(n) for each prediction)
- ❌ Memory-intensive (stores all training data)
- ❌ Sensitive to feature scaling
- ❌ Performance degrades with high-dimensional data (curse of dimensionality)
- ❌ Sensitive to noise and outliers

### Recommendations for Improvement
1. **Feature Scaling**: Standardize/normalize features before applying KNN
2. **Feature Selection**: Remove irrelevant or redundant features
3. **Distance Metrics**: Experiment with Manhattan, Minkowski distances
4. **Weighted Voting**: Weight neighbors by inverse distance
5. **Efficiency**: Use KD-trees or Ball trees for faster neighbor search
