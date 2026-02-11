Principal Component Analysis (PCA) is a dimensionality reduction technique used in unsupervised learning to transform a dataset with many variables into a smaller set that still contains most of the essential information. PCA helps us get rid of non-important features. It helps in faster training and inference, and even data visualization becomes easier. High-dimensional data often includes correlated features, which increase computational complexity and make data interpretation difficult. PCA addresses this problem by transforming the original variables into a new set of uncorrelated variables, called principal components, while retaining as much of the original information as possible.

#### 1. Objective of PCA

The objective of PCA is to identify directions in the feature space along which the variance of the data is maximized. The diagram shows how a dataset with many correlated features is first represented in a high-dimensional space. PCA then identifies new axes called principal components that point in the directions of maximum data spread. The original data points are projected onto these new axes, resulting in a reduced-dimensional dataset that still preserves most of the important information.

#### 2. Mathematical Foundation

The mathematical foundation of PCA lies in linear algebra, specifically in eigenvectors and eigenvalues. Eigenvectors define the directions of maximum variance in the data, while eigenvalues indicate the amount of variance explained by each direction.

<div style="text-align: center;">
<img src="images/pca.jpg" alt="PCA Visualization" style="max-height: 300px; width: auto;">
</div>

For a square matrix A, an eigenvector v and its corresponding eigenvalue λ satisfy the equation:

<div style="text-align: center; margin: 15px 0;">
<span style="display: inline-block; padding: 10px 20px; border: 1px solid #ccc; background-color: #f9f9f9; font-style: italic;">Av = λv</span>
</div>

In PCA, eigenvectors and eigenvalues are derived from the covariance matrix of the data. Eigenvectors associated with larger eigenvalues correspond to more informative principal components. These eigenvectors are normalized to unit length before being used for projection.

#### 3. Dimensionality Reduction

By projecting the data onto a smaller number of important principal components, PCA reduces the number of features while keeping most of the useful information. Only the components that capture a large amount of variation in the data are selected. This way, PCA makes the dataset simpler and easier to work with without losing its essential patterns.

#### 4. Merits of Using PCA

- Makes large and complex data easier to handle by reducing the number of features
- Removes repeated or similar information from the dataset
- Helps models run faster and more efficiently

#### 5. Demerits of Using PCA

- New features created by PCA are hard to understand and explain
- Some useful information may be lost during reduction
- Does not work well when data has non-linear patterns

#### 6. Algorithm

1. **Step 1: Standardize the data:**
    - For each feature: `x_scaled = (x - mean) / standard_deviation`
    - Result: All features have mean=0, std=1
2. **Step 2: Compute the Covariance Matrix:**
    - `Cov(X) = (1/(n-1)) × XᵀX`
    - Matrix size: d × d (where d = number of features)
    - Element [i,j] = covariance between feature i and feature j
3. **Step 3: Calculate Eigenvalues and Eigenvectors:**
    - Solve: `Cov(X) × v = λ × v`
    - λ = eigenvalue (represents variance captured)
    - v = eigenvector (represents new axis direction)
4. **Step 4: Sort by Eigenvalues:**
    - Arrange eigenvectors in descending order of their eigenvalues
    - First eigenvector = direction of maximum variance
    - Second eigenvector = direction of second-most variance (orthogonal to first)
5. **Step 5: Calculate Explained Variance Ratio:**
    - For each component: `ratio = λᵢ / Σλ`
    - Cumulative ratio shows total variance captured by first k components
6. **Step 6: Select Top k Components:**
    - Choose k such that cumulative variance ≥ threshold (e.g., 95%)
    - **OR** choose k based on specific requirement
7. **Step 7: Create Projection Matrix:**
    - W = matrix with top k eigenvectors as columns
8. **Step 8: Transform Data:**
    - `X_reduced = X_standardized × W`
    - New data has k dimensions instead of original d dimensions

