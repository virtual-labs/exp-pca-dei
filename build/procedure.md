### Procedure

The objective of this experiment is to determine whether a breast tumor is malignant or benign by applying Principal Component Analysis (PCA) to the Wisconsin Breast Cancer dataset. By transforming the original 30 features into a smaller set of principal components, the experiment aims to simplify the dataset, reduce redundancy, and improve visualization and model efficiency without significantly affecting classification performance.

**Step 1:** Import required libraries: pandas, numpy, matplotlib, seaborn, and required modules from scikit-learn.

**Step 2:** Load the dataset containing breast cancer diagnostic features and the diagnosis label. The dataset contains 569 samples and 32 columns. Out of these, one column is an ID, one is the target column called diagnosis (with values M for malignant and B for benign), and the remaining 30 columns are numerical features describing cell characteristics used for prediction of diagnosis.

**Step 3:** Perform exploratory data analysis (EDA).

**Step 4:** Check class distribution of the diagnosis variable to observe the number of benign and malignant samples.

**Step 5:** Define feature variables and target variable:
- Features (X): All diagnostic attributes
- Target (y): Diagnosis label

**Step 6:** Visualize feature correlations using a heatmap before applying PCA.

**Step 7:** Standardize the feature values using StandardScaler to achieve zero mean and unit variance.

**Step 8:** Split the dataset into training and testing sets using stratified sampling.

**Step 9:** Train a Logistic Regression model using the original (non-PCA) feature set.

**Step 10:** Evaluate the model performance before PCA using classification accuracy.

**Step 11:** Apply Principal Component Analysis (PCA) while retaining 95% of the total variance.

**Step 12:** Observe the reduced dimensionality after applying PCA.

**Step 13:** Train a Logistic Regression model using PCA-transformed features.

**Step 14:** Evaluate the model performance after PCA using classification accuracy.

**Step 15:** Compare model accuracy before and after PCA using a bar plot.

**Step 16:** Apply PCA with a fixed number of components to compute feature loadings.

**Step 17:** Visualize feature contributions to principal components using a heatmap.

**Step 18:** Analyze explained variance ratio to understand the contribution of principal components.