/* Main Logic for PCA Experiment Simulation */

/* 
 * Steps Data Configuration for PCA Analysis
 */
const PCA_VARIANCE_DATA = {
    1: { ev: 0.4286, cv: 0.4286 },
    2: { ev: 0.1838, cv: 0.6124 },
    3: { ev: 0.0915, cv: 0.7039 },
    4: { ev: 0.0639, cv: 0.7678 },
    5: { ev: 0.0532, cv: 0.8210 },
    6: { ev: 0.0398, cv: 0.8608 },
    7: { ev: 0.0316, cv: 0.8924 },
    8: { ev: 0.0217, cv: 0.9140 },
    9: { ev: 0.0149, cv: 0.9289 },
    10: { ev: 0.0130, cv: 0.9419 }
};

window.updatePCATestingPlot = function () {
    const val = parseInt(document.getElementById('pcTestSlider').value);
    const data = PCA_VARIANCE_DATA[val];

    document.getElementById('pcNumVal').innerText = val;
    document.getElementById('pcLabelText').innerText = `Principal Component: PC${val}`;
    document.getElementById('evRatioText').innerText = `Explained Variance Ratio: ${data.ev.toFixed(4)}`;
    document.getElementById('cvRatioText').innerText = `Cumulative Variance till PC${val}: ${data.cv.toFixed(4)}`;

    const img = document.getElementById('pcaCompImg');
    if (img) {
        img.src = `images/pca_comp${val}.png`;
        img.alt = `Top 10 Feature Contributions to PC${val}`;
    }
};

const stepsData = [
    {
        id: 'import_libraries',
        title: 'Importing Libraries',
        blocks: [
            {
                code: `# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ipywidgets as widgets
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ipywidgets import interact, IntSlider
print("Libraries Imported")`,
                output: `<div class="output-success">Libraries Imported</div>`
            }
        ]
    },
    {
        id: 'loading_dataset',
        title: 'Loading Dataset',
        blocks: [
            {
                code: `# Load the Breast cancer Wisconsin dataset
df = pd.read_csv('Breast_cancer_Wisconsin_data.csv')
print("Dataset loaded successfully")`,
                output: `<div class="output-text">Dataset loaded successfully</div>`
            }
        ]
    },
    {
        id: 'data_analysis',
        title: 'Data Analysis',
        blocks: [
            {
                code: `<div class="output-success"># Display the first 5 rows of the dataset</div>
df.head()`,
                output: `<div class="table-wrapper">
<table class="data-table">
  <thead>
    <tr>
      <th></th><th>id</th><th>radius_mean</th><th>texture_mean</th><th>perimeter_mean</th><th>area_mean</th><th>smoothness_mean</th><th>compactness_mean</th><th>concavity_mean</th><th>concave points_mean</th><th>symmetry_mean</th><th>...</th><th>texture_worst</th><th>perimeter_worst</th><th>area_worst</th><th>smoothness_worst</th><th>compactness_worst</th><th>concavity_worst</th><th>concave points_worst</th><th>symmetry_worst</th><th>fractal_dimension_worst</th><th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>842302</td><td>17.99</td><td>10.38</td><td>122.80</td><td>1001.0</td><td>0.11840</td><td>0.27760</td><td>0.3001</td><td>0.14710</td><td>0.2419</td><td>...</td><td>17.33</td><td>184.60</td><td>2019.0</td><td>0.1622</td><td>0.6656</td><td>0.7119</td><td>0.2654</td><td>0.4601</td><td>0.11890</td><td>M</td></tr>
    <tr><td>1</td><td>842517</td><td>20.57</td><td>17.77</td><td>132.90</td><td>1326.0</td><td>0.08474</td><td>0.07864</td><td>0.0869</td><td>0.07017</td><td>0.1812</td><td>...</td><td>23.41</td><td>158.80</td><td>1956.0</td><td>0.1238</td><td>0.1866</td><td>0.2416</td><td>0.1860</td><td>0.2750</td><td>0.08902</td><td>M</td></tr>
    <tr><td>2</td><td>84300903</td><td>19.69</td><td>21.25</td><td>130.00</td><td>1203.0</td><td>0.10960</td><td>0.15990</td><td>0.1974</td><td>0.12790</td><td>0.2069</td><td>...</td><td>25.53</td><td>152.50</td><td>1709.0</td><td>0.1444</td><td>0.4245</td><td>0.4504</td><td>0.2430</td><td>0.3613</td><td>0.08758</td><td>M</td></tr>
    <tr><td>3</td><td>84348301</td><td>11.42</td><td>20.38</td><td>77.58</td><td>386.1</td><td>0.14250</td><td>0.28390</td><td>0.2414</td><td>0.10520</td><td>0.2597</td><td>...</td><td>26.50</td><td>98.87</td><td>567.7</td><td>0.2098</td><td>0.8663</td><td>0.6869</td><td>0.2575</td><td>0.6638</td><td>0.17300</td><td>M</td></tr>
    <tr><td>4</td><td>84358402</td><td>20.29</td><td>14.34</td><td>135.10</td><td>1297.0</td><td>0.10030</td><td>0.13280</td><td>0.1980</td><td>0.10430</td><td>0.1809</td><td>...</td><td>16.67</td><td>152.20</td><td>1575.0</td><td>0.1374</td><td>0.2050</td><td>0.4000</td><td>0.1625</td><td>0.2364</td><td>0.07678</td><td>M</td></tr>
  </tbody>
</table>
</div>
<div class="output-text">5 rows × 32 columns</div>`
            },
            {
                code: `<div class="output-success">#Displays the last five rows of the dataset</div>
df.tail()`,
                output: `<div class="table-wrapper">
<table class="data-table">
  <thead>
    <tr>
      <th></th><th>id</th><th>radius_mean</th><th>texture_mean</th><th>perimeter_mean</th><th>area_mean</th><th>smoothness_mean</th><th>compactness_mean</th><th>concavity_mean</th><th>concave points_mean</th><th>symmetry_mean</th><th>...</th><th>texture_worst</th><th>perimeter_worst</th><th>area_worst</th><th>smoothness_worst</th><th>compactness_worst</th><th>concavity_worst</th><th>concave points_worst</th><th>symmetry_worst</th><th>fractal_dimension_worst</th><th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>564</td><td>926424</td><td>21.56</td><td>22.39</td><td>142.00</td><td>1479.0</td><td>0.11100</td><td>0.11590</td><td>0.24390</td><td>0.13890</td><td>0.1726</td><td>...</td><td>26.40</td><td>166.10</td><td>2027.0</td><td>0.14100</td><td>0.21130</td><td>0.4107</td><td>0.2216</td><td>0.2060</td><td>0.07115</td><td>M</td></tr>
    <tr><td>565</td><td>926682</td><td>20.13</td><td>28.25</td><td>131.20</td><td>1261.0</td><td>0.09780</td><td>0.10340</td><td>0.14400</td><td>0.09791</td><td>0.1752</td><td>...</td><td>38.25</td><td>155.00</td><td>1731.0</td><td>0.11660</td><td>0.19220</td><td>0.3215</td><td>0.1628</td><td>0.2572</td><td>0.06637</td><td>M</td></tr>
    <tr><td>566</td><td>926954</td><td>16.60</td><td>28.08</td><td>108.30</td><td>858.1</td><td>0.08455</td><td>0.10230</td><td>0.09251</td><td>0.05302</td><td>0.1590</td><td>...</td><td>34.12</td><td>126.70</td><td>1124.0</td><td>0.11390</td><td>0.30940</td><td>0.3403</td><td>0.1418</td><td>0.2218</td><td>0.07820</td><td>M</td></tr>
    <tr><td>567</td><td>927241</td><td>20.60</td><td>29.33</td><td>140.10</td><td>1265.0</td><td>0.11780</td><td>0.27700</td><td>0.35140</td><td>0.15200</td><td>0.2397</td><td>...</td><td>39.42</td><td>184.60</td><td>1821.0</td><td>0.16500</td><td>0.86810</td><td>0.9387</td><td>0.2650</td><td>0.4087</td><td>0.12400</td><td>M</td></tr>
    <tr><td>568</td><td>92751</td><td>7.76</td><td>24.54</td><td>47.92</td><td>181.0</td><td>0.05263</td><td>0.04362</td><td>0.00000</td><td>0.00000</td><td>0.1587</td><td>...</td><td>30.37</td><td>59.16</td><td>268.6</td><td>0.08996</td><td>0.06444</td><td>0.0000</td><td>0.0000</td><td>0.2871</td><td>0.07039</td><td>B</td></tr>
  </tbody>
</table>
</div>
<div class="output-text">5 rows × 32 columns</div>`
            },
            {
                code: `<div class="output-success"># Check dataset shape</div>
df.shape`,
                output: `<div class="output-text">(569, 32)</div>`
            },
            {
                code: `<div class="output-success"># Displays summary of dataset, including column names, data types, and non-null counts.</div>
df.info()`,
                output: `<div class="output-text">&lt;class 'pandas.core.frame.DataFrame'&gt;</div>
<div class="output-text">RangeIndex: 569 entries, 0 to 568</div>
<div class="output-text">Data columns (total 32 columns):</div>
<div class="table-wrapper">
<table class="data-table">
  <thead>
    <tr>
      <th>#</th><th>Column</th><th>Non-Null Count</th><th>Dtype</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0</td><td>id</td><td>569 non-null</td><td>int64</td></tr>
    <tr><td>1</td><td>radius_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>2</td><td>texture_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>3</td><td>perimeter_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>4</td><td>area_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>5</td><td>smoothness_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>6</td><td>compactness_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>7</td><td>concavity_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>8</td><td>concave points_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>9</td><td>symmetry_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>10</td><td>fractal_dimension_mean</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>11</td><td>radius_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>12</td><td>texture_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>13</td><td>perimeter_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>14</td><td>area_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>15</td><td>smoothness_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>16</td><td>compactness_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>17</td><td>concavity_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>18</td><td>concave points_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>19</td><td>symmetry_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>20</td><td>fractal_dimension_se</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>21</td><td>radius_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>22</td><td>texture_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>23</td><td>perimeter_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>24</td><td>area_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>25</td><td>smoothness_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>26</td><td>compactness_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>27</td><td>concavity_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>28</td><td>concave points_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>29</td><td>symmetry_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>30</td><td>fractal_dimension_worst</td><td>569 non-null</td><td>float64</td></tr>
    <tr><td>31</td><td>diagnosis</td><td>569 non-null</td><td>object</td></tr>
  </tbody>
</table>
</div>
<div class="output-text">dtypes: float64(30), int64(1), object(1)</div>
<div class="output-text">memory usage: 142.4+ KB</div>`
            },
            {
                code: `<div class="output-success"># Statistical summary</div>
df.describe()`,
                output: `<div class="table-wrapper">
<table class="data-table">
  <thead>
    <tr>
      <th></th><th>id</th><th>radius_mean</th><th>texture_mean</th><th>perimeter_mean</th><th>area_mean</th><th>smoothness_mean</th><th>compactness_mean</th><th>concavity_mean</th><th>concave points_mean</th><th>symmetry_mean</th><th>...</th><th>radius_worst</th><th>texture_worst</th><th>perimeter_worst</th><th>area_worst</th><th>smoothness_worst</th><th>compactness_worst</th><th>concavity_worst</th><th>concave points_worst</th><th>symmetry_worst</th><th>fractal_dimension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>count</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>...</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td><td>569.000000</td></tr>
    <tr><td>mean</td><td>3.037183e+07</td><td>14.127292</td><td>19.289649</td><td>91.969033</td><td>654.889104</td><td>0.096360</td><td>0.104341</td><td>0.088799</td><td>0.048919</td><td>0.181162</td><td>...</td><td>16.269190</td><td>25.677223</td><td>107.261213</td><td>880.583128</td><td>0.132369</td><td>0.254265</td><td>0.272188</td><td>0.114606</td><td>0.290076</td><td>0.083946</td></tr>
    <tr><td>std</td><td>1.250206e+08</td><td>3.524049</td><td>4.301036</td><td>24.298981</td><td>351.914129</td><td>0.014064</td><td>0.052813</td><td>0.079720</td><td>0.038803</td><td>0.027414</td><td>...</td><td>4.833242</td><td>6.146258</td><td>33.602542</td><td>569.356993</td><td>0.022832</td><td>0.157336</td><td>0.208624</td><td>0.065732</td><td>0.061867</td><td>0.018061</td></tr>
    <tr><td>min</td><td>8.670000e+03</td><td>6.981000</td><td>9.710000</td><td>43.790000</td><td>143.500000</td><td>0.052630</td><td>0.019380</td><td>0.000000</td><td>0.000000</td><td>0.106000</td><td>...</td><td>7.930000</td><td>12.020000</td><td>50.410000</td><td>185.200000</td><td>0.071170</td><td>0.027290</td><td>0.000000</td><td>0.000000</td><td>0.156500</td><td>0.055040</td></tr>
    <tr><td>25%</td><td>8.692180e+05</td><td>11.700000</td><td>16.170000</td><td>75.170000</td><td>420.300000</td><td>0.086370</td><td>0.064920</td><td>0.029560</td><td>0.020310</td><td>0.161900</td><td>...</td><td>13.010000</td><td>21.080000</td><td>84.110000</td><td>515.300000</td><td>0.116600</td><td>0.147200</td><td>0.114500</td><td>0.064930</td><td>0.250400</td><td>0.071460</td></tr>
    <tr><td>50%</td><td>9.060240e+05</td><td>13.370000</td><td>18.840000</td><td>86.240000</td><td>551.100000</td><td>0.095870</td><td>0.092630</td><td>0.061540</td><td>0.033500</td><td>0.179200</td><td>...</td><td>14.970000</td><td>25.410000</td><td>97.660000</td><td>686.500000</td><td>0.131300</td><td>0.211900</td><td>0.226700</td><td>0.099930</td><td>0.282200</td><td>0.080040</td></tr>
    <tr><td>75%</td><td>8.813129e+06</td><td>15.780000</td><td>21.800000</td><td>104.100000</td><td>782.700000</td><td>0.105300</td><td>0.130400</td><td>0.130700</td><td>0.074000</td><td>0.195700</td><td>...</td><td>18.790000</td><td>29.720000</td><td>125.400000</td><td>1084.000000</td><td>0.146000</td><td>0.339100</td><td>0.382900</td><td>0.161400</td><td>0.317900</td><td>0.092080</td></tr>
    <tr><td>max</td><td>9.113205e+08</td><td>28.110000</td><td>39.280000</td><td>188.500000</td><td>2501.000000</td><td>0.163400</td><td>0.345400</td><td>0.426800</td><td>0.201200</td><td>0.304000</td><td>...</td><td>36.040000</td><td>49.540000</td><td>251.200000</td><td>4254.000000</td><td>0.222600</td><td>1.058000</td><td>1.252000</td><td>0.291000</td><td>0.663800</td><td>0.207500</td></tr>
  </tbody>
</table>
</div>
<div class="output-text">8 rows × 31 columns</div>`
            },
            {
                code: `<div class="output-success">#Counts the frequency of each unique category</div>
df['diagnosis'].value_counts()`,
                output: `<div class="output-text">
<div style="font-weight: bold; margin-bottom: 5px;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;count</div>
<div style="font-weight: bold;">diagnosis</div>
<div>B&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;357</div>
<div>M&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;212</div>
<div style="margin-top: 5px;">Name: count, dtype: int64</div>
</div>`
            }
        ]
    },
    {
        id: 'preprocessing',
        title: 'Data Preprocessing',
        blocks: [
            {
                code: `<div class="output-success"># Separate features and target variable</div>
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
print("Feature shape:", X.shape)
print("Target shape:", y.shape)`,
                output: `<div class="output-text">Feature shape: (569, 30)</div>
<div class="output-text">Target shape: (569,)</div>`
            },
            {
                code: `<div class="output-success"># Visualize correlations among features before applying PCA</div>
plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap (Before PCA)")
plt.show()`,
                output: `<div style="text-align:center;">
    <h4 style="margin:0 0 10px 0;">Feature Correlation Heatmap (Before PCA)</h4>
    <img src="images/feature_corelation.png" alt="Feature Correlation Heatmap" style="width:100%; max-width:800px; border-radius:8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>`
            },
            {
                code: `<div class="output-success"># Standardize the features</div>
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardized using StandardScaler")`,
                output: `<div class="output-text">Features standardized using StandardScaler</div>`
            }
        ]
    },
    {
        id: 'model_training',
        title: 'Model Training',
        blocks: [
            {
                code: `<div class="output-success"># Data Splitting</div>
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print("Data has been Splitted")`,
                output: `<div class="output-text">Data has been Splitted</div>`
            },
            {
                code: `<div class="output-success">#Apply PCA, and visualize variance</div>
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("PCA applied with 95% variance retained.")`,
                output: `<div class="output-text">PCA applied with 95% variance retained.</div>`
            },
            {
                code: `<div class="output-success"># Model training using Logistic Regression</div>
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_before = accuracy_score(y_test, y_pred)
clf_pca = LogisticRegression(max_iter=2000)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_after = accuracy_score(y_test, y_pred_pca)
print("Model Training completed")`,
                output: `<div class="output-success">Model Training completed</div>`
            },

        ]
    },
    {
        id: 'model_evaluation',
        title: 'Model Evaluation',
        blocks: [
            {
                code: `<div class="output-success"># Compare classification accuracy before and after applying PCA</div>
print("Accuracy BEFORE PCA:", round(acc_before*100, 2), "%")
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print("Reduced Dimensions:", X_pca.shape[1])
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)
print("Accuracy AFTER PCA:", round(acc_after*100, 2), "%")
plt.figure(figsize=(6,4))
sns.barplot(
    x=['Before PCA', 'After PCA'],
    y=[acc_before, acc_after]
)
plt.ylabel("Accuracy")
plt.title("Classification Accuracy Comparison")
plt.ylim(0.9, 1.0)
plt.show()`,
                output: `<div class="output-text">Accuracy BEFORE PCA: 97.37 %</div>
<div class="output-text">Reduced Dimensions: 11</div>
<div class="output-text">Accuracy AFTER PCA: 97.37 %</div>
<div style="text-align:left; margin-top:10px;">
    <img src="images/classification_accuracy.png" alt="Classification Accuracy Comparison" style="width:100%; max-width:450px; border-radius:8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>`
            },
            {
                code: `<div class="output-success"># Apply PCA and compute feature loadings for each principal component</div>
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
loadings = pd.DataFrame( pca.components_.T, columns=[f"PC{i+1}" for i in range(10)], index=X.columns)
print("PCA feature loadings calculated for the top 10 principal components.")`,
                output: `<div class="output-text">PCA feature loadings calculated for the top 10 principal components.</div>`
            },
            {
                code: `<div class="output-success"># Visualize feature contributions to the first few principal components using a heatmap</div>
plt.figure(figsize=(10,8))
sns.heatmap(loadings.iloc[:, :5], cmap="coolwarm",center=0)
plt.title("Feature Contributions to Principal Components")
plt.xlabel("Principal Components")
plt.ylabel("Original Features")
plt.show()`,
                output: `<div style="text-align:center;">
    <h4 style="margin:0 0 10px 0;">Feature Contributions to Principal Components</h4>
    <img src="images/feature_contributiom.png" alt="Feature Contributions to Principal Components" style="width:100%; max-width:800px; border-radius:8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
</div>`
            },
            {
                code: `<div class="output-success"># Interactively analyze and visualize top feature contributions for a selected principal component</div>
def interactive_pca(pc_num=1):
    pc = f"PC{pc_num}"
    print(f"Principal Component: {pc}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_[pc_num-1]:.4f}")
    print(f"Cumulative Variance till {pc}: {pca.explained_variance_ratio_[:pc_num].sum():.4f}")
    top_features = loadings[pc].abs().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,4))
    sns.barplot( x=top_features.values, y=top_features.index)
    plt.xlim(0, top_features.max() * 1.1)
    plt.xlabel("Feature Contribution Strength")
    plt.ylabel("Features")
    plt.title(f"Top 10 Feature Contributions to {pc}")
    plt.grid(True)
    plt.show()

widgets.interact(interactive_pca, pc_num=widgets.IntSlider(min=1,max=10,step=1,value=1,description="PCA Component"));`,
                output: `<div style="padding:10px; font-family: 'Inter', sans-serif;">
    <div style="margin-bottom:20px;">
        <p style="color: #666; font-size: 0.9rem; margin-bottom: 10px;">Select any PCA component to visualize its feature contributions:</p>
        <div style="display:flex; align-items:center; gap:15px;">
            <span style="font-weight:bold; color: #333;">PCA Comp...</span>
            <input type="range" id="pcTestSlider" min="1" max="10" value="1" oninput="window.updatePCATestingPlot()" style="width:200px; cursor: pointer;">
            <span style="font-weight: bold; color: #F57C2A;" id="pcNumVal">1</span>
        </div>
    </div>
    
    <div style="background: #1e1e1e; color: #fff; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; font-size: 0.9rem; margin-bottom: 20px; line-height: 1.5; width: fit-content;">
        <div id="pcLabelText">Principal Component: PC1</div>
        <div id="evRatioText">Explained Variance Ratio: 0.4286</div>
        <div id="cvRatioText">Cumulative Variance till PC1: 0.4286</div>
    </div>
    
    <div style="text-align:left;">
        <img id="pcaCompImg" src="images/pca_comp1.png" alt="Top 10 Feature Contributions to PC1" style="width:100%; max-width:700px; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    </div>
</div>`
            }
        ]
    },



];

// State Management
let STATE = {
    stepIndex: 0,
    subStepIndex: 0,
    stepsStatus: stepsData.map(() => ({ unlocked: false, completed: false, partial: false }))
};

STATE.stepsStatus[0].unlocked = true;

// DOM Elements
const stepsContainer = document.getElementById('stepsContainer');
const codeDisplay = document.getElementById('codeDisplay');
const outputDisplay = document.getElementById('outputDisplay');
const runBtn = document.getElementById('runBtn');
const bottomPane = document.querySelector('.bottom-pane');

function init() {
    renderSidebar();
    loadStep(0);
}

function renderSidebar() {
    stepsContainer.innerHTML = '';
    stepsData.forEach((step, index) => {
        const status = STATE.stepsStatus[index];
        const btn = document.createElement('button');
        btn.classList.add('step-btn');

        let label = `${index + 1}. ${step.title}`;
        if (status.completed) label = `✓ ${step.title}`;
        btn.innerText = label;

        if (status.unlocked) {
            if (status.completed) btn.classList.add('completed');
            else if (status.partial) btn.classList.add('in-progress');

            btn.disabled = false;
            if (index === STATE.stepIndex) btn.classList.add('active');
            btn.onclick = () => loadStep(index);
        } else {
            btn.classList.add('disabled');
            btn.disabled = true;
        }
        stepsContainer.appendChild(btn);
    });

    // Utility Buttons
    const restartBtn = document.createElement('button');
    restartBtn.classList.add('step-btn');
    restartBtn.innerText = "Restart Experiment";
    restartBtn.style.backgroundColor = "#333";
    restartBtn.style.textAlign = "center";
    restartBtn.style.marginTop = "auto";
    restartBtn.onclick = () => { location.reload(); };
    stepsContainer.appendChild(restartBtn);

    // Add Download Button below Restart
    const downloadBtn = document.createElement('button');
    downloadBtn.classList.add('step-btn');
    downloadBtn.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align: middle;">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="7 10 12 15 17 10"></polyline>
        <line x1="12" y1="15" x2="12" y2="3"></line>
      </svg>
      Download Experiment
    `;
    downloadBtn.style.backgroundColor = "#F57C2A"; // Orange (#F57C2A)
    downloadBtn.style.textAlign = 'center';
    downloadBtn.style.marginTop = "10px";
    downloadBtn.style.color = "white";
    downloadBtn.onclick = downloadPDF;
    stepsContainer.appendChild(downloadBtn);
}

function loadStep(index) {
    STATE.stepIndex = index;
    STATE.subStepIndex = 0;
    renderSidebar();
    updateUI();
}

function updateUI() {
    const step = stepsData[STATE.stepIndex];
    const block = step.blocks[STATE.subStepIndex];

    // Comment logic
    const commentMatch = block.code.match(/#\s*([^<\n\r]*)/);
    const codeHeaderBar = document.getElementById('codeHeaderBar');
    if (commentMatch) {
        codeHeaderBar.innerText = "# " + commentMatch[1].trim();
        codeHeaderBar.style.display = 'block';
    } else {
        codeHeaderBar.style.display = 'none';
    }

    // Update Code (Remove all HTML tags and then extract code)
    const codeWithoutTags = block.code.replace(/<[^>]*>/g, '');
    const codeWithoutComment = codeWithoutTags.replace(/#\s*.*/, '').trim();
    codeDisplay.innerHTML = highlightCode(codeWithoutComment);

    // Reset Output
    bottomPane.classList.remove('active-output');
    outputDisplay.innerHTML = '<div class="placeholder-text">Click the Run button to execute...</div>';

    // Reset Button State (Same as previous template)
    runBtn.style.display = 'flex';
    runBtn.classList.remove('completed');
    runBtn.style.backgroundColor = '#F57C2A'; // Orange (#F57C2A)
    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    runBtn.disabled = false;

    // ALWAYS reset onclick to standard runStep
    runBtn.onclick = runStep;
}

function nextSubStep() {
    STATE.subStepIndex++;
    updateUI();
}

function showCompletionMessage() {
    showCompletion();
}

function runStep() {
    const step = stepsData[STATE.stepIndex];
    const block = step.blocks[STATE.subStepIndex];

    // 1. Loading State
    outputDisplay.innerHTML = '<div class="loading-spinner">Running code...</div>';
    runBtn.disabled = true;

    // 2. Simulated Delay (0.5 seconds)
    setTimeout(() => {
        // 3. Show Output
        outputDisplay.innerHTML = block.output;
        bottomPane.classList.add('active-output');

        // 4. Update Button State to Checkmark (Success)
        runBtn.classList.add('completed');
        runBtn.style.backgroundColor = '#A6CE63'; // Green (#A6CE63)
        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';

        // Mark partial progress
        STATE.stepsStatus[STATE.stepIndex].partial = true;
        renderSidebar();

        // Check if this is the Random Prediction block (if applicable)
        if (document.getElementById('randomPredTableBody')) {
            window.generateRandomPrediction && window.generateRandomPrediction();
        }

        // 5. Handle Next Logic
        const hasNextBlock = STATE.subStepIndex < step.blocks.length - 1;

        if (hasNextBlock) {
            // Wait 0.5s then change button to "Next"
            setTimeout(() => {
                runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                runBtn.style.backgroundColor = '#5FA8E4'; // Blue/Next
                runBtn.disabled = false;

                // Switch handler to Next
                runBtn.onclick = nextSubStep;
            }, 500);

        } else {
            // Step Fully Completed
            STATE.stepsStatus[STATE.stepIndex].completed = true;
            renderSidebar(); // Update Current Step to Green Immediately

            // Unlock next step logic
            if (STATE.stepIndex < stepsData.length - 1) {
                STATE.stepsStatus[STATE.stepIndex + 1].unlocked = true;
                renderSidebar(); // Update Next Step to Red Immediately

                // Manual Next Step Arrow Button
                setTimeout(() => {
                    // Change button to Blue Arrow for Next Step
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.style.backgroundColor = '#5FA8E4'; // Blue (#5FA8E4)
                    runBtn.disabled = false;

                    // Logic to go to next MAIN step
                    runBtn.onclick = function () {
                        loadStep(STATE.stepIndex + 1);
                    };
                }, 500);
            } else {
                // End of Experiment - Show "Finish" Button
                renderSidebar();
                setTimeout(() => {
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.style.backgroundColor = '#72b2f7ff'; // Blue-ish Finish
                    runBtn.disabled = false;
                    runBtn.onclick = showCompletionMessage;
                }, 500);
            }
        }

    }, 500);
}

function highlightCode(code) {
    return code
        .replace(/import /g, '<span class="kw">import </span>')
        .replace(/from /g, '<span class="kw">from </span>')
        .replace(/print/g, '<span class="func">print</span>')
        .replace(/def /g, '<span class="kw">def </span>')
        .replace(/return /g, '<span class="kw">return </span>');
}

function showCompletion() {
    outputDisplay.innerHTML = `
     <div style="text-align: center; animation: fadeIn 1s ease;">
      <h1 style="color: #2a9d8f; font-size: 2.5rem; margin-bottom: 20px;">Experiment Completed! ✔️</h1>
      <p style="font-size: 1.5rem; color: #333;">You have completed principal component analysis successfully!</p>
      <button onclick="location.reload()" style="margin-top: 30px; padding: 15px 30px; background-color: #f7a072; color: white; border: none; border-radius: 10px; font-size: 1.2rem; cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">Restart Experiment</button>
    </div>
  `;
    runBtn.style.display = 'none';
}

window.updatePCAPlot = function () {
    const val = document.getElementById('compSlider').value;
    document.getElementById('compVal').innerText = val;
    // In a real app we'd have images comp_0 to comp_10
    // For now we'll just show component selection simulation
    const img = document.getElementById('pcaPlot');
    if (img) {
        // img.src = `./images/pca_comp_${val}.png`;
        console.log("Switching to PCA component:", val);
    }
}

init();

// PDF Download Logic
function downloadPDF() {
    // Redirect to the PDF file for download
    window.open('../assets/EXP-10.pdf', '_blank');
}

