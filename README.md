# 🌍 Capstone Project – Part 1: Problem Definition & Planning

---

## I. ✅ Sector Selection

**Selected Sector:**  
☑ Environment

---

## II. 🚩 Problem Statement

**"How can real-time and historical PM2.5 air quality data in Kigali be used to predict pollution spikes, inform public health advisories, and optimize city planning for cleaner air?"**

### 🔍 Background
Air pollution is a silent but deadly environmental issue, particularly in developing cities where monitoring infrastructure is limited. Among pollutants, **PM2.5** (fine particulate matter ≤ 2.5 microns) is extremely dangerous due to its ability to penetrate deep into lungs and even the bloodstream, leading to respiratory and cardiovascular illnesses.

### 💡 Innovative Angle
This project proposes an **AI-powered pollution early warning system** that leverages open air quality datasets, **predictive modeling**, and **interactive visualizations** to:
- Detect pollution spikes in real time
- Forecast pollution 24–72 hours in advance
- Provide actionable insights via an interactive dashboard built in **Power BI**

### 🎯 Objective
To empower policymakers, environmental agencies, and citizens with a **data-driven platform** to track and understand air quality trends — and take proactive measures for cleaner air.

---

## III. 📊 Dataset Identification

- **▪ Dataset Title:**  
  **OpenAQ Global Air Quality – PM2.5 Measurements**

- **▪ Source Link:**  
[  [https://openaq.org/#/data](https://explore.openaq.org/locations/313953)]  
  *(Filtered for PM2.5 data in selected locations — e.g., Kigali)*

- **▪ Number of Rows and Columns:**  
  Approx. **50,000 – 100,000 rows**, **12 columns**  
  *(varies based on location and date range selected)*

- **▪ Data Structure:**  
  ☑ **Structured** (CSV format)  
  ☐ Unstructured

- **▪ Data Status:**  
  ☐ Clean  
  ☑ **Requires Preprocessing**  
  - Missing values  
  - Time zone normalization  
  - Unit consistency  
  - Feature engineering for model readiness (e.g., lag features, moving averages)

---

LOADING LIBRARIES
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
```

LOADING THE DATASET 
```
# Load the dataset
df = pd.read_csv('kigaliPM25 measurements.csv')

# Preview the data
df.head()
df.info()
df.describe()
```
<img width="609" height="353" alt="image" src="https://github.com/user-attachments/assets/182df388-483d-4a2d-98ee-8089796f0ac3" />

DATA CLEANING
```
def clean_data(df):
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import IsolationForest
    import pandas as pd
    import numpy as np

    # Identify numeric columns (excluding those with all NaNs)
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [col for col in num_cols if not df[col].isnull().all()]

    # Identify categorical columns (excluding those with all NaNs)
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [col for col in cat_cols if not df[col].isnull().all()]

    # Impute numeric columns with median
    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = pd.DataFrame(
        imputer_num.fit_transform(df[num_cols]),
        columns=num_cols,
        index=df.index
    )

    # Impute categorical columns with most frequent
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = pd.DataFrame(
        imputer_cat.fit_transform(df[cat_cols]),
        columns=cat_cols,
        index=df.index
    )

    # Convert datetime columns to datetime type
    df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'], errors='coerce')
    df['datetimeLocal'] = pd.to_datetime(df['datetimeLocal'], errors='coerce')

    # Remove duplicates
    df = df.drop_duplicates()

    # Outlier detection and removal using Isolation Forest (for 'value' column)
    if 'value' in df.columns:
        iso = IsolationForest(contamination=0.05, random_state=42)
        outliers = iso.fit_predict(df[['value']])
        df = df[outliers == 1]

    # Reset index after cleaning
    df = df.reset_index(drop=True)

    return df

# Usage:
df = pd.read_csv('PM25_kigali_measurements.csv')
df_cleaned = clean_data(df)
print(df_cleaned.head())
```
<img width="544" height="230" alt="image" src="https://github.com/user-attachments/assets/1d1b9686-4485-432d-915f-3ee411ac3ee6" />

EDA 
a) DESCRIPTIVE STATISTICS 
```
# Descriptive statistics for PM2.5 values
print("Descriptive statistics for PM2.5 values:")
print(df_cleaned['value'].describe())

# Count of unique values for categorical columns
print("\nUnique values in categorical columns:")
for col in ['location_name', 'parameter', 'unit', 'timezone', 'owner_name', 'provider']:
    if col in df_cleaned.columns:
        print(f"{col}: {df_cleaned[col].nunique()}")
```
<img width="526" height="157" alt="image" src="https://github.com/user-attachments/assets/cdb38d3e-4465-41b0-af03-0fc9eca28eca" />

B) MISING VALUES CHECK 
```
# Check for any remaining missing values
print("\nMissing values per column:")
print(df_cleaned.isnull().sum())
```
<img width="538" height="151" alt="image" src="https://github.com/user-attachments/assets/c3a033c7-e802-40fd-bb02-3ff6777468d5" />

C)DISTRIBUTION PLOT 
```
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df_cleaned['value'], bins=30, kde=True)
plt.title('Distribution of PM2.5 Values')
plt.xlabel('PM2.5 (µg/m³)')
plt.ylabel('Frequency')
plt.show()
```
<img width="527" height="236" alt="image" src="https://github.com/user-attachments/assets/eb35cb88-a140-4131-bccd-5055941a980a" />

D)BOX PLOT FOR OUTLIER VISUALISATION
```
plt.figure(figsize=(6,4))
sns.boxplot(x=df_cleaned['value'])
plt.title('Boxplot of PM2.5 Values')
plt.xlabel('PM2.5 (µg/m³)')
plt.show()
```
<img width="528" height="196" alt="image" src="https://github.com/user-attachments/assets/93538120-0b37-4bd9-8b98-5db221394686" />

E) TIME SERIES PLOT 

```
plt.figure(figsize=(14,6))
plt.plot(df_cleaned['datetimeLocal'], df_cleaned['value'], marker='o', linestyle='-', markersize=2)
plt.title('PM2.5 Over Time in Kigali')
plt.xlabel('Date')
plt.ylabel('PM2.5 (µg/m³)')
plt.show()
```
<img width="528" height="196" alt="image" src="https://github.com/user-attachments/assets/fa7625d2-0e82-4c04-83e2-d9ce574b0e02" />

F)PM2.5 BY OUR OF DAY 
```
df_cleaned['hour'] = df_cleaned['datetimeLocal'].dt.hour
plt.figure(figsize=(8,5))
sns.boxplot(x='hour', y='value', data=df_cleaned)
plt.title('PM2.5 by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('PM2.5 (µg/m³)')
plt.show()
```
<img width="530" height="239" alt="image" src="https://github.com/user-attachments/assets/7eae7359-3cb3-42a8-8713-aae1e24dff5b" />

G) CORRELATION HEATMAP 
```
# Only select numeric columns for correlation
numeric_df = df_cleaned.select_dtypes(include=[np.number])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numeric Columns Only)')
plt.show()
```
<img width="509" height="282" alt="image" src="https://github.com/user-attachments/assets/4485ba98-f1b5-43ef-9817-da20f7a44baa" />

3. APPLYING MACHINE LEARNING
   
A)ELBOW METHOD FOR CHOOSING NUMBER OF CLUSTERS
```
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()
```
<img width="494" height="234" alt="image" src="https://github.com/user-attachments/assets/3f3845b3-f873-429d-837a-8b3193dd3dca" />

B)FITK MEANS WITH OPTIMAL K 
```
optimal_k = 3  # Change this based on your elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_cleaned['cluster'] = kmeans.fit_predict(features_scaled)
```

C) VISUALIZING CLUSTERS

```
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(x='hour', y='value', hue='cluster', data=df_cleaned, palette='Set1')
plt.title('PM2.5 Clusters by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend(title='Cluster')
plt.show()
```
<img width="546" height="267" alt="image" src="https://github.com/user-attachments/assets/ca618261-f36e-4efd-bf14-ecd621b22970" />

D)SHOWING CLUSTERING CENTERS 
```
# Show cluster centers (in original scale)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster centers (PM2.5, Hour):")
print(centers)
```
<img width="523" height="42" alt="image" src="https://github.com/user-attachments/assets/977c6cce-3c16-4381-a757-b0710d15bba4" />

E) CLUSTERING TIME SERIES 
```
plt.figure(figsize=(14,6))
sns.scatterplot(x='datetimeLocal', y='value', hue='cluster', data=df_cleaned, palette='Set1', s=10)
plt.title('PM2.5 Clusters Over Time')
plt.xlabel('Date')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend(title='Cluster')
plt.show()
```
<img width="542" height="230" alt="image" src="https://github.com/user-attachments/assets/2e2380e6-86c4-46e4-a475-53fc427ed96b" />

4. Evaluate the Model
A)USING SILHOUTTE SCORE 
```
from sklearn.metrics import silhouette_score

# Calculate silhouette score
score = silhouette_score(features_scaled, df_cleaned['cluster'])
print(f'Silhouette Score: {score:.2f}')
```
<img width="520" height="20" alt="image" src="https://github.com/user-attachments/assets/84874a97-b485-44c4-87fe-6f63c353daae" />

5. Incorporate Innovation
A)1. Smart Alert System for High Pollution

```
def smart_alerts(df, pm_threshold=35, min_hours=3):
    """
    Identifies periods where PM2.5 exceeds a threshold for a minimum number of consecutive hours.
    Returns a list of (start_time, end_time) tuples for alert periods.
    """
    df = df.sort_values('datetimeLocal').reset_index(drop=True)
    df['high'] = df['value'] > pm_threshold
    df['group'] = (df['high'] != df['high'].shift()).cumsum()
    alerts = []
    for _, group in df.groupby('group'):
        if group['high'].iloc[0] and len(group) >= min_hours:
            alerts.append((group['datetimeLocal'].iloc[0], group['datetimeLocal'].iloc[-1]))
    return alerts

# Example usage:
alerts = smart_alerts(df_cleaned, pm_threshold=35, min_hours=3)
print("Alert periods (start, end):")
for start, end in alerts:
    print(f"{start} to {end}")
```
<img width="526" height="250" alt="image" src="https://github.com/user-attachments/assets/53f8153a-e523-41e9-975e-024a92a36cc8" />

B) CLUSTER BASED RECOMMENDATIONS 

```
def recommend_action(cluster_label):
    if cluster_label == 0:
        return "Air quality is good. Outdoor activities are safe."
    elif cluster_label == 1:
        return "Moderate air quality. Sensitive groups should limit outdoor activities."
    else:
        return "Poor air quality. Avoid outdoor activities and stay indoors."

# Example: Add recommendations to your DataFrame
df_cleaned['recommendation'] = df_cleaned['cluster'].apply(recommend_action)
print(df_cleaned[['datetimeLocal', 'value', 'cluster', 'recommendation']].head())
```
<img width="523" height="115" alt="image" src="https://github.com/user-attachments/assets/b72d2060-ffdc-4f72-bd3a-b03ccb60e0b6" />

Here’s your DAX code organized cleanly in **Markdown format** for documentation or presentation:

---

## 📊 Power BI DAX Measures for PM2.5 Dashboard

### ✅ **1. Average PM2.5**

```DAX
Average PM2.5 = 
AVERAGE('kigali_pm25_powerbi_complete'[value])
```

---

### 🔺 **2. Max PM2.5**

```DAX
Max PM2.5 = 
MAX('kigali_pm25_powerbi_complete'[value])
```

---

### 🔻 **3. Min PM2.5**

```DAX
Min PM2.5 = 
MIN('kigali_pm25_powerbi_complete'[value])
```

---

### 🔴 **4. High Pollution Hours**

(Threshold: PM2.5 > 35 µg/m³)

```DAX
High Pollution Hours = 
CALCULATE(
    COUNT('kigali_pm25_powerbi_complete'[value]),
    'kigali_pm25_powerbi_complete'[value] > 35
)
```

---


<img width="502" height="290" alt="image" src="https://github.com/user-attachments/assets/8ad66e7e-51ad-4b2d-9bdd-516f0c5ea414" />









