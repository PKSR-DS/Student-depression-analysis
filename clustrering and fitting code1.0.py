# Importing necessary library packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Loding Student Depression Dataset.csv for analysis 
df_sd = pd.read_csv('C:/Users/prave/OneDrive/Desktop/Clusturing and fitting/Student Depression Dataset.csv')
Depression_analysis_key_metrics = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
                                    'Work/Study Hours', 'Financial Stress','Depression','Work Pressure','Job Satisfaction']


# Displaying basic info about dataset
print(df_sd.head())
print(df_sd.info())
print("\nBasic statistics:")
pd.set_option('display.max_columns', None)
print(df_sd.describe())


# Cleaning the data
df_sd.drop(columns=['id', 'City'], inplace=True) # looks like id and City columns not adding more values for my analysis about student depression factors, so dropping those.
df_sd.dropna(subset=Depression_analysis_key_metrics, inplace=True)#dropping any non values in depression analysis key metrices. 


#Depression_analysis_key_metrics statistics including mean, median, std, skewness, and kurtosis.
def display_summary_statistics(df_sd, Depression_analysis_key_metrics):
    
    key_metrics_stats = df_sd[Depression_analysis_key_metrics].agg(['mean', 'median', 'std', 'skew', 'kurtosis']).T
    print("\nDepression analysis key metrics statistics summary:\n", key_metrics_stats)

display_summary_statistics(df_sd, Depression_analysis_key_metrics)


# Normalization of data and dimension reduction 
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_sd[Depression_analysis_key_metrics])

pca = PCA(n_components=2)
pca_data = pca.fit_transform(normalized_data)


#Implementing k-means clustering and trying to find best number of clusters using silhouette score.
def clustering(data, max_clusters=5):
    
    best_k = 2
    best_score = -1
    best_kmeans = None

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        print(f"\n{k} Clusters: Silhouette Score = {score:.3f}")

        if score > best_score:
            best_k = k
            best_score = score
            best_kmeans = kmeans

    print(f"Best number of clusters: {best_k} with Silhouette Score: {best_score:.3f}")
    return best_kmeans

kmeans_model = clustering(pca_data)
df_sd['Cluster'] = kmeans_model.labels_


#Scatterplot to view cluster results 
def plot_clustering(pca_data, kmeans_model):
    
    plt.figure(figsize=(10, 6), dpi=144)
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=kmeans_model.labels_, palette='viridis', alpha=0.7)
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], 
                color='red', label='Cluster Centers', s=100, marker='X')
    plt.title('Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

plot_clustering(pca_data, kmeans_model)


#Fitting linear regression model for Depression_analysis_key_metrics and visualizing the predicted vs actual values.
def Plot_linearRegression_fitting(df_sd, key_metrics, target):
    
    # Preparing the data
    X = df_sd[key_metrics]
    y = df_sd[target]

    # Normalizing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fitting the model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Getting the predictions
    y_pred = model.predict(X_scaled)

    # Printing linear regression model coefficients
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")    
    coeff_df = pd.DataFrame({
        'Feature': key_metrics + ['Intercept'],
        'Coefficient': list(model.coef_) + [model.intercept_]})
    # Printing the coefficients with key metric names
    print("\nModel Coefficients for key metrics:")
    print(coeff_df)
    
    
    # Displaying linear regression model fitting as predicted vs actual values
    plt.figure(figsize=(10, 6), dpi=144)
    plt.scatter(y, y_pred, label='Data', alpha=0.6)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Perfect Fit')
    plt.title('Multivariant Linear Fitting (Predicted vs Actual)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()
    
    
# Assigning fitting components against target Depression 
key_metrics = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction','Work/Study Hours','Financial Stress','Work Pressure','Job Satisfaction']
target = 'Depression'

Plot_linearRegression_fitting(df_sd, key_metrics, target)


def plot_elbow_method(data):
    
    #Plotting elbow method for clustering 
    
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6), dpi=144)
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

plot_elbow_method(pca_data)


def plot_correlation_heatmap(df_sd, Depression_analysis_key_metrics):
    
    #Correlation heatmap for Depression_analysis_key_metrics
    
    plt.figure(figsize=(10, 6), dpi=144)
    correlation_matrix = df_sd[Depression_analysis_key_metrics].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()

plot_correlation_heatmap(df_sd, Depression_analysis_key_metrics)


# Pie plot to analyze count % of Depression Cases
plt.figure(figsize=(10, 6), dpi=144)
df_sd['Depression'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Count % of Depression Cases (1=Yes,0=No)')
plt.show()


#Countplot to check, how less sleep increasing depression levels
plt.figure(figsize=(10, 6), dpi=144)
sns.countplot(data=df_sd, x='Sleep Duration', hue='Depression', palette='Set1')
plt.title('Comparison of Sleep Hours with Depression Levels')
plt.xlabel('Sleep Duration')
plt.ylabel('Count')
plt.show()


# Line plot to analyze that how Family History of Mental Illness affecting CGPA
plt.figure(figsize=(10, 6), dpi=144)
sns.lineplot(x='Family History of Mental Illness', y='CGPA', data=df_sd, marker='o', color='pink')
plt.title('CGPA vs Family History of Mental Illness')
plt.xlabel('Family History of Mental Illness (0 = No, 1 = Yes)')
plt.ylabel('CGPA')
plt.show()


#Scatterplot to analyze Age vs Academic Pressure by Depression Status
plt.figure(figsize=(10, 6), dpi=144)
sns.scatterplot(data=df_sd, x='Age', y='Academic Pressure', hue='Depression')
plt.title('Age vs Academic Pressure by Depression Status')
plt.show()