import pandas as pd

# Define the path to the CSV file in your Google Drive
file_path = '/content/drive/My Drive/jewelry.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Test Loaded Data
df.head()

# Identify columns with missing values
missing_columns = df.columns[df.isnull().any()]

df.dropna(subset=['2018-12-01 11:40:29 UTC'], inplace=True)

# Or, impute missing values using appropriate methods (e.g., mean, median, mode)
#df.fillna({'column_name': df['column_name'].mean()}, inplace=True)

# In this case we dropped!

# Convert the date column to datetime
df['2018-12-01 11:40:29 UTC'] = pd.to_datetime(df['2018-12-01 11:40:29 UTC'])

# Rename Columns (provide meaningful names)
df.rename(columns={
    '2018-12-01 11:40:29 UTC': 'purchase_date',
    '1924719191579951782': 'customer_id',
    '1842195256808833386': 'order_id',
    '1': 'quantity',
    '1806829201890738522': 'product_id',
    'jewelry.earring': 'product_category',
    '0': 'return_status',
    '561.51': 'price',
    '1515915625207851155': 'payment_id',
    'Unnamed: 9': 'unknown_column',
    'red': 'color',
    'gold': 'metal',
    'diamond': 'gemstone'
}, inplace=True)

df.head()

# Remove Duplicates
df.drop_duplicates(inplace=True)

# 7. Feature Engineering (Extract year, month, and day from the purchase_date)
df['purchase_year'] = df['purchase_date'].dt.year
df['purchase_month'] = df['purchase_date'].dt.month
df['purchase_day'] = df['purchase_date'].dt.day

# Import winsorize function from scipy
from scipy.stats import mstats

# Apply winsorization to 'price' column
df['price'] = mstats.winsorize(df['price'], limits=[0.01, 0.01])

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())


# Calculate CLV by summing the total purchases for each customer
clv = df.groupby('customer_id')['price'].sum().reset_index()
clv.rename(columns={'price': 'clv'}, inplace=True)

df = pd.merge(df, clv[['customer_id', 'clv']], on='customer_id', how='left')

# Calculate purchase frequency (number of purchases per customer)
purchase_frequency = df.groupby('customer_id')['order_id'].count().reset_index()
purchase_frequency.rename(columns={'order_id': 'purchase_frequency'}, inplace=True)

df = pd.merge(df, purchase_frequency[['customer_id', 'purchase_frequency']], on='customer_id', how='left')

# Calculate purchase recency (time since the last purchase)
current_date = df['purchase_date'].max()
purchase_recency = df.groupby('customer_id')['purchase_date'].max().reset_index()
purchase_recency['recency'] = (current_date - purchase_recency['purchase_date']).dt.days

# Add 'recency' as a new column to the original DataFrame 'df'
df = pd.merge(df, purchase_recency[['customer_id', 'recency']], on='customer_id', how='left')

# Calculate the most popular product category
popular_category = df['product_category'].mode()[0]

from sklearn.cluster import KMeans

# Select relevant features for clustering
X = df[['clv', 'purchase_frequency', 'recency']]

# Define the number of clusters (you can adjust this)
num_clusters = 3

# Fit K-Means clustering model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Display the resulting DataFrame with cluster assignments
print("\nDataFrame with Cluster Assignments:")
print(df[['customer_id', 'clv', 'purchase_frequency', 'recency', 'cluster']])

# Overall View Of Results
df

# Most Purchased Product print
popular_category

import matplotlib.pyplot as plt

# Visualize purchase frequency over time
plt.figure(figsize=(10, 6))
plt.plot(df['purchase_date'], df['purchase_frequency'], marker='o', linestyle='-', color='b')
plt.title('Purchase Frequency Over Time')
plt.xlabel('Purchase Date')
plt.ylabel('Purchase Frequency')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Count the number of occurrences for each product category
product_counts = df['product_category'].value_counts()

# Define the number of top categories to display
top_n = 10

# Group less frequent categories into an "Other" category
other_count = product_counts[top_n:].sum()
top_product_counts = product_counts[:top_n]
top_product_counts['Other'] = other_count

# Create a bar plot for product popularity (top N categories + 'Other')
plt.figure(figsize=(10, 6))
sns.barplot(x=top_product_counts.index, y=top_product_counts.values, palette='viridis')
plt.title('Top Product Categories Popularity')
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Visualize the distribution of CLV values to understand the range of customer values.
plt.figure(figsize=(8, 6))
sns.histplot(df['clv'], bins=20, kde=True, color='skyblue')
plt.title('Customer Lifetime Value Distribution')
plt.xlabel('CLV')
plt.ylabel('Frequency')
plt.show()

# Scatter plot to visualize the segments based on CLV and purchase frequency.
plt.figure(figsize=(8, 6))
sns.scatterplot(x='clv', y='purchase_frequency', hue='cluster', data=df, palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('CLV')
plt.ylabel('Purchase Frequency')
plt.legend(title='Cluster')
plt.show()

