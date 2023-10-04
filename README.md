# **Jewelry Store Customer Analysis**
Discovering insights by Analyzing Customer Behavior

## **Project Overview:**
In this project, we will work with data from an e-commerce dataset. The goal is to extract, transform, and load the data, perform data mining to gain insights, and utilize the Dask library for parallel and distributed computing to handle large datasets efficiently.

## **Tools and Libraries Used:**


*   **Python**

*   **Pandas** for Data Manipulation
*   **Dask** for parallel and distributed computing


*   **Matplotlib** or **Seaborn** for data visualization


## **Data Generation/Preparation**

###**Data Uploading:**
Load e-commerce dataset from Kaggle, including customer information, product data, and transaction history. In this case we will be using the [eCommerce purchase history from jewelry store](https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-jewelry-store). This dataset contains contains purchase data from December 2018 to December 2021 (3 years) from a medium sized jewelry online *store*.
For this project we will be using **Python** in **Google Colab**, the folowing code is to help you **Mount Google Drive** in your Google colab if you are using *Python in Google Colab* in order to directly **read** the **csv** file we downloaded from Kaggle into our **Drive**.

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive') # This has to be done everytime you restart your Runtime
```

## **ETL (Extract, Transform, Load):**

### **Extract:**   
Access Dataset in Colab. And Now that the dataset is in your Google Drive, you can read it into your Colab notebook using the path to your Google Drive using **Pandas**:

```python
import pandas as pd

# Define the path to the CSV file in your Google Drive
file_path = '/content/drive/My Drive/jewelry.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Test Loaded Data
df.head()
```

### **Transform (Clean and preprocess the data):**

**1. Handle Missing Values:**


*   First Identify columns with missing values.
*   Then we decide wether to drop rows with missing values or impute missing values.


```python
# Identify columns with missing values
missing_columns = df.columns[df.isnull().any()]

df.dropna(subset=['2018-12-01 11:40:29 UTC'], inplace=True)

# Or, impute missing values using appropriate methods (e.g., mean, median, mode)
#df.fillna({'column_name': df['column_name'].mean()}, inplace=True)

# In this case we dropped!
```

**2. Data Type Conversion:**

*  Convert columns to appropriate data types

```python
# Convert the date column to datetime
df['2018-12-01 11:40:29 UTC'] = pd.to_datetime(df['2018-12-01 11:40:29 UTC'])
```

**3. Rename Cloumns:**

*   Rename columns with meaningful names for future understanding.


```python
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
```


**4. Remove Duplicates:**


* Check for and remove duplicate rows if needed.


```python
# Remove Duplicates
df.drop_duplicates(inplace=True)
```

**5. Feature Creating/Engineering:**


*   Create new features from existing ones.


```python
# Feature Engineering (Extract year, month, and day from the purchase_date)
df['purchase_year'] = df['purchase_date'].dt.year
df['purchase_month'] = df['purchase_date'].dt.month
df['purchase_day'] = df['purchase_date'].dt.day
```

**6.Outlier Detection/Handling:**


*   Identify and handle outliers if they affect the analysis.


```python
# Import winsorize function from scipy
from scipy.stats import mstats

# Apply winsorization to 'price' column
df['price'] = mstats.winsorize(df['price'], limits=[0.01, 0.01])
```


# **Data Exploration and Mining:**


**1. Explore the dataset**


*   Explore the dataset to understand its structure and to determine what data mining techniques can be perform to our data.



```python
# Display basic information about the dataset
print("Dataset Info:")
print(df.info())
```

>- **1st observation:**
We understood that it's important to assess the overall contribution of each customer to the business, especially because this is tipicall to understand in a retail setting like a jewelry store. To help us with this, it is best practices to calculate **Customer Lifetime Value (CLV)** to understand how valuable each customer is to the business over time. It provides insight into the revenue generated by each customer. Once we have this calculation, in our analyzation phase, CLV can guide marketing and customer retention starategies. The way this metric can work is that high CLV customers might recieve loyalty rewards, while low CLV customers might be targeted with special promotions to increase their value.

>- **2nd Observation:**
We also noticed by skimming through our data that not only do we have customers interaction with the store but also that we have lots of purchasing data. Something ideal to examine is **Purchase Frequency and Recency** in order to segment customers based on their engagement level. These metrics help us understand how often customers make purchases and how recently they have interacted with the business. The way this metric can work is that customers who have made frequent and recent purchases are likely to be more engaged and loyal. It can inform targeted marketing efforts and help identify customers who might need re-engagement strategies.

>- **3rd Observation:**
We noticed we have data on our products purchased as **"product_category"**. This is great because it is essential to know what customers are buying the most to optimize inventory, marketing, and product development! Identifying the most popular products or categories provides insights into customer preferences and which items are driving sales. The **Most Popular Products or Categories** metric insights will guide inventory management marketing campaigns, and merchandising decisions.

>- **4th Observation:**
After understanding customers' data and characterisitic with our metrics we think that there could be a wide variation in customer behavior, and it's useful to categorize customers based on their CLV, purchase frequency, and recency. Good techniques to use are **Customer Segmentation and Clustering**. Clustering helps group customers with similar characteristics together allowing for targeted marketing and pesonalized strategies. Clustering can lead to insights such as identifying high-value customersm, at-risk customers, and different customer segments. Again, this information can be used for personlaized marketing,product recommendations and customer retention strategies.



**2. Data Mining/Transformation**


*   Customer Lifetime Value (CLV)
-  Purchase Frequency and Recency
- Most Popular Products or Categories
- Customer Segementation using Clustering Technique


>**Customer Lifetime Value (CLV)**


```python
# Calculate CLV by summing the total purchases for each customer
clv = df.groupby('customer_id')['price'].sum().reset_index()
clv.rename(columns={'price': 'clv'}, inplace=True)

df = pd.merge(df, clv[['customer_id', 'clv']], on='customer_id', how='left')
```


> **Purchase Frequency and Recency**


```python
# Calculate purchase frequency (number of purchases per customer)
purchase_frequency = df.groupby('customer_id')['order_id'].count().reset_index()
purchase_frequency.rename(columns={'order_id': 'purchase_frequency'}, inplace=True)

df = pd.merge(df, purchase_frequency[['customer_id', 'purchase_frequency']], on='customer_id', how='left')
```

```python
# Calculate purchase recency (time since the last purchase)
current_date = df['purchase_date'].max()
purchase_recency = df.groupby('customer_id')['purchase_date'].max().reset_index()
purchase_recency['recency'] = (current_date - purchase_recency['purchase_date']).dt.days

# Add 'recency' as a new column to the original DataFrame 'df'
df = pd.merge(df, purchase_recency[['customer_id', 'recency']], on='customer_id', how='left')
```


> **Most Popular Products or Categories**


```python
# Calculate the most popular product category
popular_category = df['product_category'].mode()[0]
```

> **Customer Segmentation using Clustering Technique (K-Means)**


```python
from sklearn.cluster import KMeans

# Select relevant features for clustering
X = df[['clv', 'purchase_frequency', 'recency']]

# Define the number of clusters (you can adjust this)
num_clusters = 3

# Fit K-Means clustering model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
```

> **Displayed Data Mining Results**


```python
# Display the resulting DataFrame with cluster assignments
print("\nDataFrame with Cluster Assignments:")
print(df[['customer_id', 'clv', 'purchase_frequency', 'recency', 'cluster']])

# Overall View Of Results
df
```


>- **Most Purchased Product**


```python
# Most Purchased Product print
popular_category
```


# **4. Data Visualization:**
 Let's create informative visualizations to present our findings from the transformed data. We'll use Matplotlib and Seaborn for this purpose.


 > **Customer Behavior Trends with Matplotlib:**


```python
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
```


>  **Product Popularity with Seaborn:**


```python
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
```

> **Customer Lifetime Value (CLV) Distribution:**


```python
# Visualize the distribution of CLV values to understand the range of customer values.
plt.figure(figsize=(8, 6))
sns.histplot(df['clv'], bins=20, kde=True, color='skyblue')
plt.title('Customer Lifetime Value Distribution')
plt.xlabel('CLV')
plt.ylabel('Frequency')
plt.show()
```


> **Customer Segmentation:**


```python
# Scatter plot to visualize the segments based on CLV and purchase frequency.
plt.figure(figsize=(8, 6))
sns.scatterplot(x='clv', y='purchase_frequency', hue='cluster', data=df, palette='viridis')
plt.title('Customer Segmentation')
plt.xlabel('CLV')
plt.ylabel('Purchase Frequency')
plt.legend(title='Cluster')
plt.show()
```


# **Conclusion:**

### - **Customer Behavior Insights**: Our analysis revealed a diverse range of Customer Lifetime Values (CLV) among customers. The majority of customers fall within a moderate CLV range, while some exhibit significantly higher values. Understanding CLV distribution is essential for tailoring marketing and retention strategies.

### - **Customer Segmentation**: Through clustering, we identified three distinct customer segments based on CLV, purchase frequency, and recency: High-Value Customers, Regular Customers, and Occasional Customers. Each segment has unique characteristics and requires tailored engagement approaches.

### - **Product Popularity**: Visualizing product category popularity unveiled trends in customer preferences. Certain product categories, such as "jewelry.earring" and "jewelry.ring," stood out as customer favorites. This insight can inform inventory decisions and marketing efforts.

### - **Time Series Analysis**: Examining purchase trends over three years highlighted seasonal variations and peak periods of customer activity. Recognizing these patterns enables the jewelry store to plan promotions and allocate resources effectively.


# **Recommendations**:

### **1. Segmented Marketing Strategies:** Craft targeted marketing campaigns tailored to each customer segment. High-Value Customers may respond well to exclusive offers, while Occasional Customers might benefit from incentives to increase purchase frequency.

### **2. Inventory Management:** Maintain adequate stock levels of popular product categories, especially during peak seasons. Avoid overstocking less popular items to optimize inventory turnover.

### **3. Seasonal Promotions:** Leverage insights from time series analysis to plan seasonal promotions and events. Align marketing efforts with periods of heightened customer engagement.


# **Final Thoughts**:

<img width="540" alt="Jewelry-Store 2023-10-03 at 10 47 17 PM" src="https://github.com/eddycordoba/Customer-Analysis-using-Python/assets/114699310/d822b56a-1637-4b2c-8d18-176295516b99">

Jewelry Store Exterior
<sub><sup>*Credit: https://instoremag.com*</sup></sub>


### **In closing, this analysis equips the jewelry store with actionable insights to enhance customer engagement and optimize operations. Understanding customer behavior, tailoring marketing strategies, and managing inventory effectively are key steps toward sustained growth and customer satisfaction.**

### **Photograph: A representative photograph of the jewelry store's exterior, showcasing its unique facade or branding, is a visual reminder of the physical presence behind the online store. It serves as a tangible connection to the brand and adds authenticity to the analysis.**


