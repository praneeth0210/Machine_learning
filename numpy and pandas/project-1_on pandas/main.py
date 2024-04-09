import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Ecommerce Purchases.csv")

# Print first 10 rows
print("First 10 rows:")
print(df.head(10))

# Print last 10 rows
print("\nLast 10 rows:")
print(df.tail(10))

# Count missing values
print("\nNumber of missing values in each column:")
print(df.isnull().sum())

# Display summary information
print("\nSummary information:")
print(df.info())

# Statistical analysis of 'Purchase Price' column
print("\nStatistical analysis of 'Purchase Price' column:")
print("Maximum:", df['Purchase Price'].max())
print("Minimum:", df['Purchase Price'].min())
print("Mean:", df['Purchase Price'].mean())
print("Median:", df['Purchase Price'].median())

# Data Visualization
# Histogram of Purchase Price
plt.figure(figsize=(10, 6))
sns.histplot(df['Purchase Price'], bins=30, kde=True)
plt.title('Distribution of Purchase Price')
plt.xlabel('Purchase Price')
plt.ylabel('Frequency')
plt.show()

# Bar chart of 'AM or PM' column
plt.figure(figsize=(8, 5))
sns.countplot(x='AM or PM', data=df)
plt.title('Counts of AM and PM Orders')
plt.xlabel('AM or PM')
plt.ylabel('Count')
plt.show()

# Feature Engineering
# Extract month from the 'CC Exp Date' column
df['CC Exp Date'] = pd.to_datetime(df['CC Exp Date'])
df['Expiration Month'] = df['CC Exp Date'].dt.month

# Advanced Analysis (Example: Monthly Sales)
monthly_sales = df.groupby('Expiration Month')['Purchase Price'].sum()
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='Amazon')
plt.title('Monthly Sales by Expiration Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# Documentation
# Add comments to explain each step of the code
# Provide markdown cells with explanations in a Jupyter notebook if presenting as a report
