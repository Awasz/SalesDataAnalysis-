#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import pandas as pd

combined_df = pd.DataFrame()  # Create an empty DataFrame to store the combined data
file_names = [
    'Sales_January_2019.csv',
    'Sales_February_2019.csv',
    'Sales_March_2019.csv',
    'Sales_April_2019.csv',
    'Sales_May_2019.csv',
    'Sales_June_2019.csv',
    'Sales_July_2019.csv',
    'Sales_August_2019.csv',
    'Sales_September_2019.csv',
    'Sales_October_2019.csv',
    'Sales_November_2019.csv',
    'Sales_December_2019.csv',
]

for i, file_name in enumerate(file_names):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    
    if i == 0:  # For the first dataframe, keep the header
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    else:
        combined_df = pd.concat([combined_df, df[1:]], ignore_index=True)  # Skip the header row for subsequent dataframes

# Now, combined_df contains the data from all 12 CSV files with the header from the first file and headers removed from the others


# ### <font color=blue>Q: What was the best month for sales? How much was earned that month?</font>
# 
# ### <font color=green>Q: What city had the highest number of sales?</font>
# 
# ### <font color=red>Q: What time should we display advertisements to maximize the likelihood of customers buying a product?</font>
# 
# ### <font color=orange>Q: What products are most often sold together?</font>
# 
# ### <font color=purple>Q: What product sold the most? Why do you think it sold the most?</font>
# 

# # List of column names to be removed
# column_names = ['Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Order Date', 'Purchase Address']
# 
# # Remove rows that contain the column names in any column
# combined_df = combined_df[~combined_df.apply(lambda row: row.astype(str).str.contains('|'.join(column_names)).any(), axis=1)]
# 
# # Reset the index of the DataFrame
# combined_df.reset_index(drop=True, inplace=True)
# 

# combined_df['Price Each']=combined_df['Price Each'].astype(float)

# In[36]:


combined_df.info()


# ## Convert certain columns into datatime and float
# 

# In[72]:


# Split the "Order Date" column into separate date and time columns
#combined_df[['Date', 'Order Time']] = combined_df['Order Date'].str.split(' ', 1, expand=True)

# Drop the original "Order Date" column
#combined_df.drop(columns=['Order Date'], inplace=True)


# In[153]:


combined_df


# combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%m/%d/%y')
# 

# In[155]:


combined_df.info()


# combined_df.dropna(inplace=True)
# combined_df.isnull().sum()
# 

# In[46]:


combined_df.shape


# In[48]:


#combined_df.drop_duplicates(inplace=True)


# In[50]:


#combined_df.duplicated().sum()


# <h1>Basic EDA</h1>
# 
# 
# 

# In[58]:


#combined_df['Quantity Ordered']=combined_df['Quantity Ordered'].astype(int)


# In[60]:


combined_df.shape


# # Extract month and year from the 'Date' column
# combined_df['Month'] = combined_df['Date'].dt.month
# combined_df['Year'] = combined_df['Date'].dt.year
# 
# # Calculate total sales for each row
# combined_df['Total Sales'] = combined_df['Quantity Ordered'] * combined_df['Price Each']
# 
# # Group by month and year, then calculate the sum of total sales for each month
# monthly_sales = combined_df.groupby(['Year', 'Month'])['Total Sales'].sum().reset_index()
# 
# # Find the month with the highest total sales
# best_month = monthly_sales.loc[monthly_sales['Total Sales'].idxmax()]
# 
# print("Best month for sales:", best_month['Month'])
# print("Total sales in that month:", best_month['Total Sales'])
# 

# In[64]:


combined_df


# In[157]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named combined_df with a 'Order Date' and 'Sales' column.
# If not, adjust column names accordingly.


# Extract the month from the 'Order Date' column and create a new column 'Month'
#combined_df['Month'] = combined_df['Date'].dt.month

# Group the data by month and calculate the total sales for each month
monthly_sales = combined_df.groupby('Month')['Total Sales'].sum().reset_index()

# Define the months as labels for the x-axis
months = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

# Create a bar chart
plt.figure(figsize=(10, 6))
bars=plt.bar(months, monthly_sales['Total Sales'])
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Total Sales by Month')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=10)

plt.tight_layout()


# Show the plot
plt.show()


# ### <font color=green>Q: What city had the highest number of sales?</font>

# In[68]:


combined_df.head()


# In[159]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named combined_df with a 'Purchase Address', 'Quantity Ordered', and 'Price Each' column.
# If not, adjust column names accordingly.

# Convert the 'Date' column to a datetime object (if it's not already)
#combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Extract the city from the 'Purchase Address' column
#combined_df['City'] = combined_df['Purchase Address'].apply(lambda x: x.split(',')[1])

# Calculate total sales for each row
#combined_df['Total Sales'] = combined_df['Quantity Ordered'] * combined_df['Price Each']

# Group by city and calculate the sum of total sales for each city
city_sales = combined_df.groupby('City')['Total Sales'].sum().reset_index()

# Find the city with the highest total sales
best_city = city_sales.loc[city_sales['Total Sales'].idxmax()]

print("City with the highest sales:", best_city['City'])
print("Total sales in that city:", best_city['Total Sales'])


# In[161]:


# Extract the city from the 'Purchase Address' column
#combined_df['City'] = combined_df['Purchase Address'].apply(lambda x: x.split(',')[1])

# Calculate total sales for each row
#combined_df['Total Sales'] = combined_df['Quantity Ordered'] * combined_df['Price Each']

# Group by city and calculate the sum of total sales for each city
city_sales = combined_df.groupby('City')['Total Sales'].sum().reset_index()

# Sort the cities by total sales in descending order and take the top 10
top_10_cities = city_sales.sort_values(by='Total Sales', ascending=False).head(10)

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_10_cities['City'], top_10_cities['Total Sales'], color='skyblue')
plt.xlabel('City')
plt.ylabel('Total Sales')
plt.title('Top 10 Cities by Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the bar chart
plt.show()


# ### <font color=red>Q: What time should we display advertisements to maximize the likelihood of customers buying a product?</font>

# In[163]:



# Convert the 'Time' column to a datetime object (if it's not already)
#combined_df['Order Time'] = pd.to_datetime(combined_df['Order Time'])

# Extract the hour from the 'Time' column
#combined_df['Hour'] = combined_df['Order Time'].dt.hour

# Group by hour and calculate the count of orders for each hour
hourly_orders = combined_df.groupby('Hour').size().reset_index(name='Order Count')

# Create a line chart to visualize the order counts by hour
plt.figure(figsize=(12, 6))
plt.plot(hourly_orders['Hour'], hourly_orders['Order Count'], marker='o', linestyle='-')
plt.xlabel('Hour of the Day')
plt.ylabel('Order Count')
plt.title('Order Counts by Hour of the Day')
plt.xticks(range(0, 24))
plt.grid(True)

# Find the hour with the highest order count
best_hour = hourly_orders.loc[hourly_orders['Order Count'].idxmax()]['Hour']

print("The best time to display advertisements is around", best_hour, "o'clock.")



# **To maximize the impact of our advertising, we should consider posting ads during two key time slots:**
# 
# - **From 9 AM to 1 PM**
# - **And again from 6 PM to 8 PM.**
# 
# These time slots are when you're most likely to reach potential customers and increase the likelihood of product purchases.
# 

# ### <font color=orange>What products are most often sold together?</font>

# In[50]:


#!pip install mlxtend


# In[19]:



cleaned_df.describe(include='all')


# In[38]:


basket = cleaned_df.groupby(['Order ID', 'Product'])['Quantity Ordered'].sum().unstack().fillna(0)


# In[40]:


basket=basket.applymap(lambda x:True if x>0 else False)


# In[48]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Perform Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

# Find association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Filter rules based on confidence and lift
filtered_rules = rules[(rules['confidence'] > 0) & (rules['lift'] > 1.0)]

# Display the filtered rules
print(filtered_rules)


# **No associations were found**

# ### <font color=purple>Q: What product sold the most? Why do you think it sold the most?</font>

# In[123]:


items_count=combined_df['Product'].value_counts()

items_count


# In[125]:


top10=items_count.head(10)


# In[129]:


plt.figure(figsize=(10,6))
top10.plot(kind='bar', color='skyblue')
plt.title('Top Selling Items')
plt.xlabel('Product')
plt.ylabel('Number of Sales')
plt.xticks(rotation=70)
plt.show()


# In[56]:


# Save the combined_df DataFrame to a CSV file
#combined_df.to_csv('combined_df.csv', index=False)


# In[11]:


cleaned_df=pd.read_csv('combined_df.csv')


# In[13]:


cleaned_df.shape


# In[68]:


topproduct=cleaned_df.loc[cleaned_df['Product']=='USB-C Charging Cable']
topproduct.groupby(['Month','City'])['Quantity Ordered'].sum()


# ## How Much Probability?
# 
# <span style="color: blue;">How much probability for next people will order USB-C Charging Cable?</span>
# 
# <span style="color: green;">How much probability for next people will order iPhone?</span>
# 
# <span style="color: orange;">How much probability for next people will order Google Phone?</span>
# 
# <span style="color: purple;">How much probability other people will order Wired Headphones?</span>
# 

# In[71]:


cleaned_df['Quantity Ordered'].sum()


# In[73]:


# Assuming your DataFrame is named df
total_usb_c_cables = cleaned_df[cleaned_df['Product'] == 'USB-C Charging Cable']['Quantity Ordered'].sum()


# In[77]:


total_usb_c_cables


# In[79]:


# Number of USB-C Charging Cable Orders
usb_c_orders = 23928

# Total Number of Orders
total_orders = 208796

# Calculate the Probability
probability = usb_c_orders / total_orders

print("Probability of the next purchase being a USB-C Charging Cable:", probability)


# In[81]:


total_iphone=cleaned_df[cleaned_df['Product']=='iPhone']['Quantity Ordered'].sum()


# In[83]:


total_iphone


# In[85]:


# Number of iPhone Orders
total_iphone = 6845

# Total Number of Orders
total_orders = 208796

# Calculate the Probability
probability = total_iphone / total_orders

print("Probability of the next purchase being an iPhone:", probability)


# In[91]:


# Number of googlephone Orders
total_googlephone = cleaned_df[cleaned_df['Product']=='Google Phone']['Quantity Ordered'].sum()

# Total Number of Orders
total_orders = 208796

# Calculate the Probability
probability = total_googlephone / total_orders

print("Probability of the next purchase being an googlephone:", probability)


# In[95]:


# Number of Wired HeadPhones
wired_headphones = cleaned_df[cleaned_df['Product']=='Wired Headphones']['Quantity Ordered'].sum()

# Total Number of Orders
total_orders = 208796

# Calculate the Probability
probability = wired_headphones / total_orders

print("Probability of the next purchase being an wired_headphones:", probability)


# In[101]:





# Create a dictionary with the probabilities
data = {
    'Product': ['iPhone', 'Google Phone', 'USB-C Charging Cable', 'Wired Headphones'],
    'Probability': [0.03278319508036552, 0.02648039234468093, 0.11459989654974233, 0.09827774478438284]
}

# Create a DataFrame from the dictionary
probability_df = pd.DataFrame(data)

# Set the 'Product' column as the index
probability_df.set_index('Product', inplace=True)

# Function to add color based on the values
def color_negative_red(val):
    color = 'red' if val < 0.05 else 'green'
    return 'color: %s' % color

# Apply the style to the DataFrame
styled_df = probability_df.style.applymap(color_negative_red, subset=['Probability'])

# Display the styled DataFrame
styled_df


# In[ ]:




