#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import re


# In[2]:


customer = pd.read_csv('D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\Case Study 1 - Retail Case Study\Customer.csv')


# In[3]:


transaction = pd.read_csv('D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\Case Study 1 - Retail Case Study\Transactions.csv')


# In[4]:


Product = pd.read_csv('D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\Case Study 1 - Retail Case Study\prod_cat_info.csv')


# # 1. Merge the datasets Customers, Product Hierarchy and Transactions as Customer_Final. Ensure to keep all customers who have done transactions with us and select the join type accordingly.

# In[5]:


customer.head()


# In[6]:


transaction.head()


# In[7]:


Product.head()


# In[10]:


cust_trans = pd.merge( left = customer, right = transaction, how = 'left', left_on = 'customer_Id', right_on = 'cust_id' )


# In[11]:


Customer_Final = cust_trans.merge(Product,how='left',left_on='prod_cat_code',right_on='prod_cat_code')


# In[12]:


Customer_Final.head()


# In[13]:


Customer_Final.drop(columns=['cust_id'],inplace = True)


# In[14]:


Customer_Final.head(2)


# In[15]:


Customer_Final['tran_date']=Customer_Final['tran_date'].str.replace("/",'-')


# In[16]:


Customer_Final['tran_date'] = pd.to_datetime(Customer_Final['tran_date'],format ="%d-%m-%Y")


# # 2. Prepare a summary report for the merged data set.
#  
#  a. Get the column names and their corresponding data types
#  b. Top/Bottom 10 observations
#  c. “Five-number summary” for continuous variables (min, Q1, median, Q3 and max)
#  d. Frequency tables for all the categorical variables

# In[17]:


Customer_Final.columns


# In[18]:


Customer_Final.dtypes


# In[19]:


Customer_Final.head(10)


# In[20]:


Customer_Final.tail(10)


# In[21]:


Customer_Final.info()


# In[22]:


Customer_Final.select_dtypes(include ='float64').quantile(0)


# In[23]:


Customer_Final.select_dtypes(include ='float64').quantile(0.25)


# In[24]:


Customer_Final.select_dtypes(include ='float64').quantile(0.5)


# In[25]:


Customer_Final.select_dtypes(include ='float64').quantile(0.75)


# In[26]:


Customer_Final.select_dtypes(include ='float64').quantile(1)


# In[27]:


Customer_Final.describe(include = 'float64').T


# In[28]:


Customer_Final.describe( include = 'object' ).T


# # 3. Generate histograms for all continuous variables and frequency bars for categorical variables.

# In[29]:


plt.title('This is an histogram for all continous variable')
plt.ylabel('Freq')
plt.xlabel('Bin')
Customer_Final.select_dtypes(include ='float64').hist( bins = 10 )
plt.show()


# In[30]:


category_customer = Customer_Final.loc[:,Customer_Final.dtypes=='object']


# In[31]:


category_customer


# In[32]:


category_customer['Gender']


# In[33]:


plt.figure(figsize=(8,8))
sns.countplot(x='Gender',data=category_customer)
plt.show()


# In[34]:


#Customer_Final.apply(pd.value_counts).T.stack().plot(kind='bar')


# # Calculate the following information using the merged dataset :
#  a. Time period of the available transaction data
#  b. Count of transactions where the total amount of transaction was negative

# In[35]:


# converting "DOB" and "tran_date" from object dtype to dates
Customer_Final["DOB"] = pd.to_datetime(Customer_Final["DOB"], format="%d-%m-%Y")


# In[36]:


Customer_Final.head(2)


# In[37]:


Customer_Final['tran_date'] = pd.to_datetime(Customer_Final['tran_date'])


# In[38]:


Customer_Final.sort_values(by="tran_date")


# In[39]:


min_date = Customer_Final["tran_date"].min()


# In[40]:


max_date = Customer_Final["tran_date"].max()


# In[41]:


#count of transaction_ids where total_amt was negative
negative_transaction = Customer_Final.loc[Customer_Final["total_amt"] < 0,"transaction_id"].count()


# In[42]:


print("Count of transactions where the total amount of transaction was negative is",negative_transaction)


# # 5. Analyze which product categories are more popular among females vs male customers.
# 

# In[43]:


#groupby the data set on the basis of "Gender" and "prod_cat"
product_gender = Customer_Final.groupby(["Gender","prod_cat"])[["Qty"]].sum().reset_index()
product_gender


# In[44]:


#converting to pivot table for better view
product_gender.pivot(index="Gender",columns="prod_cat",values="Qty")

Products that are popular among males are:

Books
Clothing
Electronics
Home and kitchen


Products that are popular among females are:

Bags
Footwear
# # 6. Which City code has the maximum customers and what was the percentage of customers from that city?

# In[45]:


customer_group = Customer_Final.groupby('city_code')['customer_Id'].count().sort_values(ascending =False)


# In[46]:


customer_group


# In[47]:


percentage = round((customer_group[4.0] / customer_group.sum()) * 100,2)


# In[48]:


percentage


# In[49]:


print("City code 4.0 has the maximum customers and the percentage of customers from that city is ",percentage)


# # 7. Which store type sells the maximum products by value and by quantity?
# 

# In[50]:


Customer_Final.head(2)


# In[51]:


Customer_Final.groupby("Store_type")[["Qty","Rate"]].sum().sort_values(by="Qty",ascending=False)


# In[52]:


print('e-Shop store sell the maximum products by value and by quantity')


# # 8. What was the total amount earned from the "Electronics" and "Clothing" categories from Flagship Stores?

# In[53]:


store_group = round(Customer_Final.pivot_table(index = "prod_cat",columns="Store_type", values="total_amt", aggfunc='sum'),2)


# In[54]:


store_group


# In[55]:


store_group.loc[['Clothing','Electronics'],'Flagship store']


# In[56]:


# if we have to find total amount of both 'Clothing' and 'Electronics' from ' Flagship Store'
store_group.loc[['Clothing','Electronics'],'Flagship store'].sum()


# # 9. What was the total amount earned from "Male" customers under the "Electronics" category?
# 

# In[57]:


gender_group = round(Customer_Final.pivot_table(index = "prod_cat",columns="Gender", values="total_amt", aggfunc='sum'),2)


# In[58]:


gender_group


# In[59]:


male_earning = gender_group.loc["Electronics","M"]


# In[60]:


print("The total amount earned from Male customers under the Electronics category is",male_earning)


# # 10. How many customers have more than 10 unique transactions, after removing all transactions which have any negative amounts?

# In[61]:


#creating a new dataframe that does not contain transactions with negative values
pos_trans = Customer_Final.loc[Customer_Final["total_amt"]>0,:]


# In[62]:


pos_trans


# In[63]:


# creating a dataframe that contains unique transactions 
unique_trans = pos_trans.groupby(['customer_Id','prod_cat','prod_subcat'])['transaction_id'].count().reset_index()


# In[64]:


unique_trans


# In[65]:


# now finding the customers which have unique transactions greater than 10
unique_trans_count = unique_trans.groupby('customer_Id')['transaction_id'].count().reset_index()


# In[66]:


unique_trans_count[unique_trans_count['transaction_id'] > 10]


# In[67]:


print('There are no unique transactions greater than 10')


# # 11. For all customers aged between 25 - 35, find out:
# 
a. What was the total amount spent for “Electronics” and “Books” product categories?
# In[76]:


now = pd.Timestamp('now')
Customer_Final['DOB'] = pd.to_datetime(Customer_Final['DOB'], format='%m%d%y')    # 1
Customer_Final['DOB'] = Customer_Final['DOB'].where(Customer_Final['DOB'] < now, Customer_Final['DOB'] -  np.timedelta64(100, 'Y'))   # 2
Customer_Final['AGE'] = (now - Customer_Final['DOB']).astype('<m8[us]')


# In[78]:


Customer_Final.head()


# In[80]:


summ1 = Customer_Final.groupby( by = 'prod_cat' ).total_amt.sum()


# In[81]:


summ1


# In[ ]:




