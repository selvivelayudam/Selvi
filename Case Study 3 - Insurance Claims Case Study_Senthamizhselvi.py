#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = "{:.2f}".format

sns.set_context('talk')
import scipy.stats as stats

#Remove warnings
import warnings
warnings.filterwarnings('ignore')


# # 1. Import claims_data.csv and cust_data.csv which is provided to you and combine the two datasets appropriately to create a 360-degree view ofthe data.

# In[2]:


claims = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\Case Study 3 - Insurance Claims Case Study\claims.csv")


# In[3]:


claims


# In[4]:


customer = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\Case Study 3 - Insurance Claims Case Study\cust_demographics.csv")


# In[5]:


customer


# In[6]:


cust_claim = customer.merge(claims,how='inner',left_on='CUST_ID',right_on='customer_id')


# In[7]:


cust_claim.drop(columns=['customer_id'],inplace = True)


# In[8]:


cust_claim


# In[9]:


cust_claim.info()


# # 2. Perform a data audit for the datatypes and find out if there are anymismatch within the current datatypes of the columns and theirbusiness significance.

# In[10]:


cust_claim.dtypes


# # 3. Convert the column claim_amount to numeric. Use the appropriatemodules/attributes to remove the $ sign.

# In[11]:


cust_claim["DateOfBirth"] = pd.to_datetime(cust_claim.DateOfBirth, format = "%d-%b-%y")
cust_claim.loc[(cust_claim.DateOfBirth.dt.year > 2020),"DateOfBirth"]=cust_claim[cust_claim.DateOfBirth.dt.year > 2020]["DateOfBirth"].apply(lambda x: x - pd.DateOffset(years=100))
cust_claim["claim_date"] = pd.to_datetime(cust_claim.claim_date, format = "%m/%d/%Y")
cust_claim["Contact"] = pd.to_numeric(cust_claim.Contact.str.replace("-",""),downcast='float')
cust_claim["claim_amount"] = pd.to_numeric(cust_claim.claim_amount.str.replace("$",""),downcast='float')
cust_claim.head()


# In[12]:


cust_claim.dtypes


# # 4. Of all the injury claims, some of them have gone unreported with thepolice. Create an alert flag (1,0) for all such claims.

# In[13]:


cust_claim['unreported_claims']=  np.where(cust_claim.police_report== 'Unknown',1,0)
cust_claim['unreported_claims'].head(10)


# # 5. One customer can claim for insurance more than once and in each claim,multiple categories of claims can be involved. However, customer IDshould remain unique.Retain the most recent observation and delete any duplicated records inthe data based on the customer ID column.

# In[14]:


cust_claim = cust_claim.groupby('CUST_ID').first().reset_index(drop = True)
cust_claim.head()


# # 6. Check for missing values and impute the missing values with anappropriate value. (mean for continuous and mode for categorical)

# In[15]:


cust_claim.isna().sum()


# In[16]:


cat_col = ["gender","State","Segment","incident_cause","claim_area","claim_type","fraudulent"]
cont_col = ["claim_amount"]


# In[17]:


for col in cat_col:
    cust_claim[col] = cust_claim[col].fillna(cust_claim[col].mode()[0])
cust_claim[cont_col] = cust_claim[cont_col].fillna(cust_claim[cont_col].mean())
cust_claim.head()


# In[18]:


cust_claim.isna().sum()


# # 7. Calculate the age of customers in years. Based on the age, categorize thecustomers according to the below criteria

# # Children < 18,Youth 18-30,Adult 30-60,Senior > 60
# 

# In[19]:


cust_claim["Age"] = round((cust_claim.claim_date - cust_claim.DateOfBirth).apply(lambda x: x.days)/365.25, 0)


# In[20]:


cust_claim.head(5)


# In[21]:


curr_year = pd.to_datetime('today').year
dob_year = pd.DatetimeIndex(cust_claim['DateOfBirth']).year          
x = dob_year-100                                               
v = curr_year - x
y = curr_year - dob_year
cust_claim['age'] = (np.where(dob_year > curr_year,v,y))

#Categorising age group
cust_claim.loc[(cust_claim.age < 18),'AgeGroup'] = 'Children'
cust_claim.loc[(cust_claim.age >=18) & (cust_claim.age <30),'AgeGroup'] = 'Youth'
cust_claim.loc[(cust_claim.age >=30) & (cust_claim.age <60),'AgeGroup'] = 'Adult'
cust_claim.loc[(cust_claim.age >=60),'AgeGroup'] = 'Senior'


# In[22]:


cust_claim.groupby(["AgeGroup"])["age"].count()


# # 8. What is the average amount claimed by the customers from various segments?

# In[23]:


cust_claim.groupby(by = "Segment")[["claim_amount"]].mean()


# # 9. What is the total claim amount based on incident cause for all the claims that have been done at least 20 days prior to 1st of October, 2018.

# In[24]:


cust_claim.loc[cust_claim.claim_date < "2018-09-10",:].groupby("incident_cause")["claim_amount"].sum().add_prefix("total_")


# # 10. How many adults from TX, DE and AK claimed insurance for driver related issues and causes? 

# In[25]:


cust_claim.loc[(cust_claim.incident_cause.str.lower().str.contains("driver") 
    & ((cust_claim.State == "TX") | (cust_claim.State == "DE") | (cust_claim.State == "AK"))),:].groupby(by = "State")["State"].count()


# # 11. Draw a pie chart between the aggregated value of claim amount based on gender and segment. Represent the claim amount as a percentage on the pie chart.

# In[26]:


gender_seg = round(cust_claim.groupby(by = ["gender","Segment"])["claim_amount"].sum().reset_index(),2)
gender_seggender_seg = round(cust_claim.groupby(by = ["gender","Segment"])["claim_amount"].sum().reset_index(),2)
gender_seg


# # 12. Among males and females, which gender had claimed the most for any type of driver related issues? E.g. This metric can be compared using a bar chart

# In[27]:


gender_driver = cust_claim.loc[(cust_claim.incident_cause.str.lower().str.contains("driver"))].groupby(by = "gender")[["gender"]].count().add_prefix("Count_Of_").reset_index()
gender_driver


# In[28]:


sns.barplot(x = "gender", y = "Count_Of_gender", data =gender_driver )
plt.show()


# # 13. Which age group had the maximum fraudulent policy claims? Visualize it on a bar chart.

# In[29]:


Age_Group_Fraud = cust_claim.groupby(by = "AgeGroup")[["fraudulent"]].count().reset_index()


# In[30]:


sns.barplot(x = "AgeGroup", y = "fraudulent", data = Age_Group_Fraud )
plt.show()


# # 14. Visualize the monthly trend of the total amount that has been claimed by the customers. Ensure that on the “month” axis, the month is in a chronological order not alphabetical order. 

# In[31]:


monthly_trend = cust_claim.groupby(["claim_date"])[["claim_amount"]].sum().reset_index()


# In[32]:


monthly_trend['Monthly'] = monthly_trend['claim_date'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
monthly_trend['Yearly'] = monthly_trend['claim_date'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))


# In[33]:


monthly_trend_data = monthly_trend.groupby(["Monthly"])[["claim_amount"]].sum().reset_index()


# In[34]:


plt.plot(monthly_trend_data['Monthly'], monthly_trend_data['claim_amount'], label = 'Trend Line')
plt.show()


# # 16. Is there any similarity in the amount claimed by males and females?

# In[35]:


cust_claim.gender.value_counts()


# In[36]:


cust_claim.columns


# In[37]:


final_Gender =  cust_claim.groupby(["gender", "claim_date"])[["claim_amount"]].sum().reset_index()
final_Gender


# In[38]:


#creating new columns which show "Month" and "Year"
final_Gender['Monthly'] = final_Gender['claim_date'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
final_Gender['Yearly'] = final_Gender['claim_date'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))


# In[39]:


final_Gender.head()


# In[40]:


Claim_amt = 'claim_amount'

male_spend = final_Gender.loc[ final_Gender.gender == "Male", Claim_amt ]
female_spend = final_Gender.loc[ final_Gender.gender == "Female", Claim_amt ]

print( 'mean of male spend: ', male_spend.mean(), '| mean of female spend: ', female_spend.mean() )


# # 17. Is there any relationship between age category and segment?
# 

# In[41]:


obs_freq = pd.crosstab(cust_claim.Segment, cust_claim.AgeGroup )
obs_freq


# # 18. The current year has shown a significant rise in claim amounts as compared to 2016-17 fiscal average which was $10,000.

# In[42]:


final_new =  cust_claim.groupby(["claim_date"])[["claim_amount"]].sum().reset_index()


# In[43]:


final_new['Monthly'] = final_new['claim_date'].apply(lambda x:pd.Timestamp.strftime(x,format="%B"))
final_new['Yearly'] = final_new['claim_date'].apply(lambda x:pd.Timestamp.strftime(x,format="%Y"))
final_new


# In[44]:


final_new_2017 = final_new.loc[ final_new.Yearly == '2017', 'claim_amount' ].mean()
print( final_new_2017 )


# In[45]:


final_new_2018 = final_new.loc[ final_new.Yearly == '2018', 'claim_amount' ]
final_new_2018.count()


# # 19. Is there any difference between age groups and insurance claims?

# In[46]:


age_group_1 = cust_claim['total_policy_claims'].loc[cust_claim['AgeGroup']=="Youth"]
age_group_2 = cust_claim['total_policy_claims'].loc[cust_claim['AgeGroup']=="Adult"]

anova = stats.f_oneway(age_group_1,age_group_2)

f = anova.statistic
p = anova.pvalue
print("The f-value is {} and the p value is {}".format(f,p))
if(p<0.05):
    print('We reject null hypothesis')
else:
    print('We fail to reject null hypothesis')


# # 20. Is there any relationship between total number of policy claims and the claimed amount?

# In[47]:


cust_claim.total_policy_claims.value_counts()


# In[48]:


usage = 'claim_amount'


# In[49]:


s1 = cust_claim.loc[ cust_claim.total_policy_claims == 1.0, usage ]
s2 = cust_claim.loc[ cust_claim.total_policy_claims == 2.0, usage ]
s3 = cust_claim.loc[ cust_claim.total_policy_claims == 3.0, usage ]
s4 = cust_claim.loc[ cust_claim.total_policy_claims == 4.0, usage ]
s5 = cust_claim.loc[ cust_claim.total_policy_claims == 5.0, usage ]
s6 = cust_claim.loc[ cust_claim.total_policy_claims == 6.0, usage ]
s7 = cust_claim.loc[ cust_claim.total_policy_claims == 7.0, usage ]
s8 = cust_claim.loc[ cust_claim.total_policy_claims == 8.0, usage ]

print( 'mean s1:', s1.mean(), '| mean s2:', s2.mean(), '| mean s3:', s3.mean(),'mean s4:', s4.mean(), 
       '| mean s5:', s5.mean(), '| mean s6:', s6.mean(),'|mean s7:', s7.mean(), '| mean s8:', s2.mean(), '| mean s8:', s3.mean() )


# In[50]:


stats.f_oneway( s1, s2, s3, s4, s5, s6, s7, s8 )


# In[ ]:




