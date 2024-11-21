#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

get_ipython().run_line_magic('matplotlib', 'inline')
import itertools


# In[2]:


import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[3]:


pwd


# In[5]:


credit_cons = pd.read_excel("../input_data/CreditConsumptionData.xlsx")
credit_cons.head()


# In[6]:


cust_behav = pd.read_excel("../input_data/CustomerBehaviorData.xlsx")
cust_behav.head()


# In[7]:


cust_demog = pd.read_excel("../input_data/CustomerDemographics.xlsx")
cust_demog.head()


# In[8]:


credit_cons.info()


# In[9]:


cust_behav.info()


# In[10]:


cust_demog.info()


# # UDFs

# In[11]:


def continuous_var_summary( x ):
    
    # freq and missings
    n_total = x.shape[0]
    n_miss = x.isna().sum()
    perc_miss = n_miss * 100 / n_total
    
    # outliers - iqr
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lc_iqr = q1 - 1.5 * iqr
    uc_iqr = q3 + 1.5 * iqr
    
    return pd.Series( [ x.dtype, x.nunique(), n_total, x.count(), n_miss, perc_miss,
                       x.sum(), x.mean(), x.std(), x.var(), 
                       lc_iqr, uc_iqr, 
                       x.min(), x.quantile(0.01), x.quantile(0.05), x.quantile(0.10), 
                       x.quantile(0.25), x.quantile(0.5), x.quantile(0.75), 
                       x.quantile(0.90), x.quantile(0.95), x.quantile(0.99), x.max() ], 
                     
                    index = ['dtype', 'cardinality', 'n_tot', 'n', 'nmiss', 'perc_miss',
                             'sum', 'mean', 'std', 'var',
                        'lc_iqr', 'uc_iqr',
                        'min', 'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max']) 


# In[12]:


cust_demog.isnull().sum()


# In[13]:


cust_behav.isnull().sum()


# In[14]:


credit_cons.isnull().sum()


# # Merging the Data

# In[15]:


customer=pd.merge(cust_demog,cust_behav,how='inner',on='ID')


# In[16]:


customer=pd.merge(customer,credit_cons,how='inner',on='ID')


# In[17]:


customer.head()


# # Data Exploratory analysis
Dropping loan_enq, since it is a cat var and all values are same
# In[18]:


customer.drop(columns='loan_enq', inplace=True)


# In[19]:


customer.info()


# In[20]:


customer.isnull().sum()


# # Splitting Continuous Var and Cat Var

# In[21]:


customer.select_dtypes(include=['int64','float64']).columns


# In[22]:


concol = ['age', 'Emp_Tenure_Years', 'Tenure_with_Bank','Avg_days_between_transaction', 'cc_cons_apr',
       'dc_cons_apr', 'cc_cons_may', 'dc_cons_may', 'cc_cons_jun',
       'dc_cons_jun', 'cc_count_apr', 'cc_count_may', 'cc_count_jun',
       'dc_count_apr', 'dc_count_may', 'dc_count_jun','investment_1', 'investment_2', 'investment_3',
       'investment_4', 'debit_amount_apr', 'credit_amount_apr',
       'debit_count_apr', 'credit_count_apr', 'max_credit_amount_apr',
       'debit_amount_may', 'credit_amount_may', 'credit_count_may',
       'debit_count_may', 'max_credit_amount_may', 'debit_amount_jun',
       'credit_amount_jun', 'credit_count_jun', 'debit_count_jun',
       'max_credit_amount_jun','emi_active', 'cc_cons']

catcol=['account_type', 'gender','Income','region_code','NetBanking_Flag']


# In[23]:


customercontvar=customer.loc[:,concol]
customercatvar=customer.loc[:,catcol]


# In[24]:


customercontvar.apply(continuous_var_summary)


# In[25]:


customer[['account_type', 'gender','Income']].describe()


# # Outlier Treatment

# In[26]:


customercontvar.apply( lambda x: x.clip(lower = x.quantile(0.01),upper = x.quantile(0.99),inplace=True))


# In[27]:


customercontvar.apply(continuous_var_summary)


# In[28]:


customercontvar1=customercontvar[['age', 'Emp_Tenure_Years', 'Tenure_with_Bank','Avg_days_between_transaction', 'cc_cons_apr',
       'dc_cons_apr', 'cc_cons_may', 'dc_cons_may', 'cc_cons_jun',
       'dc_cons_jun', 'cc_count_apr', 'cc_count_may', 'cc_count_jun',
       'dc_count_apr', 'dc_count_may', 'dc_count_jun','investment_1', 'investment_2', 'investment_3',
       'investment_4', 'debit_amount_apr', 'credit_amount_apr',
       'debit_count_apr', 'credit_count_apr', 'max_credit_amount_apr',
       'debit_amount_may', 'credit_amount_may', 'credit_count_may',
       'debit_count_may', 'max_credit_amount_may', 'debit_amount_jun',
       'credit_amount_jun', 'credit_count_jun', 'debit_count_jun',
       'max_credit_amount_jun','emi_active',]]


# In[30]:


customercontvar2=customercontvar['cc_cons']


# # Missing Value Imputation

# In[32]:


contcol1=['age', 'Emp_Tenure_Years', 'Tenure_with_Bank','Avg_days_between_transaction', 'cc_cons_apr',
       'dc_cons_apr', 'cc_cons_may', 'dc_cons_may', 'cc_cons_jun',
       'dc_cons_jun', 'cc_count_apr', 'cc_count_may', 'cc_count_jun',
       'dc_count_apr', 'dc_count_may', 'dc_count_jun','investment_1', 'investment_2', 'investment_3',
       'investment_4', 'debit_amount_apr', 'credit_amount_apr',
       'debit_count_apr', 'credit_count_apr', 'max_credit_amount_apr',
       'debit_amount_may', 'credit_amount_may', 'credit_count_may',
       'debit_count_may', 'max_credit_amount_may', 'debit_amount_jun',
       'credit_amount_jun', 'credit_count_jun', 'debit_count_jun',
       'max_credit_amount_jun','emi_active',]


# In[33]:


for i in contcol1:
    customercontvar1[i].fillna(customercontvar1[i].mean(),inplace=True)


# In[34]:


customercontvar1.isnull().sum()


# In[35]:


customercatvar.isnull().sum()


# In[36]:


for i in catcol:
    customercatvar[i].fillna(customercatvar[i].mode().iloc[0],inplace=True)


# In[37]:


customercatvar.isnull().sum()


# # Dummy Variables
Ordinal Categorical Variable
# In[38]:


customercatvar.Income.value_counts()


# In[40]:


customercatvar['Income']=pd.Series(np.where(customercatvar.Income=='LOW',1,np.where(customercatvar.Income=='MEDIUM',2,3)))


# In[41]:


customercatvar


# In[42]:


customercatvar.account_type.value_counts()


# In[43]:


customercatvar.gender.value_counts()


# In[44]:


for i in ['account_type','gender']:
    cols=pd.get_dummies(customercatvar[i],prefix = i, drop_first = True)
    customercatvar = pd.concat([customercatvar, cols], axis = 1)
    customercatvar.drop(i, axis = 1, inplace = True )


# In[45]:


customercatvar


# In[46]:


customercatvar.region_code.value_counts()


# In[47]:


customercatvar.NetBanking_Flag.value_counts()


# In[48]:


for i in ['region_code','NetBanking_Flag']:
    customercatvar.loc[:,i]=customercatvar[i].astype('category')
    cols=pd.get_dummies(customercatvar[i],prefix = i, drop_first = True)
    customercatvar = pd.concat([customercatvar, cols], axis = 1)
    customercatvar.drop(i, axis = 1, inplace = True )


# In[49]:


customercatvar


# In[50]:


customercatvar.columns=customercatvar.columns.str.replace('.','_')


# In[51]:


customercatvar


# # Combining the data

# In[52]:


customerclean=pd.concat([customercatvar,customercontvar1],axis=1)


# In[53]:


customerclean=pd.concat([customerclean,customercontvar2],axis=1)


# ##### Seperating the data which is available and which we have to predict

# In[54]:


customercleanavail=customerclean[~customerclean.cc_cons.isna()]


# In[55]:


customercleantopred=customerclean[customerclean.cc_cons.isna()]


# # Model

# In[56]:


feature_columns=customercleanavail.columns.difference(['cc_cons'])


# In[57]:


train_X, test_X, train_Y, test_Y = train_test_split(customercleanavail[feature_columns],customercleanavail['cc_cons'],test_size=0.3, random_state = 123)


# In[58]:


from sklearn.tree import DecisionTreeRegressor


# In[59]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[60]:


params = {'criterion': ["squared_error", "friedman_mse", "absolute_error"],
         'max_depth': [5,6,7],
         'min_samples_split': [2,3,4],
         'min_samples_leaf':  [3,4,5],
         'max_features': ["auto", "sqrt"]}


# In[61]:


DTR = GridSearchCV(DecisionTreeRegressor(),param_grid=params,cv=5,scoring='neg_root_mean_squared_error')


# In[62]:


DTR.fit(train_X,train_Y)


# In[63]:


DTR.best_score_


# In[64]:


DTR.best_params_


# In[65]:


DTRF=DecisionTreeRegressor(criterion='squared_error',max_depth=5,max_features='sqrt',min_samples_leaf=5,min_samples_split=3)


# In[66]:


DTRF.fit(train_X,train_Y)


# # Model Validation using RMSPE

# In[67]:


trainpredval=pd.Series(DTRF.predict(train_X))
np.sqrt(np.mean(np.square(((train_Y-trainpredval)/train_Y)*100)))


# In[68]:


testpredval=pd.Series(DTRF.predict(test_X))
np.sqrt(np.mean(np.square(((test_Y-testpredval)/test_Y)*100)))


# # Missing Values Predicted Output

# In[69]:


customercleantopred.drop(columns='cc_cons',inplace=True)


# In[70]:


customercleantopred.columns


# In[71]:


customercleantopred = customercleantopred.reindex(sorted(customercleantopred.columns), axis=1)


# In[72]:


predval=pd.Series(DTRF.predict(customercleantopred))
predval


# In[4]:


import pandas as pd
df_result = pd.DataFrame()

df_result['Model'] = ['DT_Reg','RF_Reg','Linear_Reg','KNN','XGB']
df_result['RMSE'] = ''
df_result['R2_Square'] =''


# In[5]:


df_result


# In[ ]:




