#!/usr/bin/env python
# coding: utf-8

# # Basic Exercises on Data Importing - Understanding - Manipulating - Analysis - Visualization

# ## Section-1: The pupose of the below exercises (1-7) is to create dictionary and convert into dataframes, how to diplay etc...
# ## The below exercises required to create data 

# ### 1. Import the necessary libraries (pandas, numpy, datetime, re etc)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import re

# set the graphs to show in the jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# set seabor graphs to a better style
sns.set(style="ticks")


# ### 2. Run the below line of code to create a dictionary and this will be used for below exercises

# In[2]:


raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']                        
            }


# ### 3. Assign it to a object called pokemon and it should be a pandas DataFrame

# In[3]:


pokemon = pd.Series(raw_data)


# In[4]:


pokemon


# ### 4. If the DataFrame columns are in alphabetical order, change the order of the columns as name, type, hp, evolution, pokedex

# In[5]:


pokemon1 = pd.DataFrame.from_dict(raw_data)
pokemon1 = pokemon1[["name", "type", "hp", "evolution", "pokedex"]]
pokemon1


# In[ ]:





# ### 5. Add another column called place, and insert places (lakes, parks, hills, forest etc) of your choice.

# In[6]:


place=['lakes','parks','hills','forest']
pokemon1['place']= place
pokemon1


# ### 6. Display the data type of each column

# In[7]:


pokemon1.dtypes


# ### 7. Display the info of dataframe

# In[8]:


pokemon1.info()


# ## Section-2: The pupose of the below exercise (8-20) is to understand deleting data with pandas.
# ## The below exercises required to use wine.data

# ### 8. Import the dataset *wine.txt* from the folder and assign it to a object called wine
# 
# Please note that the original data text file doesn't contain any header. Please ensure that when you import the data, you should use a suitable argument so as to avoid data getting imported as header.

# In[9]:


wine = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\2. Basic Data Manipulation - Visualization Exercise\Exercise Data Files\wine.txt", header =None)


# In[10]:


wine.head(2)


# ### 9. Delete the first, fourth, seventh, nineth, eleventh, thirteenth and fourteenth columns

# In[11]:


wine = wine.drop(columns = [0,3,6,8,10,12,13])


# In[12]:


wine.head(2)


# ### 10. Assign the columns as below:
# 
# The attributes are (dontated by Riccardo Leardi, riclea '@' anchem.unige.it):  
# 1) alcohol  
# 2) malic_acid  
# 3) alcalinity_of_ash  
# 4) magnesium  
# 5) flavanoids  
# 6) proanthocyanins  
# 7) hue 

# In[13]:


wine.columns = ['alcohol' , 'malic_acid' , 'alcalinity_of_ash' , 'magnesium' , 'flavanoids' , 'proanthocyanins' , 'hue']


# In[14]:


wine.head(2)


# ### 11. Set the values of the first 3 values from alcohol column as NaN

# In[15]:


wine.loc[0:2,'alcohol'] = pd.NA


# ### 12. Now set the value of the rows 3 and 4 of magnesium as NaN

# In[16]:


wine.loc[2:3 , 'magnesium'] = pd.NA


# ### 13. Fill the value of NaN with the number 10 in alcohol and 100 in magnesium

# In[17]:


wine = wine.fillna({'alcohol':10,'magnesium':100})


# ### 14. Count the number of missing values in all columns.

# In[18]:


wine.isna()


# In[19]:


wine.isna().sum().sum()


# ### 15.  Create an array of 10 random numbers up until 10 and save it.

# In[20]:


a1=np.random.randint(0,10,10)


# In[21]:


a1


# ### 16.  Set the rows corresponding to the random numbers to NaN in the column *alcohol*

# In[22]:


wine.alcohol.iloc[a1] = np.NaN


# In[23]:


wine


# ### 17.  How many missing values do we have now?

# In[24]:


wine.isna().sum()


# ### 18. Print only the non-null values in alcohol

# In[25]:


wine.notna()


# In[26]:


wine.notna().sum()


# In[27]:


wine.loc[~wine.alcohol.isna(),'alcohol']


# ### 19. Delete the rows that contain missing values

# In[28]:


wine.dropna(axis=0)


# In[ ]:





# ### 20.  Reset the index, so it starts with 0 again

# In[29]:


wine.reset_index()


# ## Section-3: The pupose of the below exercise (21-27) is to understand ***filtering & sorting*** data from dataframe.
# ## The below exercises required to use chipotle.tsv

# This time we are going to pull data directly from the internet.  
# Import the dataset directly from this link (https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv) and create dataframe called chipo

# In[30]:


chipo=pd.read_table('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv')


# In[31]:


chipo.head(2)


# ### 21. How many products cost more than $10.00? 
# 
# Use `str` attribute to remove the $ sign and convert the column to proper numeric type data before filtering.
# 

# In[32]:


chipo.item_price = chipo.item_price.str.replace('$', ' ').astype(float)


# In[33]:


chipo[(chipo.item_price > 10.00)]


# In[34]:


np.count_nonzero(chipo.item_price>10.00)


# ### 22. Print the Chipo Dataframe & info about data frame

# In[35]:


chipo.info()


# ### 23. What is the price of each item? 
# - Delete the duplicates in item_name and quantity
# - Print a data frame with only two columns `item_name` and `item_price`
# - Sort the values from the most to less expensive

# In[36]:


chipo1 = chipo.drop_duplicates(subset = ['item_name','quantity'])

chipo1 = chipo1[['item_name','item_price']]

chipo1.sort_values(by = ['item_price'],ascending = [False])


# ### 24. Sort by the name of the item

# In[37]:


chipo.sort_values('item_name')


# ### 25. What was the quantity of the most expensive item ordered?

# In[38]:


chipo.groupby('quantity').item_price.max()


# ### 26. How many times were a Veggie Salad Bowl ordered?

# In[39]:


len(chipo.loc[chipo.item_name == 'Veggie Salad Bowl'])


# ### 27. How many times people orderd more than one Canned Soda?

# In[40]:


len(chipo.loc[(chipo.item_name == 'Canned Soda') & (chipo.quantity > 1)])


# ## Section-4: The purpose of the below exercises is to understand how to perform aggregations of data frame
# ## The below exercises (28-33) required to use occupation.csv

# ###  28. Import the dataset occupation.csv and assign object as users

# In[41]:


users = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\2. Basic Data Manipulation - Visualization Exercise\Exercise Data Files/occupation.csv",sep = "|")


# In[42]:


users


# ### 29. Discover what is the mean age per occupation

# In[43]:


users.groupby('occupation').age.mean()


# ### 30. Discover the Male ratio per occupation and sort it from the most to the least.
# 
# Use numpy.where() to encode gender column.

# In[44]:


users.head(2)


# In[45]:


users['is_male'] = np.where(users['gender'] == 'M',1,0)
total_males_by_occupation = pd.pivot_table(data = users,index = 'occupation',values = 'is_male',aggfunc = 'sum')
total_users_by_occupation = pd.pivot_table(data = users,index = 'occupation',values = 'is_male',aggfunc = len)
total_males_by_occupation/total_users_by_occupation


# ### 31. For each occupation, calculate the minimum and maximum ages

# In[46]:


pd.pivot_table(data = users,values ='age',index ='occupation',aggfunc = ['min','max'] )


# ### 32. For each combination of occupation and gender, calculate the mean age

# In[47]:


pd.pivot_table(data = users,index = ['occupation','gender'],values = 'age',aggfunc = 'mean')


# ### 33.  For each occupation present the percentage of women and men

# In[48]:


gender_ocup = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})
occup_count = users.groupby(['occupation']).agg('count')
occup_gender = gender_ocup.div(occup_count, level = "occupation") * 100
occup_gender.loc[: , 'gender']


# ## Section-6: The purpose of the below exercises is to understand how to use lambda-apply-functions
# ## The below exercises (34-41) required to use student-mat.csv and student-por.csv files 

# ### 34. Import the datasets *student-mat* and *student-por* and append them and assigned object as df

# In[49]:


student_mat = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\2. Basic Data Manipulation - Visualization Exercise\Exercise Data Files\student-mat.csv")

student_por = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\2. Basic Data Manipulation - Visualization Exercise\Exercise Data Files\student-por.csv")


# In[50]:


student_mat.head(2)


# In[51]:


student_por.head(2)


# In[52]:


df = pd.concat([student_mat,student_por],axis = 0)


# In[53]:


df


# ### 35. For the purpose of this exercise slice the dataframe from 'school' until the 'guardian' column

# In[54]:


df.loc[:,'school':'guardian'].head(2)


# ### 36. Create a lambda function that captalize strings (example: if we give at_home as input function and should give At_home as output.

# In[55]:


lambda x:x.capitalize()


# ### 37. Capitalize both Mjob and Fjob variables using above lamdba function

# In[56]:


df['Mjob'] = list(map(lambda x:x.capitalize(),df['Mjob']))


# In[57]:


df['Mjob'] 


# In[58]:


df['Fjob'] = list(map(lambda x:x.capitalize(),df['Fjob']))


# In[59]:


df['Fjob']


# ### 38. Print the last elements of the data set. (Last few records)

# In[60]:


df.tail()


# ### 39. Did you notice the original dataframe is still lowercase? Why is that? Fix it and captalize Mjob and Fjob.

# In[61]:


df[['Mjob','Fjob']] = df[['Mjob','Fjob']].applymap(lambda x : x.capitalize())


# In[62]:


df[['Mjob','Fjob']]


# ### 40. Create a function called majority that return a boolean value to a new column called legal_drinker

# In[63]:


majority = lambda x: True if x > 17 else False
    


# In[64]:


df['legal_drinker'] = df.age.apply(majority)


# In[65]:


df


# ### 41. Multiply every number of the dataset by 10. 

# In[66]:


df[df.select_dtypes(include=['number']).columns] * 10


# ## Section-6: The purpose of the below exercises is to understand how to perform simple joins
# ## The below exercises (42-48) required to use cars1.csv and cars2.csv files 

# ### 42. Import the datasets cars1.csv and cars2.csv and assign names as cars1 and cars2

# In[67]:


cars1 = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\2. Basic Data Manipulation - Visualization Exercise\Exercise Data Files/cars1.csv")


# In[70]:


cars2 = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\2. Basic Data Manipulation - Visualization Exercise\Exercise Data Files/cars2.csv")


# In[71]:


cars1.head(2)


# In[72]:


cars2.head(2)


#    ### 43. Print the information to cars1 by applying below functions 
#    hint: Use different functions/methods like type(), head(), tail(), columns(), info(), dtypes(), index(), shape(), count(), size(), ndim(), axes(), describe(), memory_usage(), sort_values(), value_counts()
#    Also create profile report using pandas_profiling.Profile_Report

# In[73]:


cars1.info()


# ### 44. It seems our first dataset has some unnamed blank columns, fix cars1

# In[74]:


cars1=cars1.dropna(axis=1)


# ### 45. What is the number of observations in each dataset?

# In[75]:


cars1.shape


# In[76]:


cars2.shape


# ### 46. Join cars1 and cars2 into a single DataFrame called cars

# In[77]:


cars1.head()


# In[78]:


cars2.head()


# In[79]:


cars = pd.concat([cars1 , cars2])


# In[80]:


cars


# In[ ]:





# ### 47. There is a column missing, called owners. Create a random number Series from 15,000 to 73,000.

# In[81]:


owners = pd.Series(np.random.randint(15000, 73000, cars.shape[0]))


# In[82]:


owners


# ### 48. Add the column owners to cars

# In[83]:


cars['owners']=owners


# In[84]:


cars


# In[ ]:





# ## Section-7: The purpose of the below exercises is to understand how to perform date time operations

# ### 49. Write a Python script to display the
# - a. Current date and time
# - b. Current year
# - c. Month of year
# - d. Week number of the year
# - e. Weekday of the week
# - f. Day of year
# - g. Day of the month
# - h. Day of week

# In[85]:


dt.datetime.now()


# In[86]:


x = dt.datetime.now()
print("a. Current date and time: ",x)
print("b. Current year: ",x.year)
print("c. Month of year: ",x.month)
print("d. Week number of the year: ",x.isocalendar()[1])
print("e. Weekday of the week: ",x.strftime("%A"))
print("f. Day of year: ",x.timetuple().tm_yday)
print("g. Day of the month ",x.day)
print("h. Day of week: ",x.isocalendar()[2])


# ### 50. Write a Python program to convert a string to datetime.
# Sample String : Jul 1 2014 2:43PM 
# 
# Expected Output : 2014-07-01 14:43:00

# In[87]:


sample_date = dt.datetime.strptime('Jul 1 2014  2:43PM', '%b %d %Y %I:%M%p')
print(sample_date)


# ### 51. Write a Python program to subtract five days from current date.
# 
# Current Date : 2015-06-22
# 
# 5 days before Current Date : 2015-06-17

# In[88]:


current=dt.date.today()
sub=current-dt.timedelta(days=5)
print('Current Date :',current)
print('5 days before Current Date :',sub)


# ### 52. Write a Python program to convert unix timestamp string to readable date.
# 
# Sample Unix timestamp string : 1284105682
#     
# Expected Output : 2010-09-10 13:31:22

# In[89]:


dt.datetime.fromtimestamp(int("1284105682")).strftime('%Y-%m-%d %H:%M:%S')


# ### 53. Convert the below Series to pandas datetime : 
# 
# DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])
# 
# Make sure that the year is 19XX not 20XX

# In[90]:


DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])


# In[91]:


dob = DoB.apply(lambda x: dt.datetime.strptime(x,"%d%b%y"))
dob = dob - pd.offsets.DateOffset(years=100)
print(dob)


# In[ ]:





# ### 54. Write a Python program to get days between two dates. 

# In[92]:


from datetime import date

date1 = date(2023,10,5)
date2 = date(2023,12,14)

timedelta = date2 - date1
timedelta.days


# ### 55. Convert the below date to datetime and then change its display format using the .dt module
# 
# Date = "15Dec1989"
# 
# Result : "Friday, 15 Dec 98"

# In[93]:


Date = "15Dec1989"
Date=pd.to_datetime(Date)
dt.datetime.strftime(Date,'%A, %d %b %y')


# ## The below exercises (56-66) required to use wind.data file 

# ### About wind.data:
# 
# The data have been modified to contain some missing values, identified by NaN.  
# 
# 1. The data in 'wind.data' has the following format:
"""
Yr Mo Dy   RPT   VAL   ROS   KIL   SHA   BIR   DUB   CLA   MUL   CLO   BEL   MAL
61  1  1 15.04 14.96 13.17  9.29   NaN  9.87 13.67 10.25 10.83 12.58 18.50 15.04
61  1  2 14.71   NaN 10.83  6.50 12.62  7.67 11.50 10.04  9.79  9.67 17.54 13.83
61  1  3 18.50 16.88 12.33 10.13 11.17  6.17 11.25   NaN  8.50  7.67 12.75 12.71
"""
The first three columns are year, month and day.  The remaining 12 columns are average windspeeds in knots at 12 locations in Ireland on that day. 
# ### 56. Import the dataset wind.data and assign it to a variable called data and replace the first 3 columns by a proper date time index

# In[94]:


data = pd.read_csv('D:/Analytix lab/Senthamizhselvi Datascience_360/PYTHON/CASE STUDY/2. Basic Data Manipulation - Visualization Exercise/Exercise Data Files/wind.data')


# In[95]:


data


# In[96]:


data['Date'] = pd.to_datetime(data[['Yr','Mo','Dy']].astype(str).agg('-'.join, axis=1))


# In[97]:


data


# ### 57. Year 2061 is seemingly imporoper. Convert every year which are < 70 to 19XX instead of 20XX.

# In[98]:


data["Date"] = np.where(pd.DatetimeIndex(data["Date"]).year < 2000,data.Date,data.Date - pd.offsets.DateOffset(years=100))


# In[99]:


data


# ### 58. Set the right dates as the index. Pay attention at the data type, it should be datetime64[ns].

# In[100]:


newData = data.set_index("Date")
newData.index.astype("datetime64[ns]")
newData


# ### 59. Compute how many values are missing for each location over the entire record.  
# #### They should be ignored in all calculations below. 

# In[101]:


newData.info()


# In[102]:


newData.drop(columns=['Yr','Mo','Dy'])


# In[103]:


print(newData.isnull().values.ravel().sum())


# ### 60. Compute how many non-missing values there are in total.

# In[104]:


x=newData.count()
print("Total Non-missing values are :",x.sum())
newData


# ### 61. Calculate the mean windspeeds over all the locations and all the times.
# #### A single number for the entire dataset.

# In[105]:


y = newData.mean()
y.mean()


# ### 62. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days 
# 
# #### A different set of numbers for each location.

# In[106]:


def stats(x):
    x = pd.Series(x)
    Min = x.min()
    Max = x.max()
    Mean = x.mean()
    Std = x.std()
    res = [Min,Max,Mean,Std]
    indx = ["Min","Max","Mean","Std"]
    res = pd.Series(res,index=indx)
    return res
loc_stats = newData.apply(stats)
loc_stats


# ### 63. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day.
# 
# #### A different set of numbers for each day.

# In[107]:


day_stats = newData.apply(stats,axis=1)
day_stats.head()


# ### 64. Find the average windspeed in January for each location.  
# #### Treat January 1961 and January 1962 both as January.

# In[108]:


january_data = newData[newData.index.month == 1]
print ("January windspeeds:")
print (january_data.mean())


# ### 65. Calculate the mean windspeed for each month in the dataset.  
# #### Treat January 1961 and January 1962 as *different* months.
# #### (hint: first find a  way to create an identifier unique for each month.)

# In[109]:


newdata = newData.groupby(lambda d: (d.month, d.year))
print ("Mean wind speed for each month in each location")
print (newdata.mean())


# In[ ]:





# In[ ]:





# ### 66. Calculate the min, max and mean windspeeds and standard deviations of the windspeeds across all locations for each week (assume that the first week starts on January 2 1961) for the first 52 weeks.

# In[110]:


first_year = newData[newData.index.year == 1961]
stats1 = newData.resample('W').mean().apply(lambda x: x.describe())
print (stats1)


# In[ ]:





# In[ ]:





# ## The below exercises (67-70) required to use appl_1980_2014.csv  file

# ### 67. Import the file appl_1980_2014.csv and assign it to a variable called 'apple'

# In[111]:


apple = pd.read_csv('D:/Analytix lab/Senthamizhselvi Datascience_360/PYTHON/CASE STUDY/2. Basic Data Manipulation - Visualization Exercise/Exercise Data Files/appl_1980_2014.csv')


# In[112]:


apple


# ### 68.  Check out the type of the columns

# In[113]:


apple.dtypes


# ### 69. Transform the Date column as a datetime type

# In[114]:


apple.Date = pd.to_datetime(apple.Date,format = '%Y-%m-%d')


# In[115]:


apple


# ### 70.  Set the date as the index

# In[116]:


apple.index = apple.Date


# In[117]:


apple.index


# ### 71.  Is there any duplicate dates?

# In[118]:


apple.index.duplicated().sum()


# ### 72.  The index is from the most recent date. Sort the data so that the first entry is the oldest date.

# In[119]:


apple = apple.sort_index(ascending = True)


# In[120]:


apple


# ### 73. Get the last business day of each month

# In[121]:


date_range = pd.DataFrame({'year': apple.index.year,'month': apple.index.month,'Day': apple.index.day})
date_range.tail(1)


# ### 74.  What is the difference in days between the first day and the oldest

# In[122]:


apple.Date.max() - apple.Date.min()


# ### 75.  How many months in the data we have?

# In[123]:


len(date_range.month)


# ## Section-8: The purpose of the below exercises is to understand how to create basic graphs

# ### 76. Plot the 'Adj Close' value. Set the size of the figure to 13.5 x 9 inches

# In[124]:


apple['Adj Close'].plot(kind = 'line',figsize = [13.5,9])


# ## The below exercises (77-80) required to use Online_Retail.csv file

# ### 77. Import the dataset from this Online_Retail.csv and assign it to a variable called online_rt

# In[16]:


online_rt = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\2. Basic Data Manipulation - Visualization Exercise\Exercise Data Files\Online_Retail_1.csv", encoding = 'ISO-8859-1')


# ### 78. Create a barchart with the 10 countries that have the most 'Quantity' ordered except UK

# In[17]:


df = online_rt[online_rt.Country != 'United Kingdom']

df

df = pd.pivot_table(data = df,index = 'Country',values = 'Quantity')

df

df.sort_values(by = ['Quantity'],ascending = [False]).iloc[:10,:]

df.plot(kind = 'bar')


# ### 79.  Exclude negative Quatity entries

# In[18]:


df = online_rt[(online_rt.Country != 'United Kingdom') & (online_rt.Quantity >0)]

df

df = pd.pivot_table(data = df,index = 'Country',values = 'Quantity')

df

df.sort_values(by = ['Quantity'],ascending = [False]).iloc[:10,:]

df.plot(kind = 'bar')


# ### 80. Create a scatterplot with the Quantity per UnitPrice by CustomerID for the top 3 Countries
# Hint: First we need to find top-3 countries based on revenue, then create scater plot between Quantity and Unitprice for each country separately
# 

# In[19]:


online_rt['revenue'] = online_rt['Quantity']*online_rt['UnitPrice']

top3country = online_rt.groupby('Country')[['revenue']].sum().sort_values(by = 'revenue',ascending = False).head(3).index.to_list()

x = online_rt.query(f'Country == {top3country}').groupby(['Country','CustomerID']).agg({'Quantity':'sum','UnitPrice':'mean'}).reset_index()

y = sns.FacetGrid(x,col = 'Country')

y.map(plt.scatter,'Country','UnitPrice')


# ## The below exercises (81-90) required to use FMCG_Company_Data_2019.csv file

# ### 81. Import the dataset FMCG_Company_Data_2019.csv and assign it to a variable called company_data

# In[20]:


company_data = pd.read_csv(r"D:\Analytix lab\Senthamizhselvi Datascience_360\PYTHON\CASE STUDY\2. Basic Data Manipulation - Visualization Exercise\Exercise Data Files\FMCG_Company_Data_2019.csv")


# ### 82. Create line chart for Total Revenue of all months with following properties
# - X label name = Month
# - Y label name = Total Revenue

# In[21]:


company_data.index=company_data.Month.str.slice(stop=3)

df=company_data[['Month','Total_Revenue']]

df['Total_Revenue'].plot(kind='line',xticks=range(0,13))


# ### 83. Create line chart for Total Units of all months with following properties
# - X label name = Month
# - Y label name = Total Units
# - Line Style dotted and Line-color should be red
# - Show legend at the lower right location.

# In[22]:


df2 = company_data[['Month','Total_Units']]

df2['Total_Units'].plot(kind = 'line',xticks = range(0,13),color ='red',linestyle = '--')

plt.xlabel('Month')

plt.ylabel('Total_Units')


# ### 84. Read all product sales data (Facecream, FaceWash, Toothpaste, Soap, Shampo, Moisturizer) and show it  using a multiline plot
# - Display the number of units sold per month for each product using multiline plots. (i.e., Separate Plotline for each product ).

# In[23]:


data = company_data[['FaceCream','FaceWash','ToothPaste','Soap','Shampo','Moisturizer',]]

data.plot(kind = 'line',figsize = [10,12],xticks = range(0,12))


# ### 85. Create Bar Chart for soap of all months and Save the chart in folder

# In[24]:


folder = company_data[['Month','Total_Units']]

folder.plot(kind = 'bar')


# ### 86. Create Stacked Bar Chart for Soap, Shampo, ToothPaste for each month
# The bar chart should display the number of units sold per month for each product. Add a separate bar for each product in the same chart.

# In[25]:


bar = company_data[['Soap','Shampo','ToothPaste']]

bar.plot(kind = 'bar',stacked = True)


# ### 87. Create Histogram for Total Revenue

# In[26]:


plt.hist(company_data.Total_Revenue)


# ### 88. Calculate total sales data (quantity) for 2019 for each product and show it using a Pie chart. Understand percentage contribution from each product

# In[47]:


df1 = company_data.copy()

df1.drop(columns = ['Month','Total_Units','Total_Revenue','Total_Profit'],inplace = True)
df1
df = df1.transpose()
df =df1
df
# totalsales = df.loc[:, ['Month', 'FaceCream', 'FaceWash','ToothPaste','Soap','Shampo','Moisturizer']].groupby('Month').sum()
# df['Total_Revenue1'] =df[0]+df[1]+df[2]+df[3]+df[4]+df[5]+df[6]+df[7]+df[8]+df[9]+df[10]+df[11]

# df.Total_Revenue1.plot(kind = 'pie',autopct = '%.f')


# In[ ]:





# ### 89. Create line plots for Soap & Facewash of all months in a single plot using Subplot

# In[140]:


fig,axes=plt.subplots(nrows=1,ncols=2)
axes[0,0].line(company_data.Soap)
axes[0,1].line(company_data.Facewash)


# ### 90. Create Box Plot for Total Profit variable

# In[139]:


plt.boxplot(company_data.Total_Profit)


# In[ ]:




