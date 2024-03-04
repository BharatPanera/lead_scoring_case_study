#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# ##### Steps to follow:
# 1. Reading and understanding the data
# 2. Preparing data for modeling (split into train & test)
# 3. Training the model
# 4. Prediction and evaluation on the test data
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# # 1. Reading and understand the Dataset

# In[2]:


lead_data = pd.read_csv('Leads.csv')
lead_data.head()


# In[3]:


print(lead_data.shape)


# In[4]:


print(lead_data.columns)


# In[5]:


lead_data.describe()


# In[6]:


lead_data.info()


# # 2. EDA

# - The majority of the columns have an object data type, which we will convert accordingly.

# In[7]:


# Function to print the unique values from the cols
def get_unique_vals(data_set):
    for col in data_set.columns:
        unique_values = data_set[col].unique()
        print(f"'{col}': {unique_values}\n")

get_unique_vals(lead_data)


# - In some of the columns, the "Select" value is observed, which might be due to the non-selection of the option. Hence, we can replace the select values with null values.

# In[8]:


lead_data = lead_data.replace('Select', pd.NA)


# - We have replaced "Select" with null values. Now, we can move ahead with handling missing values.

# ### 2.1. - Handle Missing Values

# In[9]:


# identifying the null valus in data
print(lead_data.isnull().sum())


# In[10]:


# Identify the null values as a percentage.
print(lead_data.isnull().sum()*100/len(lead_data))


# - Drop the columns with null values exceeding 40%.

# In[11]:


# Function that will remove the columns having null values based on the cutoff
def drop_null_vals(data_set, cut_off):
    
    empty_vals = list(data_set.columns[100*data_set.isnull().mean() > cut_off])
    print(f"List of columns having more than {cut_off}% empty values: {empty_vals}\n")
    print(f"Total columns: {len(data_set.columns)}")
    print(f"Removed columns: {len(empty_vals)}")
    
    data_set = data_set.drop(empty_vals, axis=1)
    print(f"Remaining columns: {len(data_set.columns)}")
    
    return data_set

lead_data = drop_null_vals(lead_data, 40)


# In[12]:


print(lead_data.shape)


# In[13]:


print(lead_data.isnull().sum()*100/len(lead_data))


# In[14]:


lead_data.info()


# ### 2.2. - Categorical columns analysis

# In[15]:


get_unique_vals(lead_data)


# In[16]:


# List of categorical columns
catagorical_cols = ['Lead Origin', 'Lead Source', 'Last Activity', 'Country', 'Specialization', 'What is your current occupation', 
                    'What matters most to you in choosing a course', 'Tags', 'City', 'Last Notable Activity']


# In[17]:


# List of columns to remove
cols_to_remove = []


# In[18]:


# Function that will create graph of the variable
def create_plot(col_name, figsize=(15, 5)):
    plt.figure(figsize=figsize)
    sns.countplot(x=col_name, hue='Converted', data=lead_data)
    plt.xticks(rotation=90)
    return plt.show()


# - We have created a list of catagorical variables.
# - We have created a empty list to remove columns that are not important for the analysis, we will append the columns accordingly
# - We have created a function to create plot of the variable
# - Now we will look into each column one by one for analysis.

# **1. Lead Origin**

# In[19]:


# Check the unique values of the variable
lead_data['Lead Origin'].unique()


# In[20]:


# Check the value count of the variable
lead_data['Lead Origin'].value_counts(dropna=False)


# In[21]:


create_plot('Lead Origin', figsize=(8, 5))


# **Conclusion:**
# 1. "Landing page submissions" and "API" have identified the most leads, and the majority of the leads converted from these sources.
# 2. "Lead Add Form" has a high lead conversion rate but the total identifies leads is less compared to "Landing page submissions" and "API".
# 3. "Lead Import" and "Quick Add Form" have gathered the least leads and have very low conversion rates.
# 4. We can focus on improving the conversion rate of "API" and "Landing page submissions".

# **2. Lead Source**

# In[22]:


# Check the unique values of the variable
lead_data['Lead Source'].unique()


# In[23]:


# Check the value count of the variable
lead_data['Lead Source'].value_counts(dropna=False)


# - As we can see, there are many lead sources, some of which have collected very few leads.
# - Therefore, we will keep the top 5 lead sources in their original form and replace the values of the others with "Others".
# - We will also replace the null values with "Others"

# In[24]:


# Replace the values with others
lead_data['Lead Source'] = lead_data['Lead Source'].replace(['Welingak Website', 'Referral Sites', 'Facebook', 'bing', 'google', 'Click2call', 
                                                             'Press_Release', 'Social Media', 'Live Chat', 'youtubechannel', 'testone', 
                                                             'Pay per Click Ads', 'welearnblog_Home', 'WeLearn', 'blog', 'NC_EDM'], 'other_lead_sources')


# In[25]:


# Replace the null values with others
lead_data['Lead Source'] = lead_data['Lead Source'].fillna('other_lead_sources')


# In[26]:


# Check the value count of the variable
lead_data['Lead Source'].value_counts(dropna=False)


# In[27]:


# Create plot for better understanding
create_plot("Lead Source")


# **Conclusion:**
# 1. "Direct traffic" and "Google" have gathered the most leads and have the highest conversion rate.
# 2. "Olark chat" and "Organic search" have almost similar conversion rates, but the non-conversion rate is lower for "Organic search."
# 3. "Reference" has the highest conversion rate, and most of the leads successfully converted.
# 4. In order to improve the conversion rate, we should focus more on "Reference" by providing some kind of discount options to the referrals.
# 5. "Others" have a very low conversion rate.

# **3. Last Activity**

# In[28]:


# Check the unique values of the variable
lead_data['Last Activity'].unique()


# In[29]:


# Check the value count of the variable
lead_data['Last Activity'].value_counts(dropna=False)


# - Here, we will keep the top 10 last activities in their original form and replace the values of the others with "Others".
# - We will also replace the null values with "Others"

# In[30]:


# Replace the values with others
lead_data['Last Activity'] = lead_data['Last Activity'].replace(['Unreachable', 'Unsubscribed', 'Had a Phone Conversation', 'Approached upfront', 
                                                                 'View in browser link Clicked', 'Email Received', 'Email Marked Spam', 
                                                                 'Visited Booth in Tradeshow', 'Resubscribed to emails'], 'others')


# In[31]:


# Replace the null values with others
lead_data['Last Activity'] = lead_data['Last Activity'].fillna('others')


# In[32]:


# Check the value count of the variable
lead_data['Last Activity'].value_counts(dropna=False)


# In[33]:


# Create plot for better understanding
create_plot("Last Activity")


# **Conclusion:**
# 1. "Email Opened" and "SMS Sent" have the highest conversion rate.

# **4. Country**

# In[34]:


# Check the unique values of the variable
lead_data['Country'].unique()


# In[35]:


# Check the value count of the variable
lead_data['Country'].value_counts(dropna=False)


# - Since majority of the leads are from india, hence we will replace the null values with india.
# - We will replace the values of the other countries with "Others".

# In[36]:


# Replace the values with others
lead_data['Country'] = lead_data['Country'].replace(['United Kingdom', 'Australia', 'Qatar', 'Bahrain', 'Hong Kong', 'Oman', 'France', 'unknown', 'Kuwait', 
                                                     'South Africa', 'Canada', 'Nigeria', 'Germany', 'Sweden', 'Philippines', 'Uganda', 'Italy', 'Bangladesh', 
                                                     'Netherlands', 'Asia/Pacific Region', 'China', 'Belgium', 'Ghana', 'Kenya', 'Sri Lanka', 'Tanzania', 
                                                     'Malaysia', 'Liberia', 'Switzerland', 'Denmark', 'Russia', 'Vietnam', 'Indonesia','United States', 
                                                     'United Arab Emirates', 'Singapore', 'Saudi Arabia'], 'Other_countries')


# In[37]:


# Replace the null values with others
lead_data['Country'] = lead_data['Country'].fillna('India')


# In[38]:


# Check the value count of the variable
lead_data['Country'].value_counts(dropna=False)


# In[39]:


# Create plot for better understanding
create_plot("Country")


# In[40]:


# Append the column name in empty list to drop it
cols_to_remove.append("Country")


# **Conclusion:**
# 1. We can see that most of the leads are coming from india.
# 2. Since majority of the values in variable "Country" is "India", we can drop this variable.

# **5. Specialization**

# In[41]:


# Check the unique values of the variable
lead_data['Specialization'].unique()


# In[42]:


# Check the value count of the variable
lead_data['Specialization'].value_counts(dropna=False)


# - As we can see the "<NA>" and "NaN" values are more, which means the "Specialization" was not specified during form filling process.
# - We will replace the values of "<NA>" and "NaN" with "not_selected".

# In[43]:


# Replace the null values with others
lead_data['Specialization'] = lead_data['Specialization'].fillna('not_selected')


# In[44]:


# Check the value count of the variable
lead_data['Specialization'].value_counts(dropna=False)


# In[45]:


# Create plot for better understanding
create_plot("Specialization")


# - We can see in the graph that the "Management" specialization have good convertion rate.
# - We can bin all management "Management" specialization into one.

# In[46]:


# Replace the values
lead_data['Specialization'] = lead_data['Specialization'].replace(['Finance Management', 'Human Resource Management', 'Marketing Management', 
                                                                   'Operations Management', 'IT Projects Management', 'Supply Chain Management', 
                                                                   'Healthcare Management', 'Hospitality Management', 'Retail Management'], 'management_specialization')


# In[47]:


# Create plot for better understanding
create_plot("Specialization")


# **Conclusion:**
# 1. As we can see the "management specialization" is a very significant variable for out analysis, It has very high convertion rate.
# 2. In order to improve the lead convertion rate, we can focus on targeting more individuals with a specialization in "Management" through marketing or referrals.

# **6. What is your current occupation**

# In[48]:


# Check the value count of the variable
lead_data['What is your current occupation'].value_counts(dropna=False)


# - We can replace the null values with "Unemployed"

# In[49]:


# Replace the null values
lead_data['What is your current occupation'] = lead_data['What is your current occupation'].fillna('Unemployed')


# In[50]:


# Create plot for better understanding
create_plot("What is your current occupation")


# **Conclusion:**
# - As we can see, "Unemployed" and "Working Professional" categories exhibit a very high conversion rate, indicating a significant variable.
# - We should primarily focus on working professionals since they have a higher conversion rate compared to the non-conversion rate.
# - Unemployed individuals are also important as they might be upskilling themselves in order to seize opportunities.

# **7. What matters most to you in choosing a course**

# In[51]:


# Check the value count of the variable
lead_data['What matters most to you in choosing a course'].value_counts(dropna=False)


# - We can replace the null values with "Better Career Prospects"

# In[52]:


# Replace the null values
lead_data['What matters most to you in choosing a course'] = lead_data['What matters most to you in choosing a course'].fillna('Better Career Prospects')


# In[53]:


# Create plot for better understanding
create_plot("What matters most to you in choosing a course", figsize=(10, 5))


# In[54]:


# Append the column name in empty list to drop it
cols_to_remove.append("What matters most to you in choosing a course")


# **Conclusion:**
# 1. Since majority of the values in variable "What matters most to you in choosing a course" is "Better Career Prospects", we can drop this variable.

# **8. Tags**

# In[55]:


# Check the value count of the variable
lead_data['Tags'].value_counts(dropna=False)


# - We will keep the top 5 tage and replace the remainig tages with the values "other_tags".
# - we will also replace the null values with the values "other_tags".

# In[56]:


# Replace the values
lead_data['Tags'] = lead_data['Tags'].replace(['switched off', 'Busy', 'Lost to EINS', 'Not doing further education', 'Interested  in full time MBA', 
                                               'Graduation in progress', 'invalid number', 'Diploma holder (Not Eligible)', 'wrong number given', 'opp hangup', 
                                               'number not provided', 'in touch with EINS', 'Lost to Others', 'Still Thinking', 
                                               'Want to take admission but has financial', 'In confusion whether part time or DLP', 'Interested in Next batch', 
                                               'Lateral student', 'Shall take in the next coming month', 'University not recognized', 
                                               'Recognition issue (DEC approval)'], 'other_tags')


# In[57]:


# Replace the null values
lead_data['Tags'] = lead_data['Tags'].fillna('other_tags')


# In[58]:


# Create plot for better understanding
create_plot("Tags")


# **9. City**

# In[59]:


# Check the value count of the variable
lead_data['City'].value_counts(dropna=False)


# - We will replace the null values with "Mumbai" as it is appearing the highest

# In[60]:


lead_data['City'] = lead_data['City'].fillna('Mumbai')


# In[61]:


# Check the value count of the variable
lead_data['City'].value_counts(dropna=False)


# In[62]:


create_plot("City")


# **Conclusion:**
# 1. Most of the leads are from "Mumbai".

# **10. Last Notable Activity**

# In[63]:


# Check the value count of the variable
lead_data['Last Notable Activity'].value_counts(dropna=False)


# In[64]:


create_plot('Last Notable Activity')


# In[65]:


# Append the column name in empty list to drop it
cols_to_remove.append("Last Notable Activity")


# **Conclusion:**
# 1. It is not adding much values to the analysis, we can remove this column.

# ##### Since we have replaced the null values from many variables, we will check the percentage of null values of the variables 

# In[66]:


# Identify the null values as a percentage.
print(lead_data.isnull().sum()*100/len(lead_data))


# - As we can see, almost all the null values are taken care of, yet some null values are present in the "TotalVisits" and "Page Views Per Visit" variables.
# - We will remove these null values.

# In[67]:


lead_data = lead_data.dropna()


# In[68]:


# Identify the null values as a percentage.
print(lead_data.isnull().sum()*100/len(lead_data))


# - All the null values are removed.

# - There are many columns with values "Yes" and "No".
# - We will check those columns.

# In[69]:


yes_no_columns = []
for col in lead_data.columns:
    if lead_data[col].isin(['Yes', 'No', 'yes', 'no']).any():
        yes_no_columns.append(col)
        unique_values = lead_data[col].unique()
        print(f"'{col}': {unique_values}")
    else:
        pass


# In[70]:


print(yes_no_columns)


# - Here we will check the value count, and if the column is imbalanced then we will remove the column.

# In[71]:


imbalance_cols = {}
for col in yes_no_columns:
    
    counts = lead_data[col].value_counts()
    print(f"Values present in '{col}':\n {counts}")

    total_values = len(lead_data[col])
    print(f"Total values in '{col}': {total_values}")
    print("________________________________________________________________")

    # checking imbalance columns
    val_perc = round(counts/total_values*100, 2)
    if val_perc.max() > 90:
        imbalance_cols[col] = val_perc.to_dict()

        # append the column into list to remove it later
        cols_to_remove.append(col)
    else:
        pass


# - Imbalanced columns

# In[72]:


# imbalanced columns with the percentage
print("Imbalance ratio of the columns: \n")
for col, counts in imbalance_cols.items():
    print(f"{col}: {counts}")


# In[73]:


lead_data.head(10)


# - Since "Prospect ID" and "Lead Number" columns are not important for the analysis, we will remove these columns as well.

# In[74]:


cols_to_remove.extend(['Prospect ID', 'Lead Number'])
print(cols_to_remove)


# In[75]:


lead_data = lead_data.drop(cols_to_remove, 1)
lead_data.shape


# ### 2.3. - Numerical columns analysis

# In[76]:


lead_data.info()


# In[77]:


numerical_cols = ['Converted', 'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']


# In[78]:


# Function to create box plot
def create_box_plot(column_name):
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=lead_data[column_name])
    plt.title(f"Box Plot of '{column_name}'")
    plt.xlabel(column_name)
    plt.ylabel('Values')
    return plt.show()


# - Created a list of numerical variables.
# - Created a function to create box plot.

# - Check the correlation matrix of numerical columns

# In[79]:


# Calculate the correlation matrix
correlation_matrix = lead_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numeric Columns')
plt.show()


# **1. Converted**

# In[80]:


lead_data['Converted'].describe()


# In[81]:


lead_data['Converted'].value_counts()


# In[82]:


print(f"The percentage of the converted: {((lead_data['Converted'] == 1).sum() / len(lead_data['Converted'])) * 100}")


# **2. TotalVisits**

# In[83]:


lead_data['TotalVisits'].describe()


# In[84]:


lead_data['TotalVisits'].value_counts()


# In[85]:


create_box_plot('TotalVisits')


# - As we can see, there are outliers that need to be handled.
# - Here, we will use the winsorization method, which handles outliers by capping extreme values at a specified percentile.

# In[86]:


# Calculate the 5th and 95th percentiles of the 'TotalVisits' column
percentiles = lead_data['TotalVisits'].quantile([0.05, 0.95]).values

# Cap outliers by replacing values below the 5th percentile with the 5th percentile value
lead_data.loc[lead_data['TotalVisits'] <= percentiles[0], 'TotalVisits'] = percentiles[0]

# Cap outliers by replacing values above the 95th percentile with the 95th percentile value
lead_data.loc[lead_data['TotalVisits'] >= percentiles[1], 'TotalVisits'] = percentiles[1]


# In[87]:


create_box_plot('TotalVisits')


# **3. Total Time Spent on Website**

# In[88]:


lead_data['Total Time Spent on Website'].describe()


# In[89]:


lead_data['Total Time Spent on Website'].value_counts()


# In[90]:


create_box_plot('Total Time Spent on Website')


# - As we can see there are no outliers

# **4. Page Views Per Visit**

# In[91]:


lead_data['Page Views Per Visit'].describe()


# In[92]:


lead_data['Page Views Per Visit'].value_counts()


# In[93]:


create_box_plot('Page Views Per Visit')


# - As we can see, there are outliers that need to be handled.
# - Here, we will use the winsorization method, which handles outliers by capping extreme values at a specified percentile.

# In[94]:


# Calculate the 5th and 95th percentiles of the 'TotalVisits' column
percentiles = lead_data['Page Views Per Visit'].quantile([0.05, 0.95]).values

# Cap outliers by replacing values below the 5th percentile with the 5th percentile value
lead_data.loc[lead_data['Page Views Per Visit'] <= percentiles[0], 'Page Views Per Visit'] = percentiles[0]

# Cap outliers by replacing values above the 95th percentile with the 95th percentile value
lead_data.loc[lead_data['Page Views Per Visit'] >= percentiles[1], 'Page Views Per Visit'] = percentiles[1]


# In[95]:


create_box_plot('Page Views Per Visit')


# - checking the correlation matrix of the numerical columns

# In[96]:


# Calculate the correlation matrix
correlation_matrix = lead_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numeric Columns')
plt.show()


# # 3. Preparing data for modeling

# #### 3.1. Create dummy variables

# In[97]:


lead_data.info()


# - Making a list of catagorical columns

# In[98]:


catagorical_cols = lead_data.select_dtypes(include=['object']).columns
catagorical_cols


# In[99]:


get_unique_vals(lead_data)


# **1. Lead Origin**

# In[100]:


lead_data['Lead Origin'].value_counts()


# In[101]:


# Create dummy variable
Lead_origin = pd.get_dummies(lead_data['Lead Origin'], prefix='lead_origin')

# Drop least correlated column manually
Lead_origin = Lead_origin.drop(columns=['lead_origin_Lead Add Form'], axis=1)

#Concating the dummy variables with the original data
lead_data = pd.concat([lead_data, Lead_origin], axis=1)

lead_data.info()


# **2. Lead Source**

# In[102]:


# Create dummy variable
Lead_Source = pd.get_dummies(lead_data['Lead Source'], prefix='lead_source')

# Drop least correlated column manually
Lead_Source = Lead_Source.drop(columns=['lead_source_other_lead_sources'], axis=1)

#Concating the dummy variables with the original data
lead_data = pd.concat([lead_data, Lead_Source], axis=1)

lead_data.info()


# **3. Last Activity**

# In[103]:


# Create dummy variable
Last_Activity = pd.get_dummies(lead_data['Last Activity'], prefix='last_activity')

# Drop least correlated column manually
Last_Activity = Last_Activity.drop(columns=['last_activity_others'], axis=1)

#Concating the dummy variables with the original data
lead_data = pd.concat([lead_data, Last_Activity], axis=1)

lead_data.info()


# **4. Specialization**

# In[104]:


# Create dummy variable
Specialization = pd.get_dummies(lead_data['Specialization'], prefix='specialization')

# Drop least correlated column manually
Specialization = Specialization.drop(columns=['specialization_Services Excellence'], axis=1)

#Concating the dummy variables with the original data
lead_data = pd.concat([lead_data, Specialization ], axis=1)

lead_data.info()


# **5. What is your current occupation**

# In[105]:


# Create dummy variable
current_occupation = pd.get_dummies(lead_data['What is your current occupation'], prefix='current_occupation')

# Drop least correlated column manually
current_occupation = current_occupation.drop(columns=['current_occupation_Other'], axis=1)

#Concating the dummy variables with the original data
lead_data = pd.concat([lead_data, current_occupation ], axis=1)

lead_data.info()


# **6. City**

# In[106]:


# Create dummy variable
City = pd.get_dummies(lead_data['City'], prefix='city')

# Drop least correlated column manually
City = City.drop(columns=['city_Tier II Cities'], axis=1)

#Concating the dummy variables with the original data
lead_data = pd.concat([lead_data, City ], axis=1)

lead_data.info()


# **7. A free copy of Mastering The Interview**

# In[107]:


lead_data['A free copy of Mastering The Interview'] = lead_data['A free copy of Mastering The Interview'].replace({'Yes': 1, 'No': 0})
lead_data['A free copy of Mastering The Interview']


# - After concatinating the dummy variables into original dataset, we can remove the original categorical columns.
# 

# **8. Tags**

# In[108]:


# Create dummy variable
Tags = pd.get_dummies(lead_data['Tags'], prefix='tags')

# Drop least correlated column manually
Tags = Tags.drop(columns=['tags_Want to take admission but has financial problems'], axis=1)

#Concating the dummy variables with the original data
lead_data = pd.concat([lead_data, Tags], axis=1)

lead_data.info()


# In[109]:


lead_data = lead_data.drop(columns=['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization', 'What is your current occupation', 'City', 'Tags'], axis=1)
lead_data.info()


# - Replacing the space in column name for better understanding

# In[110]:


lead_data.columns = lead_data.columns.str.replace(' ', '_')
lead_data.info()


# In[111]:


get_unique_vals(lead_data)


# #### 3.2. Split data into train and test datasets

# In[112]:


X=lead_data.drop('Converted', axis=1)
X.head()


# In[113]:


y = lead_data['Converted']
y.head


# - Splitting data set into train and test

# In[114]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[115]:


X_train.info()


# In[116]:


get_unique_vals(X_train)


# - Scalling the features
# 

# In[117]:


# - Scalling the features
scaler = StandardScaler()

# Fit on data
X_train[['TotalVisits', 'Total_Time_Spent_on_Website', 'Page_Views_Per_Visit']] = scaler.fit_transform(X_train[['TotalVisits', 'Total_Time_Spent_on_Website', 
                                                                                                                'Page_Views_Per_Visit']])

X_train.head()


# # 4. Train the model

# - Here, we will build the model using Stats model.
# - We will go with the RFE approach to select the best 15 features for the model.

# In[118]:


lr_rfe = LogisticRegression()

rfe = RFE(estimator=lr_rfe, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[119]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[120]:


# Printing the RFE supported columns
rfe_supported_cols = X_train.columns[rfe.support_]
rfe_supported_cols


# In[121]:


# Printing the RFE rejected columns
rfe_rejected_cols = X_train.columns[~rfe.support_]
rfe_rejected_cols


# **1st Model**

# In[122]:


# Adding the constant
X_train_model = sm.add_constant(X_train[rfe_supported_cols])

# Creating the model
lr_sm1 = sm.GLM(y_train,X_train_model, family = sm.families.Binomial())

# fit the model
lr_sm_model = lr_sm1.fit()

#parameters of the model
lr_sm_model.summary()


# - we will drop "lead_source_Organic_Search" variable due to high p-value 

# **2nd Model**

# In[123]:


rfe_supported_cols = rfe_supported_cols.drop('lead_source_Organic_Search', 1)


# In[124]:


# Adding the constant
X_train_model = sm.add_constant(X_train[rfe_supported_cols])

# Creating the model
lr_sm2 = sm.GLM(y_train,X_train_model, family = sm.families.Binomial())

# fit the model
lr_sm_model = lr_sm2.fit()

#parameters of the model
lr_sm_model.summary()


# - Now we will check VIF

# In[125]:


# Function to check VIF
def check_vif(df):
    vif = pd.DataFrame()
    X = df
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif


# In[126]:


check_vif(X_train[rfe_supported_cols])


# - We will remove teh "lead_origin_Landing_Page_Submission" due to high VIF

# **3rd Model**

# In[127]:


rfe_supported_cols = rfe_supported_cols.drop('lead_origin_Landing_Page_Submission', 1)


# In[128]:


# Adding the constant
X_train_model = sm.add_constant(X_train[rfe_supported_cols])

# Creating the model
lr_sm3 = sm.GLM(y_train,X_train_model, family = sm.families.Binomial())

# fit the model
lr_sm_model = lr_sm3.fit()

#parameters of the model
lr_sm_model.summary()


# In[129]:


check_vif(X_train[rfe_supported_cols])


# - Here, we will try removing "lead_origin_Lead_Import" variable and build the model again.

# **4th Model**

# In[130]:


rfe_supported_cols = rfe_supported_cols.drop('lead_origin_Lead_Import', 1)


# In[131]:


# Adding the constant
X_train_model = sm.add_constant(X_train[rfe_supported_cols])

# Creating the model
lr_sm4 = sm.GLM(y_train,X_train_model, family = sm.families.Binomial())

# fit the model
lr_sm_model = lr_sm4.fit()

#parameters of the model
lr_sm_model.summary()


# In[132]:


check_vif(X_train[rfe_supported_cols])


# - As we can see, the VIF is decreased significantly, and p-value is also less for the variables.
# - Now we will derive the Probability on train data.

# In[133]:


rfe_supported_cols


# In[134]:


# Prediction on train set
y_train_pred = lr_sm_model.predict(X_train_model)
y_train_pred[:10]


# In[135]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# - Selecting an arbitrary cut-off probability point of 0.5 to determine the predicted labels.
# - Creating a new column named 'predicted', assigning 1 if Converted_Prob > 0.5, otherwise 0.

# In[136]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# - Confusion matrix

# In[137]:


confusion_metrix = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
print(confusion_metrix)


# - The confusion matrix indicates the following:

# In[138]:


# Predicted     not_converted    converted
# Actual
# not_converted        3774      444
# converted            543       1876


# - The overall accuracy:

# In[139]:


print('Accuracy:',metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[140]:


TP = confusion_metrix[1,1] # true positive 
TN = confusion_metrix[0,0] # true negatives
FP = confusion_metrix[0,1] # false positives
FN = confusion_metrix[1,0] # false negatives


# - Sensitivity

# In[141]:


print("Sensitivity: ",TP / float(TP+FN))


# - Calculate specificity

# In[142]:


print("Specificity: ",TN / float(TN+FP))


# - Calculate false postive rate

# In[143]:


print("False Positive Rate:",FP/ float(TN+FP))


# - Calculate positive predictive value

# In[144]:


print("Positive Predictive Value: ",TP / float(TP+FP))


# - Calculate negative predictive value

# In[145]:


print ("Negative predictive value: ",TN / float(TN+ FN))


# **ROC CURVE**

# In[146]:


# Calculate the ROC curve and get false positive rate, true positive rate, and thresholds
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)

# Calculate the AUC (Area Under the Curve) score
auc_score = metrics.roc_auc_score(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (AUC = {:.2f})'.format(auc_score), color='b')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# - The ROC curve value is 0.95 which is close to 1.

# #### Precision and recall tradeoff

# In[147]:


y_train_pred_final.Converted, y_train_pred_final.Predicted


# In[148]:


# Calculate precision-recall curve
precision, recall, thresholds = metrics.precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)

# Plot precision-recall curve
plt.plot(thresholds, precision[:-1], "g-", label="Precision")
plt.plot(thresholds, recall[:-1], "r-", label="Recall")
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()


# # 5. Prediction and evaluation on the test data

# #### 5.1. Scalling the test data set

# In[149]:


# Transform on data
X_test[['TotalVisits', 'Total_Time_Spent_on_Website', 'Page_Views_Per_Visit']] = scaler.transform(X_test[['TotalVisits', 'Total_Time_Spent_on_Website', 
                                                                                                                'Page_Views_Per_Visit']])


# #### 5.2. Testing the model on test data set

# In[150]:


# Assigning the columns supported by RFE to the X_test 
X_test = X_test[rfe_supported_cols]
X_test.head()


# In[151]:


# Adding a const
X_test_sm = sm.add_constant(X_test)

# Making predictions on the test set
y_test_pred = lr_sm_model.predict(X_test_sm)
y_test_pred[:20]


# In[152]:


# Convert y_test_pred to a dataframe which is an array.
y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()


# In[153]:


# Convert y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df


# In[154]:


# Putting Prospect ID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[155]:


# Remove index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[156]:


# Append y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[157]:


# Rename the column
y_pred_final= y_pred_final.rename(columns={0 : 'Converted_prob'})
y_pred_final.head()


# In[158]:


# Rearrange the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))
y_pred_final.head(100)


# In[159]:


y_pred_final['Final_predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.34 else 0)
y_pred_final.head()


# - Overall accuracy on test data set

# In[160]:


print("Accuracy: ",metrics.accuracy_score(y_pred_final.Converted, y_pred_final.Final_predicted))


# - Calculate confusion matrix on test data set

# In[161]:


confusion_matrix2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.Final_predicted )
confusion_matrix2


# In[162]:


TP = confusion_matrix2[1,1] # true positive 
TN = confusion_matrix2[0,0] # true negatives
FP = confusion_matrix2[0,1] # false positives
FN = confusion_matrix2[1,0] # false negatives


# In[163]:


print("Sensitivity: ",TP / float(TP+FN))


# In[164]:


print("Specificity: ",TN / float(TN+FP))


# - Assign lead score to test data set

# In[165]:


y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))
y_pred_final.head()


# #### **Conclusions:**
# ##### Upon executing the model on the test data, the following results were obtained:
#     1. Accuracy: 87%
#     2. Sensitivity: 88%
#     3. Specificity: 87.44%
# 
# - Accuracy, Sensitivity, and Specificity values of the test set are around 87%, 88%, and 87.44%, respectively, which closely align with the respective values calculated using the trained set.

# #### Metrics comparison
# Train data:
# - Accuracy: 88.66%
# - Sensitivity:  77.55%
# - Specificity:  95.47%
# 
# Test Data:
# - Accuracy:  87.84%
# - Sensitivity:  88.48%
# - Specificity:  87.44%
