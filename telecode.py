#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import Counter
#from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, GridSearchCV

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
#from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

from sklearn.decomposition import PCA

get_ipython().run_line_magic('matplotlib', 'inline')
RANDOM_STATE = 42


# In[2]:


from sklearn.preprocessing import LabelEncoder


# In[3]:


pip install imblearn --user


# In[4]:


from imblearn.over_sampling import SMOTE


# In[5]:


data = pd.read_csv('telecom_churn_data.csv')
data.head().T


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


'''
Method Name : get_cols_split
Input(s)    : 1 DataFrame
Output(s)   : 6 lists  
Description :
- This method reads the columns in the given dataframe and splits them into various categories such as months, date related cols 
  and other common cols  
- Then returns the column lists as output for each of these categories. 
'''
def get_cols_split(df):

    col_len = len(df.columns)

    jun_cols = []
    jul_cols = []
    aug_cols = []
    sep_cols = []
    common_cols = []
    date_cols = []
    
    for i in range(0, col_len):
        if any(pd.Series(df.columns[i]).str.contains('_6|jun')):
            jun_cols.append(df.columns[i])
        elif any(pd.Series(df.columns[i]).str.contains('_7|jul')):
            jul_cols.append(df.columns[i])
        elif any(pd.Series(df.columns[i]).str.contains('_8|aug')):
            aug_cols.append(df.columns[i])
        elif any(pd.Series(df.columns[i]).str.contains('_9|sep')):
            sep_cols.append(df.columns[i])
        else:
            common_cols.append(df.columns[i])
        
        if any(pd.Series(df.columns[i]).str.contains('date')):
            date_cols.append(df.columns[i])
            
    return jun_cols,jul_cols,aug_cols,sep_cols,common_cols,date_cols


# In[9]:


'''
Method Name : get_cols_sub_split
Input(s)    : 1 list
Output(s)   : 4 lists  
Description :
- This method gets the columns list as input and splits them into various sub-categories such as call_usage, recharge columns, 
  incoming and outgoing related cols  
- Then returns the column lists as output for each of these sub-categories. 
'''
def get_cols_sub_split(col_list):
    call_usage_cols = []
    recharge_cols = []
    ic_usage_cols = []
    og_usage_cols = []

    call_usage_search_for = ['og','ic','mou']

    for i in range(0, len(col_list)):
        if any(pd.Series(col_list[i]).str.contains('|'.join(['rech','rch']))):
            recharge_cols.append(col_list[i])
        elif any(pd.Series(col_list[i]).str.contains('|'.join(call_usage_search_for))):
            call_usage_cols.append(col_list[i])

        if any(pd.Series(col_list[i]).str.contains('ic')):
            ic_usage_cols.append(col_list[i])
        elif any(pd.Series(col_list[i]).str.contains('og')):
            og_usage_cols.append(col_list[i])
            
    return call_usage_cols,recharge_cols,ic_usage_cols,og_usage_cols


# In[10]:


# Get the average recharge amount for 6 and 7 month
data['avg_rech_amt_6_7'] = ( data['total_rech_amt_6'] + data['total_rech_amt_7'] ) / 2

# Get the data greater than 70th percentile of average recharge amount
data = data.loc[(data['avg_rech_amt_6_7'] > np.percentile(data['avg_rech_amt_6_7'], 70))]

# drop the average column
data.drop(['avg_rech_amt_6_7'], axis=1, inplace=True)

print(data.shape)


# In[11]:


# mark the rows as churn if the sum of the total mou and vol of 9 month is 0
tag_churn_cols = ['total_ic_mou_9', 'total_og_mou_9', 'vol_2g_mb_9', 'vol_3g_mb_9']
data['churn'] = np.where(data[tag_churn_cols].sum(axis=1) == 0, 1, 0 )


# In[12]:


data['churn'].value_counts()


# In[13]:


print('Churn Rate : {0}%'.format(round(((sum(data['churn'])/len(data['churn']))*100),2)))


# In[14]:


# Get the columns split by months
jun_cols, jul_cols, aug_cols, sep_cols, common_cols, date_cols = get_cols_split(data)


# In[15]:


# Drop all the sep columns
data.drop(sep_cols, axis=1, inplace=True)


# In[16]:


# Get the unique count
for col in data.columns:
    print(col, len(data[col].unique()))


# In[17]:


data[['mobile_number','circle_id','last_date_of_month_6','last_date_of_month_7','last_date_of_month_8',           'loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou','std_og_t2c_mou_6','std_og_t2c_mou_7','std_og_t2c_mou_8',           'std_ic_t2o_mou_6','std_ic_t2o_mou_7','std_ic_t2o_mou_8']].head(5)


# In[18]:


# Remove unwanted columns
data.drop(['mobile_number','circle_id','last_date_of_month_6','last_date_of_month_7','last_date_of_month_8',           'loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou','std_og_t2c_mou_6','std_og_t2c_mou_7','std_og_t2c_mou_8',           'std_ic_t2o_mou_6','std_ic_t2o_mou_7','std_ic_t2o_mou_8'], axis=1, inplace=True)


# In[19]:


data[['total_rech_data_6','av_rech_amt_data_6','max_rech_data_6']].head()


# In[20]:


#Rename the cols to correct name
data = data.rename(columns={'av_rech_amt_data_6':'total_rech_amt_data_6','av_rech_amt_data_7':'total_rech_amt_data_7','av_rech_amt_data_8':'total_rech_amt_data_8'})


# In[21]:


df = data.isnull().sum().reset_index(name='missing_cnt')
df.loc[df['missing_cnt']>0].sort_values('missing_cnt', ascending=False)


# In[22]:


# Get the columns split to months
jun_cols, jul_cols, aug_cols, sep_cols, common_cols, date_cols = get_cols_split(data)


# In[23]:


# Get the columns sub split for each months
jun_call_usage_cols, jun_recharge_cols, jun_ic_usage_cols, jun_og_usage_cols = get_cols_sub_split(jun_cols)
jul_call_usage_cols, jul_recharge_cols, jul_ic_usage_cols, jul_og_usage_cols = get_cols_sub_split(jul_cols)
aug_call_usage_cols, aug_recharge_cols, aug_ic_usage_cols, aug_og_usage_cols = get_cols_sub_split(aug_cols)


# In[24]:


# Filling the missing values of fb and night pack user columns as 2, as this could be an another type that was marked as NA
cols_6 = ['fb_user_6','night_pck_user_6']
cols_7 = ['fb_user_7','night_pck_user_7']
cols_8 = ['fb_user_8','night_pck_user_8']

data[cols_6] = data[cols_6].fillna(2)
data[cols_7] = data[cols_7].fillna(2)
data[cols_8] = data[cols_8].fillna(2)


# In[25]:


# filling the missing values as 0
cols_6 = ['count_rech_3g_6','max_rech_data_6','total_rech_amt_data_6','arpu_3g_6','total_rech_data_6','count_rech_2g_6','arpu_2g_6']
cols_7 = ['count_rech_3g_7','max_rech_data_7','total_rech_amt_data_7','arpu_3g_7','total_rech_data_7','count_rech_2g_7','arpu_2g_7']
cols_8 = ['count_rech_3g_8','max_rech_data_8','total_rech_amt_data_8','arpu_3g_8','total_rech_data_8','count_rech_2g_8','arpu_2g_8']

data[cols_6] = data[cols_6].fillna(0)
data[cols_7] = data[cols_7].fillna(0)
data[cols_8] = data[cols_8].fillna(0)


# In[26]:


# filling the missing values as 0 for month columns
data[jun_call_usage_cols] = data[jun_call_usage_cols].fillna(0)
data[jul_call_usage_cols] = data[jul_call_usage_cols].fillna(0)
data[aug_call_usage_cols] = data[aug_call_usage_cols].fillna(0)


# In[27]:


# Leaving date cols as null intentionally for feature engineering
df = data.isnull().sum().reset_index(name='missing_cnt')
df.loc[df['missing_cnt']>0].sort_values('missing_cnt', ascending=False)


# In[28]:


sns.countplot(x='churn', data=data)


# In[29]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20, 6))

# distribution plot for aon
sns.distplot(data['aon'], ax=ax1)

# bin the aon column with yearwise segments and plot the counts for each segments
bins = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
labels = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#sns.countplot(pd.cut(round(((data['aon']/30)/12),1), bins = bins, labels = labels ), ax =ax1)
pd.crosstab(pd.cut(round(((data['aon']/30)/12),1), bins = bins, labels = labels ), data['churn']).plot(kind='bar', stacked=True, ax = ax2)


# In[30]:


cols = [['loc_ic_mou_6','std_ic_mou_6','isd_ic_mou_6','roam_ic_mou_6','spl_ic_mou_6','ic_others_6','total_ic_mou_6'],
        ['loc_ic_mou_7','std_ic_mou_7','isd_ic_mou_7','roam_ic_mou_7','spl_ic_mou_7','ic_others_7','total_ic_mou_7'],
        ['loc_ic_mou_8','std_ic_mou_8','isd_ic_mou_8','roam_ic_mou_8','spl_ic_mou_8','ic_others_8','total_ic_mou_8']]

# column description stats
for i in range(0,3):
    display(data[cols[i]].describe())

# plot for the incoming calls usage
plt.figure(figsize=(18, 5))
for i in range(0,3):
    plt.subplot(1,3,i+1)
    X = pd.concat([data[cols[i]], data['churn']], axis=1)
    X = pd.melt(X,id_vars="churn",var_name="features",value_name='value')
    sns.boxplot(x="features", y="value", hue="churn", data=X)
    plt.xticks(rotation=90)    
    plt.suptitle('Incoming Calls Usage')


# In[31]:


cols = [['loc_og_mou_6','std_og_mou_6','isd_og_mou_6','roam_og_mou_6','spl_og_mou_6','og_others_6','total_og_mou_6'],
        ['loc_og_mou_7','std_og_mou_7','isd_og_mou_7','roam_og_mou_7','spl_og_mou_7','og_others_7','total_og_mou_7'],
        ['loc_og_mou_8','std_og_mou_8','isd_og_mou_8','roam_og_mou_8','spl_og_mou_8','og_others_8','total_og_mou_8']]

# column description stats
for i in range(0,3):
    display(data[cols[i]].describe())

# plot for the outgoing calls usage
plt.figure(figsize=(18, 5))
#ic call usage
for i in range(0,3):
    plt.subplot(1,3,i+1)
    X = pd.concat([data[cols[i]], data['churn']], axis=1)
    X = pd.melt(X,id_vars="churn",var_name="features",value_name='value')
    sns.boxplot(x="features", y="value", hue="churn", data=X)
    plt.xticks(rotation=90)    
    plt.suptitle('Outgoing Calls Usage')


# In[32]:


cols = [['onnet_mou_6','offnet_mou_6','loc_ic_t2t_mou_6','loc_ic_t2m_mou_6','loc_ic_t2f_mou_6','std_ic_t2t_mou_6','std_ic_t2m_mou_6','std_ic_t2f_mou_6'],
        ['loc_og_t2t_mou_6','loc_og_t2m_mou_6','loc_og_t2f_mou_6','loc_og_t2c_mou_6','std_og_t2t_mou_6','std_og_t2m_mou_6','std_og_t2f_mou_6'],
        ['onnet_mou_7','offnet_mou_7','loc_ic_t2t_mou_7','loc_ic_t2m_mou_7','loc_ic_t2f_mou_7','std_ic_t2t_mou_7','std_ic_t2m_mou_7','std_ic_t2f_mou_7'],
        ['loc_og_t2t_mou_7','loc_og_t2m_mou_7','loc_og_t2f_mou_7','loc_og_t2c_mou_7','std_og_t2t_mou_7','std_og_t2m_mou_7','std_og_t2f_mou_7'],
        ['onnet_mou_8','offnet_mou_8','loc_ic_t2t_mou_8','loc_ic_t2m_mou_8','loc_ic_t2f_mou_8','std_ic_t2t_mou_8','std_ic_t2m_mou_8','std_ic_t2f_mou_8'],
        ['loc_og_t2t_mou_8','loc_og_t2m_mou_8','loc_og_t2f_mou_8','loc_og_t2c_mou_8','std_og_t2t_mou_8','std_og_t2m_mou_8','std_og_t2f_mou_8']]

# column description stats
for i in range(0,6):
    display(data[cols[i]].describe())

# plot for the operatorwise calls usage
plt.figure(figsize=(18, 18))
plt.subplots_adjust(hspace=0.5)
for i in range(0,6):
    plt.subplot(3,2,i+1)
    X = pd.concat([data[cols[i]], data['churn']], axis=1)
    X = pd.melt(X,id_vars="churn",var_name="features",value_name='value')
    sns.boxplot(x="features", y="value", hue="churn", data=X)
    plt.xticks(rotation=90)    
    plt.suptitle('Operatorwise Calls Usage')


# In[33]:


# Let's derive total recharge amount for voice with the diff recharge amount of total and data
data['total_rech_amt_voice_6'] = np.where((data['total_rech_amt_6'] >= data['total_rech_amt_data_6']), (data['total_rech_amt_6'] - data['total_rech_amt_data_6']), 0)
data['total_rech_amt_voice_7'] = np.where((data['total_rech_amt_7'] >= data['total_rech_amt_data_7']), (data['total_rech_amt_7'] - data['total_rech_amt_data_7']), 0)
data['total_rech_amt_voice_8'] = np.where((data['total_rech_amt_8'] >= data['total_rech_amt_data_8']), (data['total_rech_amt_8'] - data['total_rech_amt_data_8']), 0)

cols = [
        ['total_rech_amt_6','total_rech_amt_7','total_rech_amt_8'],
        ['total_rech_amt_voice_6','total_rech_amt_voice_7','total_rech_amt_voice_8'],
        ['total_rech_amt_data_6','total_rech_amt_data_7','total_rech_amt_data_8'],
        ['max_rech_amt_6','max_rech_amt_7','max_rech_amt_8']
       ]

# column description stats
for i in range(0,4):
    display(data[cols[i]].describe())

# plot for the recharge amount columns
plt.figure(figsize=(18, 10))
plt.subplots_adjust(hspace=0.5)
for i in range(0,4):
    plt.subplot(2,2,i+1)
    X = pd.concat([data[cols[i]], data['churn']], axis=1)
    X = pd.melt(X,id_vars="churn",var_name="features",value_name='value')
    sns.boxplot(x="features", y="value", hue="churn", data=X)
    plt.xticks(rotation=90)    
    plt.suptitle('Recharge Amount')


# In[34]:


cols = [
        ['total_rech_num_6','total_rech_num_7','total_rech_num_8'],
        ['total_rech_data_6','total_rech_data_7','total_rech_data_8'],
        ['max_rech_data_6','max_rech_data_7','max_rech_data_8'],
        ['count_rech_2g_6','count_rech_2g_7','count_rech_2g_8'],
        ['count_rech_3g_6','count_rech_3g_7','count_rech_3g_8'] 
       ]

# column description stats
for i in range(0,5):
    display(data[cols[i]].describe())

# plot for the recharge count columns
plt.figure(figsize=(18, 10))
plt.subplots_adjust(hspace=0.5)
for i in range(0,5):
    plt.subplot(2,3,i+1)
    X = pd.concat([data[cols[i]], data['churn']], axis=1)
    X = pd.melt(X,id_vars="churn",var_name="features",value_name='value')
    sns.boxplot(x="features", y="value", hue="churn", data=X)
    plt.xticks(rotation=90)    
    plt.suptitle('Recharge Count')


# In[35]:


cols = [
        ['arpu_6','arpu_7','arpu_8'],
        ['arpu_2g_6','arpu_2g_7','arpu_2g_8'],
        ['arpu_3g_6','arpu_3g_7','arpu_3g_8']
       ]

# column description stats
for i in range(0,3):
    display(data[cols[i]].describe())

# plot for the arpu
plt.figure(figsize=(18, 5))
for i in range(0,3):
    plt.subplot(1,3,i+1)
    X = pd.concat([data[cols[i]], data['churn']], axis=1)
    X = pd.melt(X,id_vars="churn",var_name="features",value_name='value')
    sns.boxplot(x="features", y="value", hue="churn", data=X)
    plt.xticks(rotation=90)    
    plt.suptitle('Arpu')


# In[36]:


cols = [
        ['vol_2g_mb_6','vol_2g_mb_7','vol_2g_mb_8'],
        ['vol_3g_mb_6','vol_3g_mb_7','vol_3g_mb_8'],
        ['night_pck_user_6','night_pck_user_7','night_pck_user_8'],
        ['fb_user_6','fb_user_7','fb_user_8'],
        ['monthly_2g_6','monthly_2g_7','monthly_2g_8'],
        ['monthly_3g_6','monthly_3g_7','monthly_3g_8'],
        ['sachet_2g_6','sachet_2g_7','sachet_2g_8'],
        ['sachet_3g_6','sachet_3g_7','sachet_3g_8'],
        ['jun_vbc_3g','jul_vbc_3g','aug_vbc_3g']
       ]

# column description stats
for i in range(0,9):
    display(data[cols[i]].describe())

# plot for the 2g-3g volume
plt.figure(figsize=(18, 15))
plt.subplots_adjust(hspace=0.5)
for i in range(0,9):
    plt.subplot(3,3,i+1)
    X = pd.concat([data[cols[i]], data['churn']], axis=1)
    X = pd.melt(X,id_vars="churn",var_name="features",value_name='value')
    sns.boxplot(x="features", y="value", hue="churn", data=X)
    plt.xticks(rotation=90)    
    plt.suptitle('2G-3G Volume')


# In[37]:


cols_to_exclude = ['night_pck_user_6','night_pck_user_7','night_pck_user_8',
                   'fb_user_6','fb_user_7','fb_user_8',
                   'monthly_2g_6','monthly_2g_7','monthly_2g_8',
                   'monthly_3g_6','monthly_3g_7','monthly_3g_8',
                   'sachet_2g_6','sachet_2g_7','sachet_2g_8',
                   'sachet_3g_6','sachet_3g_7','sachet_3g_8',
                   'date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8','date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8',
                   'spl_ic_mou_6','spl_ic_mou_7','spl_ic_mou_8','spl_og_mou_6','og_others_6','spl_og_mou_7','og_others_7','spl_og_mou_8','og_others_8',
                   'loc_og_t2c_mou_6','std_og_t2f_mou_6','std_ic_t2f_mou_6','loc_ic_t2f_mou_6',
                   'loc_og_t2c_mou_7','std_og_t2f_mou_7','std_ic_t2f_mou_7','loc_ic_t2f_mou_7',
                   'loc_og_t2c_mou_8','std_og_t2f_mou_8','std_ic_t2f_mou_8','loc_ic_t2f_mou_8',
                   'aon','churn'
                  ]
cols = list(set(data.columns).difference(set(cols_to_exclude)))

# iterate through the columns and cap the values with the 99th percentile
for col in cols:
    percentiles = data[col].quantile([0.01,0.99]).values
    #data[col][data[col] <= percentiles[0]] = percentiles[0]
    data[col][data[col] >= percentiles[1]] = percentiles[1]


# In[38]:


# remove the outliers with specific columns
data = data.loc[~(
                    ((data['roam_og_mou_8'] > 2200) & (data['churn'] == 1)) |
                    ((data['arpu_7'] > 10000) & (data['churn'] == 1)) |
                    ((data['loc_og_mou_8'] > 2000) & (data['churn'] == 1)) |
                    ((data['loc_ic_mou_7'] > 4000) & (data['churn'] == 1)) |
                    ((data['std_og_mou_7'] > 7000) & (data['churn'] == 1)) |
                    ((data['vol_2g_mb_8'] > 2500) & (data['churn'] == 1)) 
                 )
               ]


# In[39]:


# Convert date columns to date format
for col in date_cols:
    data[col] = pd.to_datetime(data[col], format='%m/%d/%Y')


# In[40]:


cols = ['date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8']
# get the recent date of recharge in the last 3 months
data['last_rech_date'] = data[cols].max(axis=1)
# get the number of days from the recent recharge date till the last date of august month
data['days_since_last_rech'] = np.floor(( pd.to_datetime('2014-08-31', format='%Y-%m-%d') - data['last_rech_date'] ).astype('timedelta64[D]'))
# fill the null values as 0
data['days_since_last_rech'] = data['days_since_last_rech'].fillna(0)

# subtract it from 3 to add higher weightage for values present in all the columns. 
# len(cols) = 3,  means present in all columns, 0 means not present in any column
data['rech_weightage'] = len(cols) - (data[cols].isnull().sum(axis=1))
data.drop(['last_rech_date','date_of_last_rech_6','date_of_last_rech_7','date_of_last_rech_8'], axis=1, inplace=True)


cols = ['date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8']
# get the recent date of recharge data in the last 3 months
data['last_rech_data_date'] = data[cols].max(axis=1)
# get the number of days from the recent recharge data date till the last date of august month
data['days_since_last_data_rech'] = np.floor(( pd.to_datetime('2014-08-31', format='%Y-%m-%d') - data['last_rech_data_date'] ).astype('timedelta64[D]'))
# fill the null values as 0
data['days_since_last_data_rech'] = data['days_since_last_data_rech'].fillna(0)

# subtract it from 3 to add higher weightage for values present in all the columns. 
# len(cols) = 3, means present in all columns, 0 means not present in any column
data['rech_data_weightage'] = len(cols) - (data[cols].isnull().sum(axis=1))

# drop the unwanted columns
data.drop(['last_rech_data_date','date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8'], axis=1, inplace=True)


# In[41]:


# network columns
# get the mean of onnet mou in the last 3 months
cols = ['onnet_mou_6','onnet_mou_7','onnet_mou_8']
data['mean_onnet_mou'] = round(data[cols].mean(axis=1),2)

# get the mean of offnet mou in the last 3 months
cols = ['offnet_mou_6','offnet_mou_7','offnet_mou_8']
data['mean_offnet_mou'] = round(data[cols].mean(axis=1),2)

# get the mean total of both onnet and offnet mou in the last 3 months
data['mean_onnet_offnet_mou'] = data['mean_onnet_mou'] + data['mean_offnet_mou']


# In[42]:


# Roaming columns
# get the mean of roam ic mou in the last 3 months
cols = ['roam_ic_mou_6','roam_ic_mou_7','roam_ic_mou_8']
data['mean_roam_ic_mou'] = round(data[cols].mean(axis=1),2)

# get the mean of roam og mou in the last 3 months
cols = ['roam_og_mou_6','roam_og_mou_7','roam_og_mou_8']
data['mean_roam_og_mou'] = round(data[cols].mean(axis=1),2)

# get the mean total of both roam ic and og mou in the last 3 months
data['mean_roam_mou'] = data['mean_roam_ic_mou'] + data['mean_roam_og_mou']


# In[43]:


# loc-t2t columns
cols = ['loc_ic_t2t_mou_6','loc_ic_t2t_mou_7','loc_ic_t2t_mou_8']
data['mean_loc_ic_t2t_mou'] = round(data[cols].mean(axis=1),2)

cols = ['loc_og_t2t_mou_6','loc_og_t2t_mou_7','loc_og_t2t_mou_8']
data['mean_loc_og_t2t_mou'] = round(data[cols].mean(axis=1),2)

data['mean_loc_t2t_mou'] = data['mean_loc_ic_t2t_mou'] + data['mean_loc_og_t2t_mou']


# In[44]:


# loc-t2m columns
cols = ['loc_ic_t2m_mou_6','loc_ic_t2m_mou_7','loc_ic_t2m_mou_8']
data['mean_loc_ic_t2m_mou'] = round(data[cols].mean(axis=1),2)

cols = ['loc_og_t2m_mou_6','loc_og_t2m_mou_7','loc_og_t2m_mou_8']
data['mean_loc_og_t2m_mou'] = round(data[cols].mean(axis=1),2)

data['mean_loc_t2m_mou'] = data['mean_loc_ic_t2m_mou'] + data['mean_loc_og_t2m_mou']


# In[45]:


# loc-t2f columns
cols = ['loc_ic_t2f_mou_6','loc_ic_t2f_mou_7','loc_ic_t2f_mou_8']
data['mean_loc_ic_t2f_mou'] = round(data[cols].mean(axis=1),2)

cols = ['loc_og_t2f_mou_6','loc_og_t2f_mou_7','loc_og_t2f_mou_8']
data['mean_loc_og_t2f_mou'] = round(data[cols].mean(axis=1),2)

data['mean_loc_t2f_mou'] = data['mean_loc_ic_t2f_mou'] + data['mean_loc_og_t2f_mou']


# In[46]:


# loc-t2c columns
cols = ['loc_og_t2c_mou_6','loc_og_t2c_mou_7','loc_og_t2c_mou_8']
data['mean_loc_og_t2c_mou'] = round(data[cols].mean(axis=1),2)


# In[47]:


# std-t2t columns
cols = ['std_ic_t2t_mou_6','std_ic_t2t_mou_7','std_ic_t2t_mou_8']
data['mean_std_ic_t2t_mou'] = round(data[cols].mean(axis=1),2)

cols = ['std_og_t2t_mou_6','std_og_t2t_mou_7','std_og_t2t_mou_8']
data['mean_std_og_t2t_mou'] = round(data[cols].mean(axis=1),2)

data['mean_std_t2t_mou'] = data['mean_std_ic_t2t_mou'] + data['mean_std_og_t2t_mou']


# In[48]:


# std-t2m columns
cols = ['std_ic_t2m_mou_6','std_ic_t2m_mou_7','std_ic_t2m_mou_8']
data['mean_std_ic_t2m_mou'] = round(data[cols].mean(axis=1),2)

cols = ['std_og_t2m_mou_6','std_og_t2m_mou_7','std_og_t2m_mou_8']
data['mean_std_og_t2m_mou'] = round(data[cols].mean(axis=1),2)

data['mean_std_t2m_mou'] = data['mean_std_ic_t2m_mou'] + data['mean_std_og_t2m_mou']


# In[49]:


# std-t2f columns
cols = ['std_ic_t2f_mou_6','std_ic_t2f_mou_7','std_ic_t2f_mou_8']
data['mean_std_ic_t2f_mou'] = round(data[cols].mean(axis=1),2)

cols = ['std_og_t2f_mou_6','std_og_t2f_mou_7','std_og_t2f_mou_8']
data['mean_std_og_t2f_mou'] = round(data[cols].mean(axis=1),2)

data['mean_std_t2f_mou'] = data['mean_std_ic_t2f_mou'] + data['mean_std_og_t2f_mou']


# In[50]:


# loc columns
cols = ['loc_ic_mou_6','loc_ic_mou_7','loc_ic_mou_8']
data['mean_loc_ic_mou'] = round(data[cols].mean(axis=1),2)

cols = ['loc_og_mou_6','loc_og_mou_7','loc_og_mou_8']
data['mean_loc_og_mou'] = round(data[cols].mean(axis=1),2)

data['mean_loc_mou'] = data['mean_loc_ic_mou'] + data['mean_loc_og_mou']


# In[51]:


# std columns
cols = ['std_ic_mou_6','std_ic_mou_7','std_ic_mou_8']
data['mean_std_ic_mou'] = round(data[cols].mean(axis=1),2)

cols = ['std_og_mou_6','std_og_mou_7','std_og_mou_8']
data['mean_std_og_mou'] = round(data[cols].mean(axis=1),2)

data['mean_std_mou'] = data['mean_std_ic_mou'] + data['mean_std_og_mou']


# In[52]:


# isd columns
cols = ['isd_ic_mou_6','isd_ic_mou_7','isd_ic_mou_8']
data['mean_isd_ic_mou'] = round(data[cols].mean(axis=1),2)

cols = ['isd_og_mou_6','isd_og_mou_7','isd_og_mou_8']
data['mean_isd_og_mou'] = round(data[cols].mean(axis=1),2)

data['mean_isd_mou'] = data['mean_isd_ic_mou'] + data['mean_isd_og_mou']


# In[53]:


# spl columns
cols = ['spl_ic_mou_6','spl_ic_mou_7','spl_ic_mou_8']
data['mean_spl_ic_mou'] = round(data[cols].mean(axis=1),2)

cols = ['spl_og_mou_6','spl_og_mou_7','spl_og_mou_8']
data['mean_spl_og_mou'] = round(data[cols].mean(axis=1),2)

data['mean_spl_mou'] = data['mean_spl_ic_mou'] + data['mean_spl_og_mou']


# In[54]:


# others columns
cols = ['ic_others_6','ic_others_7','ic_others_8']
data['mean_ic_others_mou'] = round(data[cols].mean(axis=1),2)

cols = ['og_others_6','og_others_7','og_others_8']
data['mean_og_others_mou'] = round(data[cols].mean(axis=1),2)

data['mean_others_mou'] = data['mean_ic_others_mou'] + data['mean_og_others_mou']


# In[55]:


# total columns
cols = ['total_ic_mou_6','total_ic_mou_7','total_ic_mou_8']
data['mean_total_ic_mou'] = round(data[cols].mean(axis=1),2)
# Weightage for ic for the last 3 months
df = data[cols].astype(bool)
data['total_ic_weightage'] = ( df['total_ic_mou_6'] * 1 ) + ( df['total_ic_mou_7'] * 10 ) + ( df['total_ic_mou_8'] * 100 )

cols = ['total_og_mou_6','total_og_mou_7','total_og_mou_8']
data['mean_total_og_mou'] = round(data[cols].mean(axis=1),2)
# Weightage for og for the last 3 months
df = data[cols].astype(bool)
data['total_og_weightage'] = ( df['total_og_mou_6'] * 1 ) + ( df['total_og_mou_7'] * 10 ) + ( df['total_og_mou_8'] * 100 )

data['mean_total_mou'] = data['mean_total_ic_mou'] + data['mean_total_og_mou']

data['mean_total_mou_6'] = round(data[['total_ic_mou_6','total_og_mou_6']].mean(axis=1),2)
data['mean_total_mou_7'] = round(data[['total_ic_mou_7','total_og_mou_7']].mean(axis=1),2)
data['mean_total_mou_8'] = round(data[['total_ic_mou_8','total_og_mou_8']].mean(axis=1),2)


# In[56]:


# total_rech_num columns
cols = ['total_rech_num_6','total_rech_num_7','total_rech_num_8']
# mean of total recharge number
data['mean_total_rech_num'] = round(data[cols].mean(axis=1),2)
# Minimum of total recharge number
data['min_total_rech_num'] = data[cols].min(axis=1)
# Maximum of total recharge number
data['max_total_rech_num'] = data[cols].max(axis=1)


# In[57]:


# total_rech_amt columns
cols = ['total_rech_amt_6','total_rech_amt_7','total_rech_amt_8']
data['mean_total_rech_amt'] = round(data[cols].mean(axis=1),2)
data['min_total_rech_amt'] = data[cols].min(axis=1)
data['max_total_rech_amt'] = data[cols].max(axis=1)


# In[58]:


# max_rech_amt columns
cols = ['max_rech_amt_6','max_rech_amt_7','max_rech_amt_8']
data['mean_max_rech_amt'] = round(data[cols].mean(axis=1),2)

# last_day_rch_amt columns
cols = ['last_day_rch_amt_6','last_day_rch_amt_7','last_day_rch_amt_8']
data['mean_last_day_rch_amt'] = round(data[cols].mean(axis=1),2)


# In[59]:


# total_rech_data columns
cols = ['total_rech_data_6','total_rech_data_7','total_rech_data_8']
data['mean_total_rech_data'] = round(data[cols].mean(axis=1),2)
data['min_total_rech_data'] = data[cols].min(axis=1)
data['max_total_rech_data'] = data[cols].max(axis=1)


# In[60]:


# total_rech_amt_data columns
cols = ['total_rech_amt_data_6','total_rech_amt_data_7','total_rech_amt_data_8']
data['mean_total_rech_amt_data'] = round(data[cols].mean(axis=1),2)
data['min_total_rech_amt_data'] = data[cols].min(axis=1)
data['max_total_rech_amt_data'] = data[cols].max(axis=1)


# In[61]:


# total_rech_voice columns
data['mean_total_rech_voice'] = data['mean_total_rech_num'] - data['mean_total_rech_data']
data['min_total_rech_voice'] = data['min_total_rech_num'] - data['min_total_rech_data']
data['max_total_rech_voice'] = data['max_total_rech_num'] - data['max_total_rech_data']


# In[62]:


# total_rech_amt_voice columns
data['mean_total_rech_amt_voice'] = data['mean_total_rech_amt'] - data['mean_total_rech_amt_data']
data['min_total_rech_amt_voice'] = data['min_total_rech_amt'] - data['min_total_rech_amt_data']
data['max_total_rech_amt_voice'] = data['max_total_rech_amt'] - data['max_total_rech_amt_data']


# In[63]:


# max_rech_data columns
cols = ['max_rech_data_6','max_rech_data_7','max_rech_data_8']
data['mean_max_rech_data'] = round(data[cols].mean(axis=1),2)

# count_rech_2g columns
cols = ['count_rech_2g_6','count_rech_2g_7','count_rech_2g_8']
data['mean_count_rech_2g'] = round(data[cols].mean(axis=1),2)

# count_rech_3g columns
cols = ['count_rech_3g_6','count_rech_3g_7','count_rech_3g_8']
data['mean_count_rech_3g'] = round(data[cols].mean(axis=1),2)


# In[64]:


#get recharge num weightage for the last three months
cols = ['total_rech_num_6','total_rech_num_7','total_rech_num_8']
df = data[cols].astype(bool)
data['rech_num_weightage'] = ( df['total_rech_num_6'] * 1 ) + ( df['total_rech_num_7'] * 10 ) + ( df['total_rech_num_8'] * 100 )


# In[65]:


#get recharge amount weightage for the last three months
cols = ['total_rech_amt_6','total_rech_amt_7','total_rech_amt_8']
df = data[cols].astype(bool)
data['rech_amt_weightage'] = ( df['total_rech_amt_6'] * 1 ) + ( df['total_rech_amt_7'] * 10 ) + ( df['total_rech_amt_8'] * 100 )


# In[66]:


# arpu columns
# ARPU = Total Revenue / Average Subscribers
cols = ['arpu_6','arpu_7','arpu_8']
data['mean_arpu'] = round(data[cols].mean(axis=1),2)

cols = ['arpu_2g_6','arpu_2g_7','arpu_2g_8']
data['mean_arpu_2g_data'] = round(data[cols].mean(axis=1),2)

cols = ['arpu_3g_6','arpu_3g_7','arpu_3g_8']
data['mean_arpu_3g_data'] = round(data[cols].mean(axis=1),2)

cols = ['vol_2g_mb_6','vol_2g_mb_7','vol_2g_mb_8']
data['mean_vol_2g_mb_data'] = round(data[cols].mean(axis=1),2)

cols = ['vol_3g_mb_6','vol_3g_mb_7','vol_3g_mb_8']
data['mean_vol_3g_mb_data'] = round(data[cols].mean(axis=1),2)


# In[67]:


#get night_pck_user weightage for the last three months
cols = ['night_pck_user_6','night_pck_user_7','night_pck_user_8']
data['night_pck_weightage'] = ( data['night_pck_user_6'] * 1 ) + ( data['night_pck_user_7'] * 10 ) + ( data['night_pck_user_8'] * 100 )


# In[68]:


#get fb_user weightage for the last three months
cols = ['fb_user_6','fb_user_7','fb_user_8']
data['fb_user_weightage'] = ( data['fb_user_6'] * 1 ) + ( data['fb_user_7'] * 10 ) + ( data['fb_user_8'] * 100 )


# In[69]:


#get vbc mean for the last three months
cols = ['jun_vbc_3g','jul_vbc_3g','aug_vbc_3g']
data['mean_vbc_3g'] = round(data[cols].mean(axis=1),2)


# In[70]:


#get monthly pack weightage for the last three months
data['monthly_2g_weightage'] = ( data['monthly_2g_6'] * 1 ) + ( data['monthly_2g_7'] * 10 ) + ( data['monthly_2g_8'] * 100 )
data['monthly_3g_weightage'] = ( data['monthly_3g_6'] * 1 ) + ( data['monthly_3g_7'] * 10 ) + ( data['monthly_3g_8'] * 100 )

#get sachet pack weightage for the last three months
data['sachet_2g_weightage'] = ( data['sachet_2g_6'] * 1 ) + ( data['sachet_2g_7'] * 10 ) + ( data['sachet_2g_8'] * 100 )
data['sachet_3g_weightage'] = ( data['sachet_3g_6'] * 1 ) + ( data['sachet_3g_7'] * 10 ) + ( data['sachet_3g_8'] * 100 )


# In[71]:


#Taking a backup of dataset for later use
master_df = data.copy()


# In[72]:


# prepare the dataset
churn = data['churn']
data = data.drop('churn', axis=1)

#split the columns into category and numerical
cat_cols = ['night_pck_user_6','monthly_2g_6','sachet_2g_6','monthly_3g_6','sachet_3g_6','fb_user_6',
            'night_pck_user_7','monthly_2g_7','sachet_2g_7','monthly_3g_7','sachet_3g_7','fb_user_7',
            'night_pck_user_8','monthly_2g_8','sachet_2g_8','monthly_3g_8','sachet_3g_8','fb_user_8'] 

num_cols = list(set(data.columns).difference(set(cat_cols)))


# In[73]:


# dummy encode the categorical columns
data = pd.concat([data,pd.get_dummies(data[cat_cols], drop_first=True)], axis=1)

# drop the original columns
data.drop(cat_cols, axis=1, inplace=True)


# In[74]:


# log transform with constant 10000 for real numbers
data[num_cols] = np.log((10000 + data[num_cols]))
data = np.log((10000 + data))


# In[75]:


#Standardize the numeric values
data[num_cols] = (( data[num_cols] - data[num_cols].mean() ) / data[num_cols].std())


# In[76]:


# Check for missing values count
#data = data.replace([np.inf, -np.inf], np.nan)
df = data.isnull().sum().reset_index(name='missing_cnt')
df.loc[df['missing_cnt']>0].sort_values('missing_cnt', ascending=False)


# In[77]:


#pip install scikit-learn


# In[78]:


from imblearn.over_sampling import ADASYN


# In[79]:


from collections import Counter
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
import pandas as pd, numpy, string
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
#Remove Special Charactors
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


# In[80]:


from imblearn.pipeline import Pipeline


# In[81]:


X = data
Y = churn

smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=1.0)
X, Y = smote.fit_resample(X, Y)

#adasyn = ADASYN(random_state=RANDOM_STATE)
#X, Y = adasyn.fit_sample(X, Y)

#print('Class Balance count : ',Counter(Y))
#counter =  Counter({1: 27629, 0: 27390})
#Counter() = {1: 27629, 0: 27390}
#print(counter)


# In[82]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=RANDOM_STATE)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[83]:


#initialize the pca with randomized
pca = PCA(svd_solver='randomized', random_state=RANDOM_STATE)
# fit the training dataset
pca.fit(X_train)


# In[84]:


#Screeplot for the PCA components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[85]:


# Initialize pca with 50 components
pca = PCA(n_components=50, random_state=RANDOM_STATE)
# fit and transform the training dataset
X_train_pca = pca.fit_transform(X_train)
# transform the testing dataset
X_test_pca = pca.transform(X_test)
# fit and transform the whole dataset
X_pca = pca.fit_transform(X)


# In[86]:


# List to store the model scores
model_score_list = []


# In[87]:


'''
Method Name : evaluate_model
Input(s)    : 2 series
Output(s)   : 4 float values  
Description :
- This method gets 2 series (actual and predicted) as inputs and compute the accuracy, ROC, Precision and Recall scores  
- Then returns the computed scores. 
'''
def evaluate_model(actual, pred):
    # Accuracy Score
    acc_score = round(accuracy_score(actual, pred)*100,2)
    print('Accuracy Score : ',acc_score)
    # ROC AUC score
    roc_score = round(roc_auc_score(actual, pred)*100,2)
    print('ROC AUC score : ',roc_score)
    # Precision score
    prec_score = round(precision_score(actual, pred)*100,2)
    print('Precision score : ', prec_score)
    # Recall score
    rec_score = round(recall_score(actual, pred)*100,2)
    print('Recall score : ', rec_score)

    return acc_score, roc_score, prec_score, rec_score


# In[88]:


model = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)
# fit the pca training data
model.fit(X_train_pca, Y_train)
# predict the testing pca data
Y_pred = model.predict(X_test_pca)

# Model evaluation
acc_score, roc_score, prec_score, rec_score = evaluate_model(Y_test, Y_pred)
# add the model scores to score list 
model_score_list.append({'model_name':'LogisticRegression', 'acc_score':acc_score, 'roc_score':roc_score, 'precision_score':prec_score, 'recall_score':rec_score})


# In[89]:


# initialize the KNeighbors classifiers
model = KNeighborsClassifier()
# fit the pca training data
model.fit(X_train_pca, Y_train)
# predict the pca testing data
Y_pred = model.predict(X_test_pca)

# Model evaluation
acc_score, roc_score, prec_score, rec_score = evaluate_model(Y_test, Y_pred)
# add the model scores to score list
model_score_list.append({'model_name':'KNeighborsClassifier', 'acc_score':acc_score, 'roc_score':roc_score, 'precision_score':prec_score, 'recall_score':rec_score})


# In[90]:


# initialize the Ridge Classifier
model = RidgeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
# fit the pca training data
model.fit(X_train_pca, Y_train)
# predict the pca testing data
Y_pred = model.predict(X_test_pca)

# Model evaluation
acc_score, roc_score, prec_score, rec_score = evaluate_model(Y_test, Y_pred)
# add the model scores to score list
model_score_list.append({'model_name':'RidgeClassifier', 'acc_score':acc_score, 'roc_score':roc_score, 'precision_score':prec_score, 'recall_score':rec_score})


# In[91]:


# initialize the Decision Tree
model = DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
# fit the pca training data
model.fit(X_train_pca, Y_train)
# predict the pca testing data
Y_pred = model.predict(X_test_pca)

# Model evaluation
acc_score, roc_score, prec_score, rec_score = evaluate_model(Y_test, Y_pred)
# add the model scores to score list
model_score_list.append({'model_name':'DecisionTreeClassifier', 'acc_score':acc_score, 'roc_score':roc_score, 'precision_score':prec_score, 'recall_score':rec_score})


# In[ ]:




