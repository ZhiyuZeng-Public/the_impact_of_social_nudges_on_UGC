
# coding: utf-8

# # Import packages 

# In[1]:


# Some basic Python packages need to be imported
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.optimize import least_squares
from matching import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('precision', 6)


# # Load data

# Since the authors have a Non-Disclosure Agreement with the platform where the field experiments were conducted, the data files below can not be revealed. The distributional information of the data is disclosed in Online Appendix G, and can also be found in the "data and code" folder in the supplement materials.

# In[2]:


first_author_feature = pd.read_csv('social_nudge_first_exp_author_feature.csv')
first_production = pd.read_csv('social_nudge_first_exp_production.csv')
first_nudge_feature = pd.read_csv('social_nudge_first_exp_nudge_feature.csv')
first_author_historic_consumption = pd.read_csv('social_nudge_first_exp_author_historic_consumption.csv')
first_logging_consumption = pd.read_csv('social_nudge_first_exp_logging_consumption.csv')
first_diffusion = pd.read_csv('social_nudge_first_exp_diffusion.csv')


# In[3]:


print(first_author_feature.columns)
# Variables in the dataset first_author_feature: 
# 'author_id': A provider's id 
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0) 
# 'author_gender': Gender of a provider. F if a provider is female and M if a provider is male 
# 'author_follow_cnt': A provider's number of following on the day prior to the experiment 
# 'author_fans_cnt': A provider's number of followers on the day prior to the experiment 
# 'upload_days_a_week_prior': A provider's number of days when she uploaded any video during the week prior to the experiment
# 'upload_photo_cnt_a_week_prior': A provider's number of videos they uploaded during the week prior to the experiment 
# 'logging_msg_cnt': A provider's number of social nudges received on the first reception day


# In[4]:


first_author_feature[['treatment', 'author_follow_cnt', 'author_gender',
       'author_fans_cnt', 'upload_days_a_week_prior',
       'upload_photo_cnt_a_week_prior', 'logging_msg_cnt']].describe()


# In[5]:


print(first_production.columns)
# Variables in the dataset first_production:
# 'author_id': A provider's if
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0)
# 'is_upload_logging_day': Whether a provider uploaded any videos on Day 1 (the first reception day)
# 'photo_id_cnt_fillzero_logging_day': The number of videos uploaded on Day 1 (the first reception day)
# 'is_upload_next_day_1': Whether a provider uploaded any videos on Day 2
# 'photo_id_cnt_fillzero_next_day_1': The number of videos uploaded on Day 2
# 'is_upload_next_day_2': Whether a provider uploaded any videos on Day 3
# 'photo_id_cnt_fillzero_next_day_2': The number of videos uploaded on Day 3 
# 'is_upload_next_day_3': Whether a provider uploaded any videos on Day 4
# 'photo_id_cnt_fillzero_next_day_3': The number of videos uploaded on Day 4


# In[6]:


first_production.describe()


# In[7]:


print(first_nudge_feature.columns)
# Variables in the dataset first_nudge_feature:
# 'author_id': A provider' id
# 'treatment': Whether a provider was in the treatment condition, versus the control condition
# 'is_bi_follow': Whether the nudge sender was also following the provider


# In[8]:


first_nudge_feature['is_bi_follow'].describe()


# In[9]:


print(first_author_historic_consumption.columns)
# Variables in the dataset first_author_historic_consumption:
# 'author_id': A provider's id
# 'treatment': Whether a provider was in the treatment condition, versus the control condition
# 'his_like_per_play_2018': The total number of likes provider i received from January 1, 2018, 
# to the day prior to the experiment, divided by the total number of views provider i received during that same period


# In[10]:


first_author_historic_consumption['his_like_per_play_2018'].describe()


# In[11]:


print(first_logging_consumption.columns)
# Variables in the dataset first_logging_consumption:
# 'author_id': A provider's id
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0)
# 'sum_play_cnt': The total number of views a provider engendered that could be attributed to videos that provider uploaded on the first reception day 
# 'avg_complete_per_play_ratio': For a provider's videos uploaded on the reception day, the average percentage of times viewers watched a video until the end 
# 'avg_comment_per_play_ratio': For a provider's videos uploaded on the reception day, the average percentage of viewers who commented on a video in the comments section beneath it 
# 'avg_like_per_play_ratio': For a provider's videos uploaded on the reception day, the average percentage of viewers who gave likes to a video
# 'avg_follow_per_play_ratio': For a provider's videos uploaded on the reception day, the average percentage of viewers who chose to follow provider i while watching a video


# In[12]:


first_logging_consumption[['sum_play_cnt', 'avg_complete_per_play_ratio',
       'avg_comment_per_play_ratio', 'avg_like_per_play_ratio',
       'avg_follow_per_play_ratio']].describe()


# In[13]:


print(first_diffusion.columns)
# Variables in the dataset first_diffusion:
# 'author_id': user id of a provider
# 'treatment': whether a provider was in the treatment condition, versus the control condition
# 'nudges_to_others_logging_day': the number of social nudges sent by each provider i to other providers on Day 1 (the first reception day)
# 'nudges_to_others_next_day_1': the number of social nudges sent by each provider i to other providers on Day 2
# 'nudges_to_others_next_day_2': the number of social nudges sent by each provider i to other providers on Day 3


# In[14]:


first_diffusion[['nudges_to_others_logging_day',
       'nudges_to_others_next_day_1', 'nudges_to_others_next_day_2']].describe()


# ## Create new variables

# In[15]:


print(first_author_feature['author_gender'].value_counts())


# In[16]:


# convert gender to binary variable
first_author_feature['author_gender_binary'] = first_author_feature['author_gender'].replace({'F':1, 'M':0, 'UNKNOWN':None})


# In[17]:


first_diffusion = first_diffusion.merge(first_author_feature[['author_id', 'author_follow_cnt']], how = 'left')
for col in['logging_day', 
       'next_day_1',  'next_day_2']:
    col_per_link = 'nudges_to_others_' + col + '_per_link' 
    first_diffusion[col_per_link] = first_diffusion['nudges_to_others_' + col]/first_diffusion['author_follow_cnt']


# In[18]:


# winsorize variable sum_play_cnt
col = 'sum_play_cnt'
win_col = 'wins_'+col
first_logging_consumption[win_col] = first_logging_consumption[col]
cut_point = first_logging_consumption.loc[first_logging_consumption[win_col]>0, win_col].quantile(0.95)
print(win_col+'_cut', '=', cut_point)
first_logging_consumption.loc[first_logging_consumption[win_col] > cut_point, win_col] = cut_point


# In[19]:


# winsorize variable nudges_to_others
for col in ['nudges_to_others_logging_day', 
       'nudges_to_others_next_day_1',  'nudges_to_others_next_day_2']:
    win_col = 'wins_'+col
    first_diffusion[win_col] = first_diffusion[col]
    cut_point = first_diffusion.loc[first_diffusion[win_col]>0, win_col].quantile(0.95)
    print(win_col+'_cut', '=', cut_point)
    first_diffusion.loc[first_diffusion[win_col] > cut_point, win_col] = cut_point


# In[20]:


# standardize variable
for col in ['author_follow_cnt','author_fans_cnt','upload_days_a_week_prior','upload_photo_cnt_a_week_prior']:
    stand_col = 'stand_'+col
    first_author_feature[stand_col] = first_author_feature[col]/first_author_feature[col].std()


# In[21]:


# standardize variable
for col in ['photo_id_cnt_fillzero_logging_day', 'photo_id_cnt_fillzero_next_day_1', 
           'photo_id_cnt_fillzero_next_day_2', 'photo_id_cnt_fillzero_next_day_3']:
    stand_col = 'stand_'+col
    first_production[stand_col] = first_production[col]/first_production[col].std()


# In[22]:


# standardize variable
for col in ['his_like_per_play_2018']:
    stand_col = 'stand_'+col
    first_author_historic_consumption[stand_col] = first_author_historic_consumption[col]/first_author_historic_consumption[col].std()


# In[23]:


# standardize variable
for col in ['wins_sum_play_cnt', 'avg_complete_per_play_ratio',
       'avg_comment_per_play_ratio', 'avg_like_per_play_ratio',
       'avg_follow_per_play_ratio']:
    stand_col = 'stand_'+col
    first_logging_consumption[stand_col] = first_logging_consumption[col]/first_logging_consumption[col].std()


# In[24]:


# standardize variable
for col in ['wins_nudges_to_others_logging_day', 'wins_nudges_to_others_next_day_1',
       'wins_nudges_to_others_next_day_2']:
    stand_col = 'stand_'+col
    first_diffusion[stand_col] = first_diffusion[col]/first_diffusion[col].std()


# # Section 3.2 Data and Randomization Check

# ### Table 1 Randomization Check

# In[25]:


# Two-sample proportion test for Proportion of Females 
sub_data = first_author_feature
treatment_female = sub_data[(sub_data['treatment'] == 1) & (sub_data['author_gender'] == 'F')].shape[0]
treatment_male = sub_data[(sub_data['treatment'] == 1) & (sub_data['author_gender'] == 'M')].shape[0]
control_female = sub_data[(sub_data['treatment'] == 0) & (sub_data['author_gender'] == 'F')].shape[0]
control_male = sub_data[(sub_data['treatment'] == 0) & (sub_data['author_gender'] == 'M')].shape[0]
print('# of treatment_female =', treatment_female, ';', '# of treatment_male =', treatment_male, ';', 'sum of two parts =', treatment_female+treatment_male, ';', 'female proportion =', '{:.2%}'.format(treatment_female/(treatment_female+treatment_male)))
print('# of control_female =', control_female, ';', '# of control_male =', control_male, ';', 'sum of two parts =', control_female+control_male, ';', 'female proportion =',  '{:.2%}'.format(control_female/(control_female+control_male)))
print('# of null gender =', len(first_author_feature)-(treatment_female+treatment_male+control_female+control_male),       ';','proportion of null gender =', '{:.2%}'.format((len(first_author_feature)-(treatment_female+treatment_male+control_female+control_male))/len(first_author_feature)))

female_proportion = np.array([[treatment_female, control_female], [treatment_female+treatment_male, control_female+control_male]])

chi2, p, dof, ex = chi2_contingency(female_proportion)

print('chi2:', chi2, ';', 'p:', p)


# In[26]:


# T-tests for number of users who were following them (``number of followers") on the day prior to the experiment, 
# and number of users they were following (``number of following") on the day prior to the experiment, 
# as well as the number of videos they uploaded 
# and the number of days when they uploaded any video during the week prior to the experiment

outcomes = ['stand_author_fans_cnt', 'stand_author_follow_cnt','stand_upload_photo_cnt_a_week_prior','stand_upload_days_a_week_prior']


print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'pvalue', 'exp_std', 'base_std', 'exp-base', '(exp-base)/base')

sub_data = first_author_feature

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1)][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0)][outcome]

    t_statistics, pvalue = ttest_ind(d1, d0)

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)

    print(outcome, obs1, obs0, '&', '%.4f'%mean1, '&', '%.4f'%mean0, '&', '%.2f'%pvalue, '%.4f'%std1, '%.4f'%std0, '%.4f'%(mean1-mean0), '%.4f'%((mean1-mean0)/mean0))


# # Section 4 Direct Effects of Social Nudges on Content Production

# In[27]:


first_production['photo_id_cnt_fillzero_logging_day_conditional_uploding'] = first_production['photo_id_cnt_fillzero_logging_day']
first_production.loc[first_production['is_upload_logging_day']==0, 'photo_id_cnt_fillzero_logging_day_conditional_uploding'] = None

first_production['stand_photo_id_cnt_fillzero_logging_day_conditional_uploding'] = first_production['stand_photo_id_cnt_fillzero_logging_day']
first_production.loc[first_production['is_upload_logging_day']==0, 'stand_photo_id_cnt_fillzero_logging_day_conditional_uploding'] = None

first_production = first_production.merge(first_nudge_feature, how = 'left')


# ### Table 2 Direct Effects of Social Nudges on Content Production on the First Reception Day

# In[28]:


# Table 2; using raw data

df = first_production

mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('is_upload_logging_day ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day_conditional_uploding ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment*is_bi_follow', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/df[df['treatment']==0]['photo_id_cnt_fillzero_logging_day'].mean() ) )
print('relative effect size:', '{:.2%}'.format((mod4.params['treatment']+mod4.params['treatment:is_bi_follow'])/df[df['treatment']==0]['photo_id_cnt_fillzero_logging_day'].mean() ) )


# In[29]:


# Table 2; using standardized data
mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('is_upload_logging_day ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day_conditional_uploding ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment*is_bi_follow', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/df[df['treatment']==0]['stand_photo_id_cnt_fillzero_logging_day'].mean() ) )
print('relative effect size:', '{:.2%}'.format((mod4.params['treatment']+mod4.params['treatment:is_bi_follow'])/df[df['treatment']==0]['stand_photo_id_cnt_fillzero_logging_day'].mean() ) )


# ### Table 3 Effects of Social Nudges on Video Consumption and Quality

# In[30]:


first_logging_consumption = first_logging_consumption.merge(first_author_historic_consumption,                                                                           how='left')
first_logging_consumption = first_logging_consumption.merge(first_production[['author_id','is_upload_logging_day']], how='left')


# In[31]:


# Table 3; using raw data

df = first_logging_consumption

mod1 = sm.OLS.from_formula('wins_sum_play_cnt ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('avg_complete_per_play_ratio ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('avg_like_per_play_ratio ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('avg_comment_per_play_ratio ~ 1 + treatment', data=df).fit()
mod5 = sm.OLS.from_formula('avg_follow_per_play_ratio ~ 1 + treatment', data=df).fit()
mod6 = sm.OLS.from_formula('his_like_per_play_2018 ~ 1 + treatment', data=df[df['is_upload_logging_day']==1]).fit()
mod7 = sm.OLS.from_formula('avg_like_per_play_ratio ~ 1 + treatment + his_like_per_play_2018', data=df[df['is_upload_logging_day']==1]).fit()

print('\n##########')
print('panel A column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('panel A column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('panel A column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel A column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )

print('\n##########')
print('panel A column (5):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod5.params['Intercept']) )

print('\n##########')
print('panel B column (1):')
print(mod6.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod6.params['treatment']/mod6.params['Intercept']) )

print('\n##########')
print('panel B column (2):')
print(mod7.get_robustcov_results().summary2(float_format="%.5f"))


# In[32]:


# Table 3; using standardized data

df = first_logging_consumption

mod1 = sm.OLS.from_formula('stand_wins_sum_play_cnt ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_avg_complete_per_play_ratio ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_avg_like_per_play_ratio ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_avg_comment_per_play_ratio ~ 1 + treatment', data=df).fit()
mod5 = sm.OLS.from_formula('stand_avg_follow_per_play_ratio ~ 1 + treatment', data=df).fit()
mod6 = sm.OLS.from_formula('stand_his_like_per_play_2018 ~ 1 + treatment', data=df[df['is_upload_logging_day']==1]).fit()
mod7 = sm.OLS.from_formula('stand_avg_like_per_play_ratio ~ 1 + treatment + stand_his_like_per_play_2018', data=df[df['is_upload_logging_day']==1]).fit()


print('\n##########')
print('panel A column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('panel A column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('panel A column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel A column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )

print('\n##########')
print('panel A column (5):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod5.params['Intercept']) )

print('\n##########')
print('panel B column (1):')
print(mod6.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod6.params['treatment']/mod6.params['Intercept']) )

print('\n##########')
print('panel B column (2):')
print(mod7.get_robustcov_results().summary2(float_format="%.5f"))


# Footnote 11

# In[33]:


# robustness check of winsorizing sum_play_cnt at the 99th percentile of nonzero values 
col = 'sum_play_cnt'
win_col = 'wins_'+col+'_99th'
first_logging_consumption[win_col] = first_logging_consumption[col]
cut_point = first_logging_consumption.loc[first_logging_consumption[win_col]>0, win_col].quantile(0.99)
print(win_col+'_cut', '=', cut_point)
first_logging_consumption.loc[first_logging_consumption[win_col] > cut_point, win_col] = cut_point

first_logging_consumption['stand_wins_sum_play_cnt_99th'] = first_logging_consumption['wins_sum_play_cnt_99th']/first_logging_consumption['wins_sum_play_cnt_99th'].std()

mod1 = sm.OLS.from_formula('stand_wins_sum_play_cnt_99th ~ 1 + treatment', data=first_logging_consumption).fit()
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )


# ### Table 4 Over-Time Direct Effects of Social Nudges on Content Production

# In[34]:


# Table 4; using raw data

df = first_production

mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )


# In[35]:


# Table 4; using standardized data

df = first_production

mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )


# # Section 5 Indirect Effects of Social Nudges on Production via Nudge Diffusion

# In[36]:


first_diffusion = first_diffusion.merge(first_nudge_feature, how = 'left')


# ### Table 5 Effect of Social Nudges on Nudge Diffusion on the First Reception Day

# In[37]:


# Table 5; using raw data

df = first_diffusion

mod1 = sm.OLS.from_formula('wins_nudges_to_others_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('wins_nudges_to_others_logging_day ~ 1 + treatment*is_bi_follow', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/df[df['treatment']==0]['wins_nudges_to_others_logging_day'].mean() ) )
print('relative effect size:', '{:.2%}'.format((mod2.params['treatment']+mod2.params['treatment:is_bi_follow'])/df[df['treatment']==0]['wins_nudges_to_others_logging_day'].mean() ) )


# In[38]:


# Table 5; using standardized data

df = first_diffusion

mod1 = sm.OLS.from_formula('stand_wins_nudges_to_others_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_wins_nudges_to_others_logging_day ~ 1 + treatment*is_bi_follow', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/df[df['treatment']==0]['stand_wins_nudges_to_others_logging_day'].mean() ) )
print('relative effect size:', '{:.2%}'.format((mod2.params['treatment']+mod2.params['treatment:is_bi_follow'])/df[df['treatment']==0]['stand_wins_nudges_to_others_logging_day'].mean() ) )


# ### Table 6 Effects of Social Nudges on Nudge Diffusion Over Time

# In[39]:


# Table 6; using raw data

df = first_diffusion

mod1 = sm.OLS.from_formula('wins_nudges_to_others_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('wins_nudges_to_others_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('wins_nudges_to_others_next_day_2 ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )


# In[40]:


# Table 6; using standardized data

df = first_diffusion

mod1 = sm.OLS.from_formula('stand_wins_nudges_to_others_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_wins_nudges_to_others_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_wins_nudges_to_others_next_day_2 ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )


# # Appendix A Robustness Checks of the Main Results From the First Social-Nudge Experiment

# ### Table 10 Effects of Social Nudges Among All Providers Who Were Sent at Least One Social Nudge in the First Social-Nudge Experiment

# In[41]:


first_production_no_prior_nudge_limit = pd.read_csv('social_nudge_first_exp_production_no_prior_nudge_limit.csv')


# In[42]:


print(first_production_no_prior_nudge_limit.columns)
# Variables in the dataset first_production_no_prior_nudge_limit:
# 'author_id': A provider's id 
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0)
# 'is_upload': Whether a provider uploaded any videos on the first reception day
# 'photo_id_cnt_fillzero': A provider’s number of videos uploaded on the first reception day


# In[43]:


first_diffusion_no_prior_nudge_limit = pd.read_csv('social_nudge_first_exp_diffusion_no_prior_nudge_limit.csv')


# In[44]:


print(first_diffusion_no_prior_nudge_limit.columns)
# Variables in the dataset first_diffusion_no_prior_nudge_limit: 
# 'author_id': A provider's id 
# 'nudges_to_others_logging_day':  A provider’s number of social nudges sent by a provider to other providers on the first reception day


# In[45]:


# winsorize variable
for col in ['nudges_to_others_logging_day']:
    win_col = 'wins_'+col
    first_diffusion_no_prior_nudge_limit[win_col] = first_diffusion_no_prior_nudge_limit[col]
    cut_point = first_diffusion_no_prior_nudge_limit.loc[first_diffusion_no_prior_nudge_limit[win_col]>0, win_col].quantile(0.95)
    print(win_col+'_cut', '=', cut_point)
    first_diffusion_no_prior_nudge_limit.loc[first_diffusion_no_prior_nudge_limit[win_col] > cut_point, win_col] = cut_point


# In[46]:


# standarize variable
first_production_no_prior_nudge_limit['stand_photo_id_cnt_fillzero'] = first_production_no_prior_nudge_limit['photo_id_cnt_fillzero']/first_production_no_prior_nudge_limit['photo_id_cnt_fillzero'].std()
first_diffusion_no_prior_nudge_limit['stand_wins_nudges_to_others_logging_day'] = first_diffusion_no_prior_nudge_limit['wins_nudges_to_others_logging_day']/first_diffusion_no_prior_nudge_limit['wins_nudges_to_others_logging_day'].std()


# In[47]:


first_no_prior_nudge_limit = first_diffusion_no_prior_nudge_limit.merge(first_production_no_prior_nudge_limit, how = 'left')


# In[48]:


# Table 10; using raw data
df = first_no_prior_nudge_limit

mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero  ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('wins_nudges_to_others_logging_day ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )


# In[49]:


# Table 10; using standardized data

mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero  ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_wins_nudges_to_others_logging_day ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )


# ### Table 11 Effects of Social Nudges on Content Production Within 24 Hours Following the First Nudge

# In[50]:


first_production_in_24h = pd.read_csv('social_nudge_first_exp_production_in_24h.csv')


# In[51]:


print(first_production_in_24h.columns)
# Variables in the dataset first_production_in_24h
# 'author_id': A provider's id 
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0)
# 'is_upload': Whether a provider uploaded any videos during 24 hours since the first social nudge
# 'photo_id_cnt_fillzero': A provider's number of videos uploaded during 24 hours since the first social nudge


# In[52]:


# standardize data
first_production_in_24h['stand_photo_id_cnt_fillzero'] = first_production_in_24h['photo_id_cnt_fillzero']/first_production_in_24h['photo_id_cnt_fillzero'].std()


# In[53]:


# Table 11; using raw data
df = first_production_in_24h

mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero  ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('is_upload ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )


# In[54]:


# Table 11; using standardized data
df = first_production_in_24h

mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero  ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('is_upload ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )


# # Appendix B The Second Social-Nudge Experiment as a Replication

# In[55]:


second_exp = pd.read_csv('social_nudge_second_exp_production_diffusion.csv')


# In[56]:


print(second_exp.columns)
# Variables in the dataset second_exp:
# 'author_id': A provider's id  
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0) 
# 'is_upload_logging_day': Whether a provider uploaded any videos on the first reception day (or Day 1)
# 'photo_id_cnt_fillzero_logging_day': A provider's number of videos uploaded on the first reception day (or Day 1)
# 'is_upload_next_day_1': Whether a provider uploaded any videos on the first day after the first reception day (or Day 2)
# 'photo_id_cnt_fillzero_next_day_1': A provider's number of videos uploaded on the first day after the first reception day (or Day 2) 
# 'is_upload_next_day_2': Whether a provider uploaded any videos on Day 3
# 'photo_id_cnt_fillzero_next_day_2': A provider's number of videos uploaded on Day 3
# 'is_upload_next_day_3': Whether a provider uploaded any videos on Day 4
# 'photo_id_cnt_fillzero_next_day_3': A provider's number of videos uploaded on Day 4
# 'is_upload_next_day_4': Whether a provider uploaded any videos on Day 5
# 'photo_id_cnt_fillzero_next_day_4': A provider's number of videos uploaded on Day 5
# 'logging_msg_cnt': A provider’s number of social nudges received on the first reception day 
# 'nudges_to_others_logging_day': A provider’s number of social nudges sent by a provider to other providers on the first reception day (or Day 1)  
# 'nudges_to_others_next_day_1':  A provider’s number of social nudges sent by a provider to other providers on Day 2
# 'nudges_to_others_next_day_2': A provider’s number of social nudges sent by a provider to other providers on Day 3
# 'nudges_to_others_next_day_3': A provider’s number of social nudges sent by a provider to other providers on Day 4
# 'nudges_to_others_next_day_4': A provider’s number of social nudges sent by a provider to other providers on Day 5
# 'author_follow_cnt':


# In[57]:


first_diffusion = first_diffusion.merge(first_author_feature[['author_id', 'author_follow_cnt']], how = 'left')
for col in['logging_day', 
       'next_day_1',  'next_day_2']:
    col_per_link = 'nudges_to_others_' + col + '_per_link' 
    second_exp[col_per_link] = second_exp['nudges_to_others_' + col]/second_exp['author_follow_cnt']


# In[58]:


# winsorize variable
for col in ['nudges_to_others_logging_day', 
       'nudges_to_others_next_day_1',  'nudges_to_others_next_day_2',
        'nudges_to_others_next_day_3',  'nudges_to_others_next_day_4']:
    win_col = 'wins_'+col
    second_exp[win_col] = second_exp[col]
    cut_point = second_exp.loc[second_exp[win_col]>0, win_col].quantile(0.95)
    print(win_col+'_cut', '=', cut_point)
    second_exp.loc[second_exp[win_col] > cut_point, win_col] = cut_point


# In[59]:


# standardize variable
for col in ['wins_nudges_to_others_logging_day', 'wins_nudges_to_others_next_day_1',
       'wins_nudges_to_others_next_day_2', 'wins_nudges_to_others_next_day_3',
       'wins_nudges_to_others_next_day_4',
           'photo_id_cnt_fillzero_logging_day', 'photo_id_cnt_fillzero_next_day_1', 
           'photo_id_cnt_fillzero_next_day_2', 'photo_id_cnt_fillzero_next_day_3',
           'photo_id_cnt_fillzero_next_day_4']:
    stand_col = 'stand_'+col
    second_exp[stand_col] = second_exp[col]/second_exp[col].std()


# ### Table 12 Over-Time Direct Effects of Social Nudges on Content Production (Replicated)

# In[60]:


# Table 12; using raw data

df = second_exp

mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )


# In[61]:


# Table 12; using standardized data

df = second_exp

mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )


# ### Table 13 Over-Time Effects of Social Nudges on Nudge Diffusion (Replicated)

# In[62]:


# Table 13; using raw data

df = second_exp

mod1 = sm.OLS.from_formula('wins_nudges_to_others_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('wins_nudges_to_others_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('wins_nudges_to_others_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('wins_nudges_to_others_next_day_3 ~ 1 + treatment', data=df).fit()
mod5 = sm.OLS.from_formula('wins_nudges_to_others_next_day_4 ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )

print('\n##########')
print('column (5):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod5.params['Intercept']) )


# In[63]:


# Table 13; using standardized data

df = second_exp

mod1 = sm.OLS.from_formula('stand_wins_nudges_to_others_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_wins_nudges_to_others_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_wins_nudges_to_others_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_wins_nudges_to_others_next_day_3 ~ 1 + treatment', data=df).fit()
mod5 = sm.OLS.from_formula('stand_wins_nudges_to_others_next_day_4 ~ 1 + treatment', data=df).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )

print('\n##########')
print('column (5):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod5.params['Intercept']) )


# # Appendix C Additional Analyses about the Direct Effects of Social Nudges

# In[64]:


first_private_message = pd.read_csv('social_nudge_first_exp_private_message.csv')


# In[65]:


print(first_private_message.columns)
# Variables in the dataset:
# 'author_id': A provider's id  
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0) 
# 'whether_follower_send_author_msg_exp_time': Whether the follower who sent a provider the first social nudge 
#    in the experiment (i.e.,the first social-nudge sender) also sent any private messages to that provider between 
#    the start date of the experiment and that provider's first reception day (including both ends)


# In[66]:


first_control_group_did = pd.read_csv('social_nudge_first_exp_control_group_did.csv')


# In[67]:


print(first_control_group_did.columns)
# Variables in the dataset:
# 'author_id': A provider's id  
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0) 
# 'post': 1 if an observation corresponded to the first reception day and 0 if
#    the observation corresponded to the day before the experiment
# 'whether_follower_send_author_msg_exp_time': Whether the follower who sent a provider the first social nudge 
#    in the experiment (i.e.,the first social-nudge sender) also sent any private messages to that provider between 
#    the start date of the experiment and that provider's first reception day (including both ends)
# 'is_upload': Whether a provider uploaded any videos on the corresponding day 
#    (either the first reception day or the day before the experiment) 
# 'photo_id_cnt_fillzero': A provider's number of videos uploaded on the corresponding day 
#.   (either the first reception day or the day before the experiment)


# In[68]:


first_control_group_did['stand_photo_id_cnt_fillzero'] = first_control_group_did['photo_id_cnt_fillzero']/first_control_group_did['photo_id_cnt_fillzero'].std()


# In[69]:


first_production = first_production.merge(first_private_message, how = 'left')


# Table 14
# 
# The Role of Private Messages in Content Production Among Control Provider

# In[70]:


# table 14; raw data
mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero ~ 1 + whether_follower_send_author_msg_exp_time*post', data=first_control_group_did).fit(cov_type='cluster', cov_kwds={'groups': first_control_group_did['author_id']})
mod2 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=first_production[first_production['whether_follower_send_author_msg_exp_time']==1]).fit()
mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=first_production[first_production['whether_follower_send_author_msg_exp_time']==0]).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )



# In[71]:


# table 14; standardized data
mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero ~ 1 + whether_follower_send_author_msg_exp_time*post', data=first_control_group_did).fit(cov_type='cluster', cov_kwds={'groups': first_control_group_did['author_id']})
mod2 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=first_production[first_production['whether_follower_send_author_msg_exp_time']==1]).fit()
mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=first_production[first_production['whether_follower_send_author_msg_exp_time']==0]).fit()

print('\n##########')
print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))

print('\n##########')
print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('\n##########')
print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )


# ### Table 15 Effects of Social Nudges on Content Production With or Without Controlling for the Role of Likes and Comments

# In[72]:


first_like_and_comment = pd.read_csv('social_nudge_first_exp_like_and_comment.csv')


# In[73]:


print(first_like_and_comment.columns)
# Variables in the dataset first_like_and_comment
# 'author_id': A provider's id  
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0) 
# 'sum_like_cnt_logging_day': The number of likes obtained by a provider on the first reception day
# 'sum_comment_cnt_logging_day': The number of comments obtained by a provider on the first reception day 


# In[74]:


# winsorize variable
for col in ['sum_like_cnt_logging_day','sum_comment_cnt_logging_day'] :
    win_col = 'wins_'+col
    first_like_and_comment[win_col] = first_like_and_comment[col]
    cut_point = first_like_and_comment.loc[first_like_and_comment[win_col]>0, win_col].quantile(0.95)
    print(win_col+'_cut', '=', cut_point)
    first_like_and_comment.loc[first_like_and_comment[win_col] > cut_point, win_col] = cut_point


# In[75]:


# standardize variable
for col in ['wins_sum_like_cnt_logging_day', 'wins_sum_comment_cnt_logging_day']:
    stand_col = 'stand_'+col
    first_like_and_comment[stand_col] = first_like_and_comment[col]/first_like_and_comment[col].std()


# In[76]:


first_production = first_production.merge(first_like_and_comment, how = 'left')


# In[77]:


# Table 15; using raw data

df = first_production
mod1 = sm.OLS.from_formula('wins_sum_like_cnt_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('wins_sum_comment_cnt_logging_day ~ 1 + treatment', data=df).fit()

print('\n##########')
print('panel A column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('panel A column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )


mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment + wins_sum_like_cnt_logging_day', data=df).fit()
mod5 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment + wins_sum_comment_cnt_logging_day', data=df).fit()
mod6 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment + wins_sum_like_cnt_logging_day + wins_sum_comment_cnt_logging_day', data=df).fit()

print('\n##########')
print('panel B column (1):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel B column (2):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel B column (3):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel B column (4):')
print(mod6.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod6.params['treatment']/mod3.params['Intercept']) )


mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment + wins_sum_like_cnt_logging_day', data=df).fit()
mod5 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment + wins_sum_comment_cnt_logging_day', data=df).fit()
mod6 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment + wins_sum_like_cnt_logging_day + wins_sum_comment_cnt_logging_day', data=df).fit()

print('\n##########')
print('panel C column (1):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel C column (2):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel C column (3):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel C column (4):')
print(mod6.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod6.params['treatment']/mod3.params['Intercept']) )


# In[78]:


# Table 15; using standardized data

df = first_production
mod1 = sm.OLS.from_formula('stand_wins_sum_like_cnt_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_wins_sum_comment_cnt_logging_day ~ 1 + treatment', data=df).fit()

print('\n##########')
print('panel A column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('\n##########')
print('panel A column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )


mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment + stand_wins_sum_like_cnt_logging_day', data=df).fit()
mod5 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment + stand_wins_sum_comment_cnt_logging_day', data=df).fit()
mod6 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment + stand_wins_sum_like_cnt_logging_day + stand_wins_sum_comment_cnt_logging_day', data=df).fit()

print('\n##########')
print('panel B column (1):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel B column (2):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel B column (3):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel B column (4):')
print(mod6.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod6.params['treatment']/mod3.params['Intercept']) )


mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment + stand_wins_sum_like_cnt_logging_day', data=df).fit()
mod5 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment + stand_wins_sum_comment_cnt_logging_day', data=df).fit()
mod6 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment + stand_wins_sum_like_cnt_logging_day + stand_wins_sum_comment_cnt_logging_day', data=df).fit()

print('\n##########')
print('panel C column (1):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel C column (2):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel C column (3):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod3.params['Intercept']) )

print('\n##########')
print('panel C column (4):')
print(mod6.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod6.params['treatment']/mod3.params['Intercept']) )


# ### Table 16 Users Who Sent Social Nudges Did Not Reduce the Usage of Likes/Comment

# In[79]:


like_comment_cannibalize_data = pd.read_csv("social_nudge_first_exp_like_comment_cannibalize_data.csv")


# In[80]:


print(like_comment_cannibalize_data.columns)
# Variables in the dataset like_comment_cannibalize_data:
# 'user_id': Id of a user among a random sample of $10, 000, 000$ viewers who logged onto Platform O between 
#    December 1, 2021 and December 7, 2021 for the cannibalization analyses
# 'gender': Gender of a user
# 'age_range': The range of a user's age (0-12, 12-17, 18-23, 24-30, 31-40, 41-49, and 50+)
# 'fre_community_type': A user's residential community type (i.e., countryside, town, or urban area)
# 'fre_city_level': The tier of city a user lives in (e.g., first tier, second tier, etc.)
# 'follow_user_num': The number of users a user was following on November 30, 2021 
# 'fans_user_num': The number of followers a user had on November 30, 2021
# 'prior_online_duration': How long she watched videos Platform O in the previous week,
# 'prior_photo_play_duration': How long a user was active on Platform O in the previous week (November 24—November 30, 2021)
# 'prior_is_photo_author': Whether a user had uploaded any video by November 30, 2021
# 'prior_is_social_ban': Whether a user was banned from social interactions (such as marking likes and sending comments)
# 'sending_nudge_cnt': The number of social nudges sent by a viewer during December 1, 2021 and December 7, 2021
# 'sending_like_cnt': The number of likes marked by a user during December 1, 2021 and December 7, 2021 
# 'sending_comment_cnt': The number of comments left by a user during December 1, 2021 and December 7, 2021 
# 'prior_sending_like_cnt': The number of likes marked by a user during the prior week
# 'prior_sending_comment_cnt': The number of comments left by a user during the prior week


# In[81]:


like_comment_cannibalize_data['treatment'] = (like_comment_cannibalize_data['sending_nudge_cnt']>=1).astype('int')


# In[82]:


print(like_comment_cannibalize_data['age_range'].value_counts())
print(like_comment_cannibalize_data['fre_community_type'].value_counts())
print(like_comment_cannibalize_data['fre_city_level'].value_counts())


# In[83]:


# matching variables to construct comparabel treatment and control groups
feature_cols = [
'gender',
'age_range',
'fre_community_type',
'fre_city_level',
'follow_user_num',
'fans_user_num',
'prior_online_duration',
'prior_photo_play_duration',
'prior_is_photo_author',
'prior_is_social_ban']

# define class
match_cem = matching(df = like_comment_cannibalize_data,
                     feature_cols = feature_cols,
                     is_K2K = True, 
                     break_method = 'sturges')
# match
df_matched_cem = match_cem.cem()


# In[84]:


# raw data
temp1 = df_matched_cem[['user_id', 'treatment', 'sending_like_cnt', 'sending_comment_cnt']]
temp1['post'] = 1
temp2 = df_matched_cem[['user_id', 'treatment', 'prior_sending_like_cnt', 'prior_sending_comment_cnt']]
temp2.columns = ['user_id', 'treatment', 'sending_like_cnt', 'sending_comment_cnt']
temp2['post'] = 0

matched_like_comment_cannibalize_data  = pd.concat([temp1, temp2])

mod1 = sm.OLS.from_formula('sending_like_cnt ~ 1 + treatment*post', data=matched_like_comment_cannibalize_data).fit(cov_type='cluster', cov_kwds={'groups': matched_like_comment_cannibalize_data['user_id']})
print(mod1.summary2(float_format="%.6f"))

mod2 = sm.OLS.from_formula('sending_comment_cnt ~ 1 + treatment*post', data=matched_like_comment_cannibalize_data).fit(cov_type='cluster', cov_kwds={'groups': matched_like_comment_cannibalize_data['user_id']})
print(mod2.summary2(float_format="%.6f"))


# In[85]:


# standardize variable
for col in ['sending_like_cnt', 'sending_comment_cnt']:
    stand_col = 'stand_'+col
    matched_like_comment_cannibalize_data[stand_col] = matched_like_comment_cannibalize_data[col]/matched_like_comment_cannibalize_data[col].std()


# In[86]:


# table 16; std data
mod1 = sm.OLS.from_formula('stand_sending_like_cnt ~ 1 + treatment*post', data=matched_like_comment_cannibalize_data).fit(cov_type='cluster', cov_kwds={'groups': matched_like_comment_cannibalize_data['user_id']})
print(mod1.summary2(float_format="%.6f"))

mod2 = sm.OLS.from_formula('stand_sending_comment_cnt ~ 1 + treatment*post', data=matched_like_comment_cannibalize_data).fit(cov_type='cluster', cov_kwds={'groups': matched_like_comment_cannibalize_data['user_id']})
print(mod2.summary2(float_format="%.6f"))


# ### Table 17 Direct Effects of Social Nudge Across Providers With Different Historical Production Levels

# In[87]:


first_production = first_production.merge(first_author_feature, how = 'left')


# In[88]:


# Table 17; using raw data
# Low-Productivity Providers; Medium-Productivity Providers; High-Productivity Providers

sub_data = first_production
print('quantile 0.9 = ', first_production['upload_photo_cnt_a_week_prior'].quantile(0.9))
print('quantile 0.99 = ', first_production['upload_photo_cnt_a_week_prior'].quantile(0.99))
##########

##########
print('##########')
print('> 0.99')
res = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=sub_data[sub_data['upload_photo_cnt_a_week_prior'] > first_production['upload_photo_cnt_a_week_prior'].quantile(0.99)]).fit()
print(res.get_robustcov_results().summary2(float_format="%.6f"))
print('pvalue:', '%.4f'%res.pvalues['treatment'])
print('res:', '{:.2%}'.format(res.params['treatment']/res.params['Intercept']) )

print('##########')
print('> 0.9 & <= 0.99')
res = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=sub_data[(sub_data['upload_photo_cnt_a_week_prior'] > first_production['upload_photo_cnt_a_week_prior'].quantile(0.9)) & (sub_data['upload_photo_cnt_a_week_prior'] <= first_production['upload_photo_cnt_a_week_prior'].quantile(0.99))]).fit()
print(res.get_robustcov_results().summary2(float_format="%.6f"))
print('pvalue:', '%.4f'%res.pvalues['treatment'])
print('res:', '{:.2%}'.format(res.params['treatment']/res.params['Intercept']) )


print('##########')
print('<= 0.9')
res = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=sub_data[sub_data['upload_photo_cnt_a_week_prior'] <= first_production['upload_photo_cnt_a_week_prior'].quantile(0.9)]).fit()
print(res.get_robustcov_results().summary2(float_format="%.6f"))
print('pvalue:', '%.4f'%res.pvalues['treatment'])
print('res:', '{:.2%}'.format(res.params['treatment']/res.params['Intercept']) )


# In[89]:


# Table 17; using standardized data
# Low-Productivity Providers; Medium-Productivity Providers; High-Productivity Providers

print('##########')
print('> 0.99')
res = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=sub_data[sub_data['upload_photo_cnt_a_week_prior'] > first_production['upload_photo_cnt_a_week_prior'].quantile(0.99)]).fit()
print(res.get_robustcov_results().summary2(float_format="%.6f"))
print('pvalue:', '%.4f'%res.pvalues['treatment'])
print('res:', '{:.2%}'.format(res.params['treatment']/res.params['Intercept']) )

print('##########')
print('> 0.9 & <= 0.99')
res = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=sub_data[(sub_data['upload_photo_cnt_a_week_prior'] > first_production['upload_photo_cnt_a_week_prior'].quantile(0.9)) & (sub_data['upload_photo_cnt_a_week_prior'] <= first_production['upload_photo_cnt_a_week_prior'].quantile(0.99))]).fit()
print(res.get_robustcov_results().summary2(float_format="%.6f"))
print('pvalue:', '%.4f'%res.pvalues['treatment'])
print('res:', '{:.2%}'.format(res.params['treatment']/res.params['Intercept']) )


print('##########')
print('<= 0.9')
res = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=sub_data[sub_data['upload_photo_cnt_a_week_prior'] <= first_production['upload_photo_cnt_a_week_prior'].quantile(0.9)]).fit()
print(res.get_robustcov_results().summary2(float_format="%.6f"))
print('pvalue:', '%.4f'%res.pvalues['treatment'])
print('res:', '{:.2%}'.format(res.params['treatment']/res.params['Intercept']) )


# ### Table 18 Comparison of Social Nudge and Platform-Initiated Nudge

# In[90]:


platform_nudge_production = pd.read_csv('platform_nudge_production.csv')


# In[91]:


print(platform_nudge_production.columns)
# Variables in the dataset platform_nudge_production:
# 'author_id': A provider's id 
# 'treatment': Whether a provider was in the treatment condition (1), versus the control condition (0)  
# 'photo_id_cnt_fillzero_logging_day': A provider’s number of videos uploaded on the first reception day
# 'photo_id_cnt_fillzero_next_day_1': A provider’s number of videos uploaded on the first day after the first reception day (or Day 2)
# 'photo_id_cnt_fillzero_next_day_2': A provider’s number of videos uploaded on the second day after the first reception day (or Day 3)
# 'photo_id_cnt_fillzero_next_day_3': A provider’s number of videos uploaded on the third day after the first reception day (or Day 4)


# In[92]:


overlap_author = platform_nudge_production[['author_id']].merge(first_production[['author_id']], how = 'inner')


# In[93]:


# standardize variable
for col in ['photo_id_cnt_fillzero_logging_day', 'photo_id_cnt_fillzero_next_day_1', 
           'photo_id_cnt_fillzero_next_day_2', 'photo_id_cnt_fillzero_next_day_3']:
    stand_col = 'stand_'+col
    platform_nudge_production[stand_col] = platform_nudge_production[col]/platform_nudge_production[col].std()


# In[94]:


# Table 18; using raw data

df = platform_nudge_production

mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('panel A column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('panel A column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('panel A column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('panel A column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )

mod5 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment',                            data=platform_nudge_production.merge(overlap_author, how='inner')).fit()

mod6 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment',                            data=first_production.merge(overlap_author, how='inner')).fit()

print('panel B column (1):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod5.params['Intercept']) )

print('panel B column (2):')
print(mod6.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod6.params['treatment']/mod6.params['Intercept']) )


# In[95]:


# Table 18; using standardized data

df = platform_nudge_production

mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('panel A column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('panel A column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('panel A column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('panel A column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )

mod5 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment',                            data=platform_nudge_production.merge(overlap_author, how='inner')).fit()

mod6 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment',                            data=first_production.merge(overlap_author, how='inner')).fit()

print('panel B column (1):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod5.params['Intercept']) )

print('panel B column (2):')
print(mod6.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod6.params['treatment']/mod6.params['Intercept']) )


# ### Figure 4

# In[96]:


# Figure 4

dvs = ['stand_photo_id_cnt_fillzero_logging_day',
           'stand_photo_id_cnt_fillzero_next_day_1',
           'stand_photo_id_cnt_fillzero_next_day_2',
           'stand_photo_id_cnt_fillzero_next_day_3']

outcome = 'photo_id_cnt_fillzero'

X_0_cnt = []
Y_0_cnt = []
Y_error_low_cnt = []
Y_error_high_cnt = []

X_0_cnt_social = []
Y_0_cnt_social= []
Y_error_low_cnt_social = []
Y_error_high_cnt_social = []

sub_data = platform_nudge_production 

for date in range(4):
    f = dvs[date] +' ~ 1 + treatment'
    mod = sm.OLS.from_formula(f, data=sub_data)
    res = mod.fit()
    print(res.get_robustcov_results().summary2(float_format="%.6f"))
    print('res:', '{:.2%}'.format(res.params['treatment']/res.params['Intercept']) )
    X_0_cnt.append(date+1)
    Y_0_cnt.append(res.params['treatment']/res.params['Intercept']*100)
    Y_error_low_cnt.append(res.conf_int(alpha=0.05).loc['treatment', 0]/res.params['Intercept']*100)
    Y_error_high_cnt.append(res.conf_int(alpha=0.05).loc['treatment', 1]/res.params['Intercept']*100)

sub_data = first_production
for date in range(4):
    f = dvs[date] +' ~ 1 + treatment'
    mod = sm.OLS.from_formula(f, data=sub_data)
    res = mod.fit()
    print(res.get_robustcov_results().summary2(float_format="%.6f"))
    print('res:', '{:.2%}'.format(res.params['treatment']/res.params['Intercept']) )
    X_0_cnt_social.append(date+1)
    Y_0_cnt_social.append(res.params['treatment']/res.params['Intercept']*100)
    Y_error_low_cnt_social.append(res.conf_int(alpha=0.05).loc['treatment', 0]/res.params['Intercept']*100)
    Y_error_high_cnt_social.append(res.conf_int(alpha=0.05).loc['treatment', 1]/res.params['Intercept']*100)

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.figure(figsize=(10, 10), dpi=80, edgecolor='black') 
plt.gca().spines['bottom'].set_color('grey')
plt.gca().spines['top'].set_color('grey') 
plt.gca().spines['right'].set_color('grey')
plt.gca().spines['left'].set_color('grey')


colors_1 = ['darkgoldenrod', 'darkgoldenrod', 'darkgoldenrod', 'darkgoldenrod', 'darkgoldenrod', 'darkgoldenrod', 'darkgoldenrod', 'darkgoldenrod']
colors_2 = ['crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson']


for i in range(4):
    plt.plot(X_0_cnt[i], Y_0_cnt[i], color = colors_1[i], marker = '<', markersize = 12, alpha = 0.9, label = 'photo_id_cnt_fillzero')
    plt.plot(X_0_cnt[i], Y_error_low_cnt[i], color= colors_1[i], marker = '_', markersize = 12, alpha = 0.5)
    plt.plot(X_0_cnt[i], Y_error_high_cnt[i], color= colors_1[i], marker = '_', markersize = 12, alpha = 0.5)
    plt.vlines(x = X_0_cnt[i], ymin = Y_error_low_cnt[i], ymax = Y_error_high_cnt[i], color = colors_1[i], linestyles = 'dotted', alpha = 0.5)
    
    plt.plot(X_0_cnt_social[i], Y_0_cnt_social[i], color = colors_2[i], marker = 'o', markersize = 12, alpha = 0.4, label = 'photo_id_cnt_social_fillzero')
    plt.plot(X_0_cnt_social[i], Y_error_low_cnt_social[i], color= colors_2[i], marker = '_', markersize = 12, alpha = 0.5)
    plt.plot(X_0_cnt_social[i], Y_error_high_cnt_social[i], color= colors_2[i], marker = '_', markersize = 12, alpha = 0.5)
    plt.vlines(x = X_0_cnt_social[i], ymin = Y_error_low_cnt_social[i], ymax = Y_error_high_cnt_social[i], color = colors_2[i], linestyles = 'dotted', alpha = 0.5)

    plt.xlabel('Day t from the first reception day on', fontsize=18)
    plt.ylabel('Relative Effect on Number of Videos Uploaded (%)', fontsize=18)
    plt.grid(True, linestyle = "-", color = "grey", linewidth = "0.5", alpha = 0.5)
    
    colors = ['darkgoldenrod', 'crimson']
    lines = [Line2D([0], [0], color='crimson', linewidth=1, linestyle='dotted', marker = 'o', markersize = 12, alpha = 0.4), Line2D([0], [0], color='darkgoldenrod', linewidth=1, linestyle='dotted', marker = '<', markersize = 12, alpha = 0.9) ]
    labels = ['Social Nudge', 'Platform-Initiated Nudge']
    plt.legend(lines, labels, fontsize = 'xx-large')
    
    plt.savefig('OVP_nudge_all_providers_over_time_rel_effect_cnt.png', bbox_inches='tight')
    #plt.savefig('.svg', bbox_inches='tight', dpi=100, format="svg")


# # Appendix E Social Network Model Estimation Details
# 

# ### Table 19 The Results of a Logistic Regression Model Predicting Social-Nudge Incidence

# In[97]:


intrinsic_motivation_control_group = pd.read_csv('social_nudge_first_exp_intrinsic_motivation_control_group.csv')


# In[98]:


print(intrinsic_motivation_control_group.columns)
# Variables in the dataset intrinsic_motivation_control_group
# 'target_id': e_d's id 
# 'source_id': e_o's id
# 'author_follow_cnt':e_d’s number of following
# 'author_fans_cnt': e_d’s number of followers
# 'source_follow_cnt': e_o’s number of following
# 'source_fans_cnt':  e_o’s number of followers 
# 'whether_author_follow_source': Whether e_d was also following e_o
# 'sending_nudge_cnt': The number of social nudges sent by e_o to e_d
# 'author_daily_upload_cnt_h30d': e_d's average number of videos uploaded per day across the 30 days before the experiment


# In[99]:


intrinsic_motivation_control_group['whether_sending_nudge'] = (intrinsic_motivation_control_group['sending_nudge_cnt'] >= 1).astype('int')
intrinsic_motivation_control_group.fillna(0, inplace = True)


# In[100]:


# describle variables
for col in ['sending_nudge_cnt', 'whether_sending_nudge']:
    print(col,': # null = ', sum(pd.isnull(intrinsic_motivation_control_group[col])))
    print(intrinsic_motivation_control_group[col].value_counts())


# In[101]:


# data processing  
# cut
for col in ['author_follow_cnt', 'author_fans_cnt', 'source_follow_cnt', 'source_fans_cnt', 
             'author_daily_upload_cnt_h30d']:
    col_name = 'high_' + col
    high_cut = int(intrinsic_motivation_control_group[col].quantile(0.5))
    print(col+'_high_cut = ', high_cut)
    intrinsic_motivation_control_group[col_name] = (intrinsic_motivation_control_group[col] > high_cut).astype('int')

# winsorize
for col in ['author_follow_cnt', 'author_fans_cnt', 'source_follow_cnt', 'source_fans_cnt', 
           'author_daily_upload_cnt_h30d'] :
    win_col = 'wins_'+col
    intrinsic_motivation_control_group[win_col] = intrinsic_motivation_control_group[col]
    cut_point = intrinsic_motivation_control_group.loc[intrinsic_motivation_control_group[win_col]>0, win_col].quantile(0.95)
    print('mu_' + win_col+'_cut = ', cut_point)
    intrinsic_motivation_control_group.loc[intrinsic_motivation_control_group[win_col] > cut_point, win_col] = cut_point

# log-transforming
for col in ['author_follow_cnt', 'author_fans_cnt', 'source_follow_cnt', 'source_fans_cnt', 
           'author_daily_upload_cnt_h30d'] :
    log_col = 'log_'+col
    intrinsic_motivation_control_group[log_col] = np.log(intrinsic_motivation_control_group[col]+1)# data processing (cutting, winsorizing, log-transforming)


# In[102]:


# logistic regression
intrinsic_motivation_control_group['Intercept'] = 1
print('##################')
sub_data = intrinsic_motivation_control_group
res = sm.Logit.from_formula('whether_sending_nudge ~ 1 + high_source_follow_cnt + high_source_fans_cnt + whether_author_follow_source + author_daily_upload_cnt_h30d', data=sub_data).fit()
print(res.summary2(float_format="%.6f"))
print('mu_coef_Intercept=', '%.10f'%res.params['Intercept'], 'exp coef = ', '%.6f'%np.exp(res.params['Intercept']))
print('mu_coef_high_source_follow_cnt=', '%.10f'%res.params['high_source_follow_cnt'], 'exp coef = ', '%.6f'%np.exp(res.params['high_source_follow_cnt']))
print('mu_coef_high_source_fans_cnt=', '%.10f'%res.params['high_source_fans_cnt'], 'exp coef = ', '%.6f'%np.exp(res.params['high_source_fans_cnt']))
print('mu_coef_whether_author_follow_source=', '%.10f'%res.params['whether_author_follow_source'], 'exp coef = ', '%.6f'%np.exp(res.params['whether_author_follow_source']))
print('mu_coef_author_daily_upload_cnt_h30d=', '%.10f'%res.params['author_daily_upload_cnt_h30d'], 'exp coef = ', '%.6f'%np.exp(res.params['author_daily_upload_cnt_h30d']))


# In[103]:


# cross validation
intrinsic_motivation_control_group['Intercept'] = 1

col = ['Intercept', 'wins_source_follow_cnt', 'wins_source_fans_cnt', 'whether_author_follow_source', 'wins_author_daily_upload_cnt_h30d']

X = intrinsic_motivation_control_group[col]
y = intrinsic_motivation_control_group['whether_sending_nudge']

cv_results = cross_validate(LogisticRegression(), X, y, cv=5, scoring=['accuracy', 'roc_auc'])
print('accuracy = ',cv_results['test_accuracy'].mean())
print('auc = ', cv_results['test_roc_auc'].mean())


# ### Table 20 Over-Time Direct Effects of Receiving One Social Nudge on Content Production

# In[104]:


first_production = first_production.merge(first_author_feature, how='left')


# In[105]:


# Table 20; using raw data

df = first_production[first_production['logging_msg_cnt'] == 1]

mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod1.params['treatment'])
p_11 = mod1.params['treatment']

print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod2.params['treatment'])
p_12 = mod2.params['treatment']

print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod3.params['treatment'])
p_13 = mod3.params['treatment']

print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod4.params['treatment'])
p_14 = mod4.params['treatment']

print('########')
df = second_exp[second_exp['logging_msg_cnt'] == 1]

mod1 = sm.OLS.from_formula('photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()
mod5 = sm.OLS.from_formula('photo_id_cnt_fillzero_next_day_4 ~ 1 + treatment', data=df).fit()

print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod1.params['treatment'])
p_21 = mod1.params['treatment']

print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod2.params['treatment'])
p_22 = mod2.params['treatment']

print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod3.params['treatment'])
p_23 = mod3.params['treatment']

print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod4.params['treatment'])
p_24 = mod4.params['treatment']

print('column (5):')
print(mod5.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod5.params['treatment']/mod5.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod5.params['treatment'])
p_25 = mod5.params['treatment']


# In[106]:


# Table 20; using standardized data

df = first_production[first_production['logging_msg_cnt'] == 1]

mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )

print('########')
df = second_exp[second_exp['logging_msg_cnt'] == 1]

mod1 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_logging_day ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_1 ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_2 ~ 1 + treatment', data=df).fit()
mod4 = sm.OLS.from_formula('stand_photo_id_cnt_fillzero_next_day_3 ~ 1 + treatment', data=df).fit()

print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('column (4):')
print(mod4.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod4.params['treatment']/mod4.params['Intercept']) )


# ### Table 8 Estimation of Parameters in the Social Network Model  -- p and alpha_p

# In[107]:


# p; first experiment; based on absolute effect

y = np.array([p_11, p_12, p_13])

def objective_function(x):
    x1 = x[0]
    x2 = x[1]
    sum_squared_error = 0
    for i in range(len(y)):
        sum_squared_error = sum_squared_error + (y[i]-x1*np.power(x2,i))**2
    return sum_squared_error

x0 = [0.01, 0.6]
res = least_squares(objective_function, x0, bounds=([0, 0], [1, 1]), ftol=3e-16, xtol=3e-16, gtol=3e-16, max_nfev=10000000, method ='dogbox')

print('[p, alpha_p]')
print('initial guess: ', x0, ';', 'initial cost: ', objective_function(x0))
print('final guess: ', res.x, ';', 'final cost: ', res.cost)
print('---------------')
print('p = ', '%.10f'%(res.x[0]), '%.10f'%res.x[0])
print('alpha_p = ', '%.10f'%res.x[1])
print('p/(1-alpha_p) = ', '%.10f'%(res.x[0]/(1-res.x[1])))


# In[108]:


# second experiment; based on absolute effect

y = np.array([p_21, p_22, p_23, p_24])

def objective_function(x):
    x1 = x[0]
    x2 = x[1]
    sum_squared_error = 0
    for i in range(len(y)):
        sum_squared_error = sum_squared_error + (y[i]-x1*np.power(x2,i))**2
    return sum_squared_error

x0 = [0.01, 0.6]
res = least_squares(objective_function, x0, bounds=([0, 0], [1, 1]), ftol=3e-16, xtol=3e-16, gtol=3e-16, max_nfev=10000000, method ='dogbox')

print('[p, alpha_p]')
print('initial guess: ', x0, ';', 'initial cost: ', objective_function(x0))
print('final guess: ', res.x, ';', 'final cost: ', res.cost)
print('---------------')
print('p = ', '%.10f'%(res.x[0]), '%.10f'%res.x[0])
print('alpha_p = ', '%.10f'%res.x[1])
print('p/(1-alpha_p) = ', '%.10f'%(res.x[0]/(1-res.x[1])))


# ### Table 21 Over-Time Diffusion Effects of Receiving One Social Nudge

# In[109]:


first_diffusion = first_diffusion.merge(first_author_feature, how='left')


# In[110]:


first_diffusion.loc[first_diffusion['author_follow_cnt'] == 0, 'nudges_to_others_logging_day_per_link'] = None
first_diffusion.loc[first_diffusion['author_follow_cnt'] == 0, 'nudges_to_others_next_day_1_per_link'] = None
first_diffusion.loc[first_diffusion['author_follow_cnt'] == 0, 'nudges_to_others_next_day_1_per_link'] = None


# In[111]:


for col in ['nudges_to_others_logging_day_per_link', 
'nudges_to_others_next_day_1_per_link',
'nudges_to_others_next_day_2_per_link']:
    stand_col = 'stand_'+col
    first_diffusion[stand_col] = first_diffusion[col]/first_diffusion[col].std()


# In[112]:


second_exp.loc[second_exp['author_follow_cnt'] == 0, 'nudges_to_others_logging_day_per_link'] = None
second_exp.loc[second_exp['author_follow_cnt'] == 0, 'nudges_to_others_next_day_1_per_link'] = None
second_exp.loc[second_exp['author_follow_cnt'] == 0, 'nudges_to_others_next_day_1_per_link'] = None


# In[113]:


for col in ['nudges_to_others_logging_day_per_link', 
'nudges_to_others_next_day_1_per_link',
'nudges_to_others_next_day_2_per_link']:
    stand_col = 'stand_'+col
    second_exp[stand_col] = second_exp[col]/second_exp[col].std()


# In[114]:


# Table 21; using raw data

df = first_diffusion[(first_diffusion['logging_msg_cnt'] == 1) & (first_diffusion['author_follow_cnt'] > 0)]

mod1 = sm.OLS.from_formula('nudges_to_others_logging_day_per_link ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('nudges_to_others_next_day_1_per_link ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('nudges_to_others_next_day_2_per_link ~ 1 + treatment', data=df).fit()

print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod1.params['treatment'])
d_11 = mod1.params['treatment']

print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod2.params['treatment'])
d_12 = mod2.params['treatment']

print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod3.params['treatment'])
d_13 = mod3.params['treatment']

print('########')
df = second_exp[(second_exp['logging_msg_cnt'] == 1) & (second_exp['author_follow_cnt'] > 0)]

mod1 = sm.OLS.from_formula('nudges_to_others_logging_day_per_link ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('nudges_to_others_next_day_1_per_link ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('nudges_to_others_next_day_2_per_link ~ 1 + treatment', data=df).fit()

print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod1.params['treatment'])
d_21 = mod1.params['treatment']

print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod2.params['treatment'])
d_22 = mod2.params['treatment']

print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )
print('coef of treatment:', '%.10f'%mod3.params['treatment'])
d_23 = mod3.params['treatment']


# In[115]:


# Table 21; using standardized data

df = first_diffusion[(first_diffusion['logging_msg_cnt'] == 1) & (first_diffusion['author_follow_cnt'] > 0)]

mod1 = sm.OLS.from_formula('stand_nudges_to_others_logging_day_per_link ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_nudges_to_others_next_day_1_per_link ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_nudges_to_others_next_day_2_per_link ~ 1 + treatment', data=df).fit()

print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )

print('########')
df = second_exp[(second_exp['logging_msg_cnt'] == 1) & (second_exp['author_follow_cnt'] > 0)]

mod1 = sm.OLS.from_formula('stand_nudges_to_others_logging_day_per_link ~ 1 + treatment', data=df).fit()
mod2 = sm.OLS.from_formula('stand_nudges_to_others_next_day_1_per_link ~ 1 + treatment', data=df).fit()
mod3 = sm.OLS.from_formula('stand_nudges_to_others_next_day_2_per_link ~ 1 + treatment', data=df).fit()

print('column (1):')
print(mod1.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod1.params['treatment']/mod1.params['Intercept']) )

print('column (2):')
print(mod2.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod2.params['treatment']/mod2.params['Intercept']) )

print('column (3):')
print(mod3.get_robustcov_results().summary2(float_format="%.5f"))
print('relative effect size:', '{:.2%}'.format(mod3.params['treatment']/mod3.params['Intercept']) )


# ### Table 8 Estimation of Parameters in the Social Network Model -- d and alpha_d

# In[116]:


# d; ; first experiment; based on absolute effect

y = np.array([d_11, d_12])*10000

def objective_function(x):
    x1 = x[0]
    x2 = x[1]
    sum_squared_error = 0
    for i in range(len(y)):
        sum_squared_error = sum_squared_error + (y[i]-x1*np.power(x2,i))**2
    return sum_squared_error

x0 = [0.00018*10000, 0.3]
res = least_squares(objective_function, x0, bounds=([0, 0], [10000, 1]), ftol=3e-16, xtol=3e-16, gtol=3e-16, max_nfev=1000000, method ='dogbox')
print('[d, alpha_d]')
print('initial guess: ', x0, ';', 'initial cost: ', objective_function(x0))
print('final guess: ', res.x, ';', 'initial cost: ', res.cost)
print('---------------')
print('d = ', '%.10f'%(res.x[0]/10000), '%.10f'%(res.x[0]))
print('alpha_d = ', '%.10f'%res.x[1])
print('d/alpha_d = ', '%.10f'%((res.x[0]/10000)/res.x[1]))


# In[117]:


# d; ; second experiment; based on absolute effect

y = np.array([d_21, d_22])*10000

def objective_function(x):
    x1 = x[0]
    x2 = x[1]
    sum_squared_error = 0
    for i in range(len(y)):
        sum_squared_error = sum_squared_error + (y[i]-x1*np.power(x2,i))**2
    return sum_squared_error

x0 = [0.00018*10000, 0.3]
res = least_squares(objective_function, x0, bounds=([0, 0], [10000, 1]), ftol=3e-16, xtol=3e-16, gtol=3e-16, max_nfev=1000000, method ='dogbox')
print('[d, alpha_d]')
print('initial guess: ', x0, ';', 'initial cost: ', objective_function(x0))
print('final guess: ', res.x, ';', 'initial cost: ', res.cost)
print('---------------')
print('d = ', '%.10f'%(res.x[0]/10000), '%.10f'%(res.x[0]))
print('alpha_d = ', '%.10f'%res.x[1])
print('d/alpha_d = ', '%.10f'%((res.x[0]/10000)/res.x[1]))


# ## Appendix G Data discloure

# ### Tables 25--36

# In[118]:


# variables first appearing in table 1
outcomes = ['author_gender_binary', 'stand_author_fans_cnt', 'stand_author_follow_cnt', 'stand_upload_photo_cnt_a_week_prior',
           'stand_upload_days_a_week_prior']

sub_data = first_author_feature
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[119]:


# variables first appearing in  table 2
outcomes = ['stand_photo_id_cnt_fillzero_logging_day', 'is_upload_logging_day', 
            'stand_photo_id_cnt_fillzero_logging_day_conditional_uploding', 'is_bi_follow']

sub_data = first_production
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[120]:


# variables first appearing in table 3
outcomes = ['stand_wins_sum_play_cnt', 'stand_avg_complete_per_play_ratio', 'stand_avg_like_per_play_ratio',
           'stand_avg_comment_per_play_ratio', 'stand_avg_follow_per_play_ratio', 'stand_his_like_per_play_2018']

sub_data = first_logging_consumption
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[121]:


# variables first appearing in table 4
outcomes = ['stand_photo_id_cnt_fillzero_next_day_1','stand_photo_id_cnt_fillzero_next_day_2',
           'stand_photo_id_cnt_fillzero_next_day_3']

sub_data = first_production
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[122]:


# variables first appearing in table 5 and table 6
outcomes = ['stand_wins_nudges_to_others_logging_day',
            'stand_wins_nudges_to_others_next_day_1','stand_wins_nudges_to_others_next_day_2']

sub_data = first_diffusion
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[123]:


# variables appearing in table 10
outcomes = ['stand_photo_id_cnt_fillzero', 'stand_wins_nudges_to_others_logging_day']

sub_data = first_no_prior_nudge_limit
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[124]:


# variables appearing in table 11
outcomes = ['stand_photo_id_cnt_fillzero', 'is_upload']

sub_data = first_production_in_24h
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[125]:


# variables appearing in  table 12 and table 13
outcomes = ['stand_photo_id_cnt_fillzero_logging_day',
            'stand_photo_id_cnt_fillzero_next_day_1','stand_photo_id_cnt_fillzero_next_day_2',
           'stand_photo_id_cnt_fillzero_next_day_3',
           'stand_wins_nudges_to_others_logging_day',
            'stand_wins_nudges_to_others_next_day_1',
            'stand_wins_nudges_to_others_next_day_2',
            'stand_wins_nudges_to_others_next_day_3',
            'stand_wins_nudges_to_others_next_day_4']

sub_data = second_exp
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[126]:


# variables appearing in  table 14
outcomes = ['stand_photo_id_cnt_fillzero', 'whether_follower_send_author_msg_exp_time', 'post']

sub_data = first_control_group_did
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[127]:


# variables appearing in table 14
outcomes = ['stand_photo_id_cnt_fillzero_logging_day']

sub_data = first_production[first_production['whether_follower_send_author_msg_exp_time']==1]
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[128]:


# variables appearing in table 14
outcomes = ['stand_photo_id_cnt_fillzero_logging_day']

sub_data = first_production[first_production['whether_follower_send_author_msg_exp_time']==0]
print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[129]:


# variables appearing in table 15
outcomes = ['stand_wins_sum_like_cnt_logging_day', 'stand_wins_sum_comment_cnt_logging_day']

sub_data = first_like_and_comment

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[130]:


# variables appearing in table 16
sub_data = matched_like_comment_cannibalize_data

outcomes = ['stand_sending_like_cnt', 'stand_sending_comment_cnt', 'treatment', 'post']

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')

for p in [0,1]:
    temp_data = sub_data[sub_data['post'] == p]
    print('post = ', p)
    for col in outcomes:
        outcome = col

        d1 = temp_data[(temp_data['treatment'] == 1) & pd.notnull(temp_data[outcome])][outcome]
        d0 = temp_data[(temp_data['treatment'] == 0) & pd.notnull(temp_data[outcome])][outcome]

        obs1 = len(d1)
        obs0 = len(d0)

        mean1 = np.mean(d1)
        std1 = np.std(d1, ddof=1)

        mean0 = np.mean(d0)
        std0 = np.std(d0, ddof=1)

        print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)



print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[131]:


# variables appearing in table 17
outcomes = ['stand_photo_id_cnt_fillzero_logging_day']

sub_data = first_production[first_production['upload_photo_cnt_a_week_prior'] <= 3]

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[132]:


# variables appearing in  table 17
outcomes = ['stand_photo_id_cnt_fillzero_logging_day']

sub_data = first_production[(first_production['upload_photo_cnt_a_week_prior'] > 3) & ((first_production['upload_photo_cnt_a_week_prior'] <= 13))]

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[133]:


# variables appearing in  table 17
outcomes = ['stand_photo_id_cnt_fillzero_logging_day']

sub_data = first_production[first_production['upload_photo_cnt_a_week_prior'] > 13]

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[134]:


# variables appearing in  table 18
outcomes = ['stand_photo_id_cnt_fillzero_logging_day',
            'stand_photo_id_cnt_fillzero_next_day_1','stand_photo_id_cnt_fillzero_next_day_2',
           'stand_photo_id_cnt_fillzero_next_day_3']

sub_data = platform_nudge_production

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[135]:


# variables appearing in table 18
outcomes = ['stand_photo_id_cnt_fillzero_logging_day']

sub_data = platform_nudge_production.merge(overlap_author, how = 'inner')

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[136]:


# variables appearing in  table 18
outcomes = ['stand_photo_id_cnt_fillzero_logging_day']

sub_data = first_production.merge(overlap_author, how = 'inner')

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[137]:


# variables appearing in Table 20

outcomes = ['stand_photo_id_cnt_fillzero_logging_day',
            'stand_photo_id_cnt_fillzero_next_day_1','stand_photo_id_cnt_fillzero_next_day_2',
           'stand_photo_id_cnt_fillzero_next_day_3']

sub_data = first_production[first_production['logging_msg_cnt'] == 1]

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[138]:


# variables appearing in Table 21

outcomes = ['stand_nudges_to_others_logging_day_per_link',
            'stand_nudges_to_others_next_day_1_per_link', 'stand_nudges_to_others_next_day_2_per_link']

sub_data = first_diffusion[first_diffusion['logging_msg_cnt'] == 1]

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# In[139]:


# variables appearing in Tables 20,21
outcomes = ['stand_photo_id_cnt_fillzero_logging_day',
            'stand_photo_id_cnt_fillzero_next_day_1','stand_photo_id_cnt_fillzero_next_day_2',
           'stand_photo_id_cnt_fillzero_next_day_3', 'stand_photo_id_cnt_fillzero_next_day_4',
           'stand_nudges_to_others_logging_day_per_link',
            'stand_nudges_to_others_next_day_1_per_link', 'stand_nudges_to_others_next_day_2_per_link']

sub_data = second_exp[second_exp['logging_msg_cnt'] == 1]

print('outcome', 'exp_obs_cnt', 'base_obs_cnt', 'exp_mean', 'base_mean', 'exp_std', 'base_std')

for col in outcomes:
    outcome = col
    
    d1 = sub_data[(sub_data['treatment'] == 1) & pd.notnull(sub_data[outcome])][outcome]
    d0 = sub_data[(sub_data['treatment'] == 0) & pd.notnull(sub_data[outcome])][outcome]

    obs1 = len(d1)
    obs0 = len(d0)

    mean1 = np.mean(d1)
    std1 = np.std(d1, ddof=1)

    mean0 = np.mean(d0)
    std0 = np.std(d0, ddof=1)
    
    print(outcome,':', format(obs1, ',d'), '&', '%.4f'%mean1, '&','%.4f'%std1, '&', format(obs0, ',d'), '&', '%.4f'%mean0, '&', '%.4f'%std0)

print('\n')
print('outcome', 'min', '25%', '50%', '75%', 'max')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].min(), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].max())

print('\n')
print('outcome', '1%', '25%', '50%', '75%', '99%')
for col in outcomes:
    outcome = col
    print(outcome,':', '%.4f'%sub_data[outcome].quantile(0.01), '&', '%.4f'%sub_data[outcome].quantile(0.25), '&',           '%.4f'%sub_data[outcome].quantile(0.5), '&', '%.4f'%sub_data[outcome].quantile(0.75), '&',           '%.4f'%sub_data[outcome].quantile(0.99))


# ### Correlation between variables

# In[140]:


temp1 = first_production[['author_id', 'author_gender_binary', 'stand_author_fans_cnt', 'stand_author_follow_cnt', 'stand_upload_photo_cnt_a_week_prior',
'stand_upload_days_a_week_prior', 'stand_photo_id_cnt_fillzero_logging_day', 'is_upload_logging_day', 
'stand_photo_id_cnt_fillzero_logging_day_conditional_uploding', 'is_bi_follow', 
 'stand_photo_id_cnt_fillzero_next_day_1','stand_photo_id_cnt_fillzero_next_day_2',
'stand_photo_id_cnt_fillzero_next_day_3']]
 
temp2 = first_logging_consumption[['author_id','stand_wins_sum_play_cnt', 'stand_avg_complete_per_play_ratio', 'stand_avg_like_per_play_ratio',
'stand_avg_comment_per_play_ratio', 'stand_avg_follow_per_play_ratio', 'stand_his_like_per_play_2018']]

temp3 = first_diffusion[['author_id','stand_wins_nudges_to_others_logging_day',
            'stand_wins_nudges_to_others_next_day_1','stand_wins_nudges_to_others_next_day_2']]


# In[141]:


sub_data = temp1.merge(temp2, how = 'left')
sub_data = sub_data.merge(temp3, how = 'left')


# In[142]:


sub_data.drop(columns = ['author_id'], inplace=True)


# In[143]:


temp = sub_data.corr().reset_index()


# In[144]:


temp


# ### Table 37

# In[145]:


col = sub_data.columns
i_1 = 0

i = 0
for col_1 in range(len(col)):
    for col_2 in range(col_1+1, len(col)):
        
        corr = temp.loc[temp['index'] == col[col_1], col[col_2]].values[0]
        if abs(corr)>0.3:
            print(col[col_1], '&', col[col_2], '&', '%.4f'%corr)
            i = i+1
print(i)

