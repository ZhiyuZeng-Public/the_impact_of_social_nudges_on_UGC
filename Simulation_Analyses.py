
# coding: utf-8

# # Import packadges

# In[1]:


import pandas as pd 
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# # Summarize data

# In[2]:


provider_data = pd.read_csv('social_nudge_network_provider_data.csv')
indegree_data = pd.read_csv('social_nudge_network_indegree_data.csv')
outdegree_data = pd.read_csv('social_nudge_network_outdegree_data.csv')


# In[3]:


print(provider_data.columns)
# 'target_id': id of user i in hat V
# 'mu_target_daily_upload_cnt': previous productivity


# In[4]:


provider_data.head()


# In[5]:


provider_data.shape


# In[6]:


print(indegree_data.columns)
# 'source_id': a user who is following user i in hat V
# 'source_follow_cnt': number of following that source_id has
# 'source_fans_cnt': number of followers that source_id has
# 'target_id': id of user i in hat V
# 'is_bi_follow': whether target_id is also following source_id


# In[7]:


indegree_data.head()


# In[8]:


print(outdegree_data.columns)
# 'target_id': id of user i in hat V
# 'second_stage_target_id': a user whom user i is following
# 'second_stage_target_follow_cnt': number of following that second_stage_target_id has
# 'second_stage_target_fans_cnt': number of followers that second_stage_target_id has
# 'second_stage_is_bi_follow': whether second_stage_target_id is also following target_id


# In[9]:


outdegree_data.head()


# In[10]:


temp1 = indegree_data.groupby('target_id').agg({'source_id':'count'}).reset_index()
temp1.columns = ['target_id', 'source_id_cnt']
temp2 = outdegree_data.groupby('target_id').agg({'second_stage_target_id':'count'}).reset_index()
temp2.columns = ['target_id', 'second_stage_target_id_cnt']

provider_data = provider_data.merge(temp1, how = 'left')
provider_data = provider_data.merge(temp2, how = 'left')

provider_data[['source_id_cnt', 'second_stage_target_id_cnt']].fillna(0, inplace=True)


# In[14]:


provider_data[['source_id_cnt', 'second_stage_target_id_cnt']].quantile([0.001,0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])


# In[16]:


# scale_number = ***  # uncovered due to data privacy issue
provider_data['scaled_source_id_cnt'] = provider_data['source_id_cnt']/scale_number
provider_data['scaled_second_stage_target_id_cnt'] = provider_data['second_stage_target_id_cnt']/scale_number

print(provider_data['scaled_source_id_cnt'].quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]))
print(provider_data['scaled_second_stage_target_id_cnt'].quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]))


# # Calculation of the global effect

# In[18]:


provider_data = pd.read_csv('social_nudge_network_provider_data.csv')
indegree_data = pd.read_csv('social_nudge_network_indegree_data.csv')
outdegree_data = pd.read_csv('social_nudge_network_outdegree_data.csv')


# In[19]:


# # Table 22; using another network sample
# provider_data = pd.read_csv('social_nudge_network_provider_r1_data.csv')
# indegree_data = pd.read_csv('social_nudge_network_indegree_r1_data.csv')
# outdegree_data = pd.read_csv('social_nudge_network_outdegree_r1_data.csv')


# In[20]:


# p = *** # uncovered due to data privacy issue
alpha_p = 0.6344675898 
# d = *** # uncovered due to data privacy issue
alpha_d = 0.3749880798 


# In[21]:


source_follow_cnt_cut = 332
source_fans_cnt = 8
# source_follow_cnt_cut = *** # uncovered due to data privacy issue
# source_fans_cnt = *** # uncovered due to data privacy issue
indegree_data['mu_source_follow_cnt'] = (indegree_data['source_follow_cnt'] > source_follow_cnt_cut).astype('int')
indegree_data['mu_source_fans_cnt'] = (indegree_data['source_fans_cnt'] > source_fans_cnt).astype('int')
indegree_data = indegree_data.merge(provider_data[['target_id', 'mu_target_daily_upload_cnt']], how = 'left')
indegree_data.fillna(0, inplace=True)


# In[22]:


indegree_data['mu'] = 1/(1+np.exp(-(-9.9943001405 + indegree_data['is_bi_follow']*1.0308743231                                     + indegree_data['mu_source_follow_cnt']*(-0.8518251036) +                                     indegree_data['mu_source_fans_cnt']*1.4398194834 +                                     indegree_data['mu_target_daily_upload_cnt']*(-0.3976876693)))) 


# In[23]:


first_stage_production_boost = indegree_data.groupby('target_id').agg({'mu':'sum'}).reset_index()
first_stage_production_boost.columns = ['target_id', 'sum_mu']
first_stage_production_boost['original_boost'] = first_stage_production_boost['sum_mu']*p/(1-alpha_p)


# In[24]:


outdegree_data = outdegree_data.merge(first_stage_production_boost[['target_id', 'sum_mu']], how = 'left')


# In[25]:


outdegree_data['diffusion_boost'] = outdegree_data['sum_mu']*d*p/(1-alpha_p)/(1-alpha_d)


# In[26]:


second_stage_production_boost = outdegree_data.groupby(['target_id']).agg({'diffusion_boost':'sum'}).reset_index()


# In[27]:


print(first_stage_production_boost['original_boost'].sum()*(1-alpha_p))
print(first_stage_production_boost['original_boost'].sum())
print(second_stage_production_boost['diffusion_boost'].sum())


# ### Using parameters estimated from the replicated experiment

# In[28]:


# p =  *** # uncovered due to data privacy issue
alpha_p =  0.6944755157
# d = *** # uncovered due to data privacy issue
alpha_d =  0.3377556067


# In[29]:


indegree_data['mu'] = 1/(1+np.exp(-(-9.9943001405 + indegree_data['is_bi_follow']*1.0308743231                                     + indegree_data['mu_source_follow_cnt']*(-0.8518251036) +                                     indegree_data['mu_source_fans_cnt']*1.4398194834 +                                     indegree_data['mu_target_daily_upload_cnt']*(-0.3976876693)))) 

first_stage_production_boost = indegree_data.groupby('target_id').agg({'mu':'sum'}).reset_index()
first_stage_production_boost.columns = ['target_id', 'sum_mu']
first_stage_production_boost['original_boost'] = first_stage_production_boost['sum_mu']*p/(1-alpha_p)

outdegree_data = outdegree_data.merge(first_stage_production_boost[['target_id', 'sum_mu']], how = 'left')

outdegree_data['diffusion_boost'] = outdegree_data['sum_mu']*d*p/(1-alpha_p)/(1-alpha_d)

second_stage_production_boost = outdegree_data.groupby(['target_id']).agg({'diffusion_boost':'sum'}).reset_index()


# In[30]:


print(first_stage_production_boost['original_boost'].sum()*(1-alpha_p))
print(first_stage_production_boost['original_boost'].sum())
print(second_stage_production_boost['diffusion_boost'].sum())


# # Optimization example --  increasing nudge motivation

# In[31]:


provider_data = pd.read_csv('social_nudge_network_provider_data.csv')
indegree_data = pd.read_csv('social_nudge_network_indegree_data.csv')
outdegree_data = pd.read_csv('social_nudge_network_outdegree_data.csv')


# In[32]:


# p = *** # uncovered due to data privacy issue
alpha_p = 0.6344675898 
# d = *** # uncovered due to data privacy issue
alpha_d = 0.3749880798 


# In[33]:


indegree_data['mu_source_follow_cnt'] = (indegree_data['source_follow_cnt'] > 332).astype('int')
indegree_data['mu_source_fans_cnt'] = (indegree_data['source_fans_cnt'] > 8).astype('int')
indegree_data = indegree_data.merge(provider_data[['target_id', 'mu_target_daily_upload_cnt']], how = 'left')
indegree_data.fillna(0, inplace=True)


# In[34]:


indegree_data['mu'] = 1/(1+np.exp(-(-9.9943001405 + indegree_data['is_bi_follow']*1.0308743231                                     + indegree_data['mu_source_follow_cnt']*(-0.8518251036) +                                     indegree_data['mu_source_fans_cnt']*1.4398194834 +                                     indegree_data['mu_target_daily_upload_cnt']*(-0.3976876693)))) 


# In[35]:


outdegree_data['diffusion_boost_a_nudge'] = d*p/(1-alpha_p)/(1-alpha_d)
indegree_data['original_boost_a_nudge'] = p/(1-alpha_p)
second_stage_diffusion = outdegree_data.groupby(['target_id']).agg({'diffusion_boost_a_nudge':'sum'}).reset_index()


# In[36]:


indegree_data = indegree_data.merge(second_stage_diffusion, how = 'left')


# In[37]:


delta_mu = 0.1
indegree_data['additional_boost'] = delta_mu * indegree_data['mu'] * (indegree_data['original_boost_a_nudge']                                                                     +indegree_data['diffusion_boost_a_nudge'])

n = len(indegree_data)


i_list = []
relative_improvement_list = []
for i in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30]:
    K = int(n*i/100)
    optimal_strategy = indegree_data.sort_values(by = ['additional_boost'], ascending  = False).head(K)
    random_strategy = indegree_data.sample(K)
    
    rel_imp = (optimal_strategy['additional_boost'].sum()-random_strategy['additional_boost'].sum())/random_strategy['additional_boost'].sum()
    
    print(i, optimal_strategy['additional_boost'].sum(), random_strategy['additional_boost'].sum(), rel_imp)
    
    i_list.append(i)
    relative_improvement_list.append(rel_imp)


# In[38]:


# plot
X_0 = i_list
Y_0 = relative_improvement_list


# In[39]:


plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.figure(figsize=(16, 10), dpi=80, edgecolor='black') 
plt.gca().spines['bottom'].set_color('grey')
plt.gca().spines['top'].set_color('grey') 
plt.gca().spines['right'].set_color('grey')
plt.gca().spines['left'].set_color('grey')

for i in range(20):
    plt.plot(X_0[i], round(Y_0[i]*100), color = 'navy', marker = 'o', markersize = 12, alpha = 1)  
    plt.xticks(np.arange(min(X_0), max(X_0)+1, 1.0))
    plt.xlabel('Percentage of the Total Number of Edges that Receive Pushes (%), i.e., |K|', fontsize=26)
    plt.ylabel('Relative Improvement (%)', fontsize=26)
    plt.grid(True, linestyle = "-", color = "grey", linewidth = "0.5", alpha = 0.5)
    # plt.savefig('Rel_improvement.png', bbox_inches='tight')


# In[6]:


#
font = {'family': 'Times New Roman', 'weight':'normal', 'size':26}
X_0 = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30]
Y_0 = [5.633530, 5.284508, 5.085645, 4.904212, 4.495706, 4.045889, 3.688940, 3.408085, 
       3.172831, 2.974058, 2.657306, 2.424436, 2.244427, 2.091777, 1.955749,
       1.828393, 1.702475, 1.574598, 1.458981, 1.354918]
    
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.figure(figsize=(16, 10), dpi=80, edgecolor='black') 
plt.gca().spines['bottom'].set_color('grey')
plt.gca().spines['top'].set_color('grey') 
plt.gca().spines['right'].set_color('grey')
plt.gca().spines['left'].set_color('grey')

for i in range(20):
    plt.plot(X_0[i], round(Y_0[i]*100), color = 'navy', marker = 'o', markersize = 12, alpha = 1)  
    plt.xticks(np.arange(min(X_0), max(X_0)+1, 1.0))
    plt.xlabel('Percentage of the Total Number of Edges that Receive Pushes (%), i.e., |K|', font)
    plt.ylabel('Relative Improvement (%)', font)
    plt.grid(True, linestyle = "-", color = "grey", linewidth = "0.5", alpha = 0.5)
    #plt.savefig('Rel_improvement.png', bbox_inches='tight')


# # Optimization example -- recommendating new friends

# In[40]:


# p = *** # uncovered due to data privacy issue
alpha_p = 0.6344675898 
# d = *** # uncovered due to data privacy issue
alpha_d = 0.3749880798 


# In[41]:


provider_data = pd.read_csv('social_nudge_network_provider_data.csv')
outdegree_data = pd.read_csv('social_nudge_network_outdegree_data.csv')


# In[42]:


temp = outdegree_data.groupby(['target_id']).agg({'second_stage_target_id':'count'}).reset_index()
temp.columns = ['target_id', 'second_stage_target_id_cnt']
new_friends_data = provider_data.merge(temp, how = 'left')
new_friends_data.fillna(0, inplace=True)


# In[43]:


new_friends_data['mu'] = 1/(1+np.exp(-(-9.9943001405 + new_friends_data['mu_target_daily_upload_cnt']*(-0.3976876693)))) 
new_friends_data['diffusion_prodution_boost'] = new_friends_data['second_stage_target_id_cnt']*new_friends_data['mu']*d*p/(1-alpha_d)/(1-alpha_p)
new_friends_data['direct_prodution_boost'] = new_friends_data['mu']*p/(1-alpha_p)
new_friends_data['SNI'] = new_friends_data['direct_prodution_boost']+new_friends_data['diffusion_prodution_boost']


# In[45]:


m_list = []
relative_improvement_list = []
for m in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30]:
    c = np.random.uniform(low=0.0, high=1.0, size=m)
    c = -np.sort(-c)
    
    opt_total_boost = 0
    rand_total_boost = 0

    for uid in range(1000):
        temp1 = new_friends_data.sample(100000, random_state = np.random.randint(100))
        temp1['new_user_id'] = uid
        temp1['SNI_rank'] = temp1['SNI'].rank(ascending=False, method='first')
        temp1.sort_values(by = 'SNI', ascending=False, inplace = True)
        temp1_opt = temp1[temp1['SNI_rank'] <= m]
        temp1_opt['c'] = c
        # temp1_opt['additional_boost'] = temp1_opt['c']*temp1_opt['SNI']
        temp1_opt['additional_boost'] = temp1_opt['SNI']
        opt_total_boost = opt_total_boost + temp1_opt['additional_boost'].sum()

        temp1_rand = temp1.sample(m, random_state = np.random.randint(100))
        temp1_rand['c'] = c
        # temp1_rand['additional_boost'] = temp1_rand['c']*temp1_rand['SNI']
        temp1_rand['additional_boost'] = temp1_rand['SNI']
        rand_total_boost = rand_total_boost + temp1_rand['additional_boost'].sum()

    m_list.append(m)
    relative_improvement_list.append((opt_total_boost-rand_total_boost)/rand_total_boost)
    print(m, (opt_total_boost-rand_total_boost)/rand_total_boost)
print(m_list)
print(relative_improvement_list)


# In[46]:


# plot
X_0 = m_list
Y_0 = relative_improvement_list


# In[47]:


plt.rcParams['savefig.facecolor'] = 'white'
plt.figure(figsize=(16, 10), dpi=80, edgecolor='black') 
plt.gca().spines['bottom'].set_color('grey')
plt.gca().spines['top'].set_color('grey') 
plt.gca().spines['right'].set_color('grey')
plt.gca().spines['left'].set_color('grey')

for i in range(len(X_0)):
    plt.plot(X_0[i], Y_0[i]*100, color = 'navy', marker = 'o', markersize = 12, alpha = 1)  
    plt.xticks(np.arange(min(X_0), max(X_0)+1, 1.0))
    plt.xlabel('Length of the Provider List Selected for a New User, i.e., m', fontsize=26)
    plt.ylabel('Relative Improvement (%)', fontsize=26)
    plt.grid(True, linestyle = "-", color = "grey", linewidth = "0.5", alpha = 0.5)
    # plt.savefig('Rel_improvement_new_friends.png', bbox_inches='tight')

