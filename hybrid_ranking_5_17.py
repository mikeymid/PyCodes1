
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import matplotlib.pyplot as plt

os.chdir("C:/Users/MMOHA14/Desktop/Projects/Proj 1 - Soil Nutrient Classification/BreedingSoilSampleData/MNPX")

#from IPython import embed

#embed()


# In[5]:


df1 = pd.read_csv('hav_final.csv')
df2 = pd.read_csv('newry_final.csv')

mean_df1 = df1.groupby(['Hybrid'], as_index=False).mean()
mean_df2 = df2.groupby(['Hybrid'], as_index=False).mean()

merged_df = pd.merge(mean_df1, mean_df2, on='Hybrid', suffixes=['_hav', '_newry'])

cor_results = spearmanr(merged_df.Yield_hav.values, merged_df.Yield_newry.values)

print('---------------------------')
print(cor_results)
print('---------------------------')



# In[4]:

#print(df1.Hybrid.value_counts())
#print('---------------------------')


#print(df1.Hybrid.nunique())
#print('---------------------------')


#print(df2.Hybrid.value_counts())
#print('---------------------------')

#print(df1.Hybrid.nunique())
#print('---------------------------')

print('---------------------------')
print(merged_df.Hybrid.nunique())
print('---------------------------')


# In[ ]:



#df1.Hybrid.value_counts()
#merged_df.sort_values('Yield_hav', ascending = False, inplace = True)
#sub_df = merged_df[:58]
#sub_df
#cor_results = spearmanr(merged_df.Yield_hav.values, merged_df.Yield_newry.values)
#cor_results

# In[7]:

merged_df


# In[11]:

merged_df.to_csv('hybrid_yield.csv')


# In[ ]:



