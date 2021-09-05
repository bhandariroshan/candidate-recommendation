#!/usr/bin/env python
# coding: utf-8

# In[30]:


import matplotlib.pyplot as plt
import pandas as pd 
plt.rcParams.update({'font.size': 14})

langs = ['C', 'Python', 'Javascript', 'C++', 'Java', 'Ruby', 'Go', 'PhP']
count = [11583953,9262434,7948270,6464431,4646971, 3911434, 3346530, 2570913]

# create Pandas dataframe from two lists
df = pd.DataFrame({"Language":langs, "#Commits":count})
df_sorted = df.sort_values('#Commits')

plt.figure(figsize=(10,6))
# make bar plot with matplotlib
plt.bar('Language', '#Commits',data=df_sorted)
plt.xlabel("Language", size=15)
plt.ylabel("#Commits", size=15)
plt.title("Language Vs Commit Count", size=15)


# In[16]:


# language: javascript    count: 30580
# language: python        count: 15325
# language: html  count: 10852
# language: ruby  count: 9858
# language: java  count: 7596
# language: c++   count: 6048
# language: c     count: 5221
# language: css   count: 4599
# language: c#    count: 3995
# language: jupyter notebook      count: 3322

langs = ['Javascript', 'Python', 'Ruby', 'Java', 'C++', 'C', 'C#']
count = [30580,15325,9858,7596,6048, 5221, 3995]

# create Pandas dataframe from two lists
df = pd.DataFrame({"Language":langs, "# Projects":count})
df_sorted = df.sort_values('# Projects', ascending=False)

plt.figure(figsize=(10,6))
# make bar plot with matplotlib
plt.bar('Language', '# Projects',data=df_sorted)
plt.xlabel("Language", size=14)
plt.ylabel("# Projects", size=14)
plt.title("Language Vs Project Count", size=14)


# In[29]:


langs = ['Javascript', 'Python', 'Ruby', 'Java', 'C++', 'C', 'C#']
count = [30580,15325,9858,7596,6048, 5221, 3995]

# create Pandas dataframe from two lists
df = pd.DataFrame({"Language":langs, "# Projects":count})
df_sorted = df.sort_values('# Projects')

plt.figure(figsize=(10,6))
# make bar plot with matplotlib
plt.bar('Language', '# Projects',data=df_sorted)
plt.xlabel("Language", size=15)
plt.ylabel("#Project", size=15)
plt.title("Language Vs Project Count", size=15)


# In[28]:


# language: javascript    count: 60166
# language: python        count: 30126
# language: html  count: 21303
# language: ruby  count: 19456
# language: java  count: 14748
# language: c++   count: 11793
# language: c     count: 10084
# language: css   count: 9061
# language: c#    count: 7781
# language: jupyter notebook      count: 6595

plt.rcParams.update({'font.size': 15})
langs = ['Javascript', 'Python', 'Ruby', 'Java', 'C++', 'C', 'C#']
count = [60166,30126,19456,14748,11793, 10084, 7781]

# create Pandas dataframe from two lists
df = pd.DataFrame({"Language":langs, "# Developers":count})
df_sorted = df.sort_values('# Developers')

plt.figure(figsize=(10,6))
# make bar plot with matplotlib
plt.bar('Language', '# Developers',data=df_sorted)
plt.xlabel("Language", size=15)
plt.ylabel("# Developers", size=15)
plt.title("Language Vs Developer Count", size=15)


# In[27]:


# Minimum degree of a graph is:  1
# Maximum degree of a graph is:  10436
# Number of Nodes 446635
# Number of edges 766852
# Average Degree of the graph is:  1.7169545602113583
# Nodes in strongest connected component 33137
# Number of user nodes:  309105
# Number of project nodes:  137530

plt.rcParams.update({'font.size': 14})
langs = ['Max. Deg', '# Nodes', '# Edges', '# Dev Nodes', '# Repo. Nodes']
count = [10436,446635,766852,309105,137530]

# create Pandas dataframe from two lists
df = pd.DataFrame({"Feature":langs, "Count":count})
df_sorted = df.sort_values('Count')

plt.figure(figsize=(10,6))
# make bar plot with matplotlib
plt.bar('Feature', 'Count',data=df_sorted)
plt.xlabel("Feature", size=14)
plt.ylabel("Count", size=14)
plt.title("Feature Vs Count", size=14)


# In[ ]:




