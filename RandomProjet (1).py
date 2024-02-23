#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.misc import derivative
from scipy.stats import norm
import seaborn as sns


# In[2]:


dfOriginal=pd.read_csv("D:/GUC/Sem5/(NETW504) Random Signals and Noise (466)/Project/UNR-IDD.csv")
df=pd.read_csv("D:/GUC/Sem5/(NETW504) Random Signals and Noise (466)/Project/UNR-IDD.csv")


# In[3]:


df.columns=df.columns.str.lower()
df.columns = [c.replace(' ', '_') for c in df.columns]
df.columns = [c.replace('(', '') for c in df.columns]
df.columns = [c.replace(')', '') for c in df.columns]
df.head()


# In[4]:


columns=df.columns
print(columns)


# In[5]:


DataTypes=df.dtypes
print(DataTypes)


# In[6]:


df.isnull().any()


# In[7]:


for col in df.columns:
    temp=df[col]
    for i,j in temp.iteritems():
        if(j==np.inf)|(j==-np.inf):
            print(True)
        


# In[8]:


for col in df.columns:
    print("The number of categories in: ",col,"is equal to ",len(df[col].unique()))


# In[9]:


df.max()


# In[10]:


df.min()


# In[11]:


df.mean()


# In[12]:


df.var()


# In[13]:


dfSorted=pd.read_csv("D:/GUC/Sem5/(NETW504) Random Signals and Noise (466)/Project/UNR-IDD.csv")
dfSorted = dfSorted.apply(lambda x: x.sort_values().values)


# In[14]:


df1=dfSorted.iloc[:9353,:]
df2=dfSorted.iloc[9353:9352*2+1,:]
df3=dfSorted.iloc[9352*2+1:9352*3+1,:]
df4=dfSorted.iloc[9352*3+1:,:]


# In[15]:


df1.head(50)


# In[16]:


max1 = df1.max()
min1 = df1.min()
mean1 = df1.mean()
var1 = df1.var()


# In[17]:


max2 = df2.max()
min2 = df2.min()
mean2 = df2.mean()
var2 = df2.var()


# In[18]:


max3 = df3.max()
min3 = df3.min()
mean3 = df3.mean()
var3 = df3.var()


# In[19]:


max4 = df4.max()
min4 = df4.min()
mean4 = df4.mean()
var4 = df4.var()


# In[20]:


df.label.unique()


# In[21]:


TCP_SYN=np.zeros((37411,1))
Blackhole=np.zeros((37411,1))
Diversion=np.zeros((37411,1))
Overflow=np.zeros((37411,1))
Normal=np.zeros((37411,1))
PortScan=np.zeros((37411,1))


# In[22]:


counter=0
for data in df.label:
    if data=="TCP-SYN":
        TCP_SYN[counter]=1
    elif data=="Blackhole":
        Blackhole[counter]=1
    elif data=="Diversion":
        Diversion[counter]=1
    elif data=="Overflow":
        Overflow[counter]=1
    elif data=="Normal":
        Normal[counter]=1
    elif data=="PortScan":
        PortScan[counter]=1
    counter+=1      


# In[23]:


df["TCP-SYN"]=TCP_SYN.astype(np.int64)
df["Blackhole"]=Blackhole.astype(np.int64)
df["Diversion"]=Diversion.astype(np.int64)
df["Overflow"]=Overflow.astype(np.int64)
df["Normal"]=Normal.astype(np.int64)
df["PortScan"]=PortScan.astype(np.int64)


# In[24]:


df.head()


# In[25]:


print(df.Blackhole.unique())
print(df.Diversion.unique())
print(df.Overflow.unique())
print(df.Normal.unique())
print(df.PortScan.unique())


# In[26]:


df.dtypes


# In[ ]:





# In[27]:


pmfIndex=[0,1,7,8,9,10,15,16,17,18,19,20,26,27,31,32,33]
counter=1
for index in pmfIndex:
    field1 = dfOriginal[dfOriginal.columns[index]].unique()
    prob1 = [0] * len(field1)
    for i in range(len(field1)):
        for data in dfOriginal[dfOriginal.columns[index]]:
            if(data == field1[i]):
                prob1[i] += 1

    for y in range(len(prob1)):
        prob1[y] = prob1[y]/len(dfOriginal)
    plt.figure(figsize=(15,100))
    plt.subplot(17, 1, counter)
    plt.xticks(rotation=90)
    plt.title(dfOriginal.columns[index])
    plt.stem(field1, prob1)
    plt.show()
    counter+=1


# In[28]:


pdfIndex=[2,3,4,5,6,11,12,13,14,21,22,23,24,25,28,29,30]
counter=1
for index in pdfIndex:
    field7=dfOriginal[dfOriginal.columns[index]].sort_values()
    x,y= np.histogram(field7,bins=500,density=True)
#     mean7=np.mean(field7)
#     std7=np.std(field7)
#     pdf7= stats.norm.pdf(field7.sort_values(), mean7, std7)
    plt.figure(figsize=(15,100))
    plt.subplot(17, 1, counter)
    plt.xticks(rotation=90)
    plt.title(dfOriginal.columns[index])
    plt.plot(y[:-1],x)
    counter+=1


# In[29]:


counter = 1
for col in dfOriginal.columns:
    field7=dfOriginal[col].sort_values()
    if field7.dtype=="int64":
        mean7=np.mean(field7)
        std7=np.std(field7)
        x,y= np.histogram(field7,bins=500)
        plt.figure(figsize=(15,200))
        plt.subplot(40, 1, counter)
        plt.plot(y[:-1],np.cumsum(x/len(df)))
        plt.xticks(rotation=90)
        plt.title(col)
        plt.show()
        counter+=1


# In[30]:


dfa = dfOriginal.loc[ :9080 , :] # TCP-SYN
dfb = dfOriginal.loc[9081:17500 , :] # Blackhole
dfc = dfOriginal.loc[17501 : 23115, :] # Diversion
dfd = dfOriginal.loc[23116 : 24137, :] # Overflow
dfe = dfOriginal.loc[24138 :27910 , :] # Normal
dff = dfOriginal.loc[27911 : , :] # PortScann


# In[31]:


dfa.columns=dfa.columns.str.lower()
dfa.columns = [c.replace(' ', '_') for c in dfa.columns]
dfa.columns = [c.replace('(', '') for c in dfa.columns]
dfa.columns = [c.replace(')', '') for c in dfa.columns]
dfb.columns=dfb.columns.str.lower()
dfb.columns = [c.replace(' ', '_') for c in dfb.columns]
dfb.columns = [c.replace('(', '') for c in dfb.columns]
dfb.columns = [c.replace(')', '') for c in dfb.columns]
dfc.columns=dfa.columns.str.lower()
dfc.columns = [c.replace(' ', '_') for c in dfc.columns]
dfc.columns = [c.replace('(', '') for c in dfc.columns]
dfc.columns = [c.replace(')', '') for c in dfc.columns]
dfd.columns=dfa.columns.str.lower()
dfd.columns = [c.replace(' ', '_') for c in dfd.columns]
dfd.columns = [c.replace('(', '') for c in dfd.columns]
dfd.columns = [c.replace(')', '') for c in dfd.columns]
dfe.columns=dfa.columns.str.lower()
dfe.columns = [c.replace(' ', '_') for c in dfe.columns]
dfe.columns = [c.replace('(', '') for c in dfe.columns]
dfe.columns = [c.replace(')', '') for c in dfe.columns]
dff.columns=dfa.columns.str.lower()
dff.columns = [c.replace(' ', '_') for c in dff.columns]
dff.columns = [c.replace('(', '') for c in dff.columns]
dff.columns = [c.replace(')', '') for c in dff.columns]


# In[32]:


dfa["label"].unique()


# In[33]:


pmfIndex=[0,1,7,8,9,10,15,16,17,18,19,20,26,27,31,32,33]  #given TCP-SYN
counter=1
for index in pmfIndex:
    field1 = dfOriginal[dfOriginal.columns[index]].unique()
    prob1 = [0] * len(field1)
    for i in range(len(field1)):
        for data in dfOriginal[dfOriginal.columns[index]]:
            if(data == field1[i]):
                prob1[i] += 1

    for y in range(len(prob1)):
        prob1[y] = prob1[y]/len(dfOriginal)

    field2 = dfa[dfa.columns[index]].unique()
    prob2 = [0] * len(field2)
    for i in range(len(field2)):
        for data in dfa[dfa.columns[index]]:
            if(data == field2[i]):
                prob2[i] += 1

    for y in range(len(prob2)):
        prob2[y] = prob2[y]/len(dfa)

    plt.figure(figsize=(15,100))
    plt.subplot(17, 1, counter)
    plt.xticks(rotation=90)
    plt.title(dfOriginal.columns[index])
    plt.stem(field1, prob1, 'b')
    plt.stem(field2, prob2,'r')
    plt.show()
    counter+=1


# In[34]:


pdfIndex=[2,3,4,5,6,11,12,13,14,21,22,23,24,25,28,29,30]
counter=1
for index in pdfIndex:
    field7=dfOriginal[dfOriginal.columns[index]].sort_values()
    mean7=np.mean(field7)
    std7=np.std(field7)
    pdf7= stats.norm.pdf(field7.sort_values(), mean7, std7)
    field8=dfa[dfa.columns[index]].sort_values()
    mean8=np.mean(field8)
    std8=np.std(field8)
    pdf8= stats.norm.pdf(field8.sort_values(), mean8, std8)
    plt.figure(figsize=(15,100))
    plt.subplot(17, 1, counter)
    plt.xticks(rotation=90)
    plt.title(dfOriginal.columns[index])
    plt.plot(field7, pdf7, 'b')
    plt.plot(field8, pdf8, 'r')
    counter+=1


# In[35]:


field7=df["sent_packets"].sort_values()
mean7=np.mean(field7)
std7=np.std(field7)
pdf7= stats.norm.pdf(field7.sort_values(), mean7, std7)
plt.scatter(field7, pdf7)

field7=df["received_packets"].sort_values()
mean7=np.mean(field7)
std7=np.std(field7)
pdf7= stats.norm.pdf(field7.sort_values(), mean7, std7)
plt.scatter(field7, pdf7)


# In[36]:


sns.jointplot(data = df, x = df.sent_packets, y = df.received_packets )
#sns.scatterplot(df.sent_packets,df.received_packets)


# In[37]:


sns.jointplot(data = dfa, x = dfa.sent_packets, y = dfa.received_packets )
#sns.scatterplot(dfa.sent_packets,dfa.received_packets)  #given TCP-SYN attack


# In[38]:


dfOriginal.info()


# In[39]:


dfa.info()


# In[40]:


dfOriginal.corr()


# In[41]:


dfa.corr()


# In[42]:


df.corr().style.background_gradient(cmap='coolwarm')


# In[43]:


dfa.corr().var()


# In[44]:


dfa.corr().mean()


# In[45]:


df.corr().style.background_gradient(cmap='coolwarm')


# In[46]:


field7=dfa["received_packets"].sort_values()
mean7=np.mean(field7)
std7=np.std(field7)
pdf7= stats.norm.pdf(field7.sort_values(), mean7, std7)
plt.xticks(rotation=90)
plt.plot(field7, pdf7,'b')
field8=dfa["sent_packets"].sort_values()
mean8=np.mean(field8)
std8=np.std(field8)
pdf8= stats.norm.pdf(field8.sort_values(), mean8, std8)
plt.xticks(rotation=90)
plt.plot(field8, pdf8,'r')


# In[ ]:





# In[47]:


#TASK2


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


train,test=train_test_split(df, test_size=0.2)


# In[50]:


train.shape


# In[51]:


test.shape


# In[52]:


(29928/(29928+7483))*100


# In[53]:


(7483/(29928+7483))*100


# In[54]:


test.head(50)


# In[55]:


# %matplotlib inline

# import warnings
# import numpy as np
# import pandas as pd
# import scipy.stats as st
# import statsmodels.api as sm
# from scipy.stats._continuous_distns import _distn_names
# import matplotlib
# import matplotlib.pyplot as plt

# matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
# matplotlib.style.use('ggplot')

# # Create models from data
# def best_fit_distribution(data, bins=500, ax=None):
#     """Model data by finding best fit distribution to data"""
#     # Get histogram of original data
#     y, x = np.histogram(data, bins=bins, density=True)
#     x = (x + np.roll(x, -1))[:-1] / 2.0

#     # Best holders
#     best_distributions = []

#     # Estimate distribution parameters from data
#     for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

#         print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

#         distribution = getattr(st, distribution)

#         # Try to fit the distribution
#         try:
#             # Ignore warnings from data that can't be fit
#             with warnings.catch_warnings():
#                 warnings.filterwarnings('ignore')
                
#                 # fit dist to data
#                 params = distribution.fit(data)

#                 # Separate parts of parameters
#                 arg = params[:-2]
#                 loc = params[-2]
#                 scale = params[-1]
                
#                 # Calculate fitted PDF and error with fit in distribution
#                 pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
#                 sse = np.sum(np.power(y - pdf, 2.0))
                
#                 # if axis pass in add to plot
#                 try:
#                     if ax:
#                         pd.Series(pdf, x).plot(ax=ax)
#                     end
#                 except Exception:
#                     pass

#                 # identify if this distribution is better
#                 best_distributions.append((distribution, params, sse))
        
#         except Exception:
#             pass

    
#     return sorted(best_distributions, key=lambda x:x[2])

# def make_pdf(dist, params, size=10000):
#     """Generate distributions's Probability Distribution Function """

#     # Separate parts of parameters
#     arg = params[:-2]
#     loc = params[-2]
#     scale = params[-1]

#     # Get sane start and end points of distribution
#     start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
#     end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

#     # Build PDF and turn into pandas Series
#     x = np.linspace(start, end, size)
#     y = dist.pdf(x, loc=loc, scale=scale, *arg)
#     pdf = pd.Series(y, x)

#     return pdf
# pdfIndex=[2,3,4,5,6,11,12,13,14,21,22,23,24,25,28,29,30]
# bestFit=[]
# for index in pdfIndex:
#     temp=train[train.columns[index]]
# # Load data from statsmodels datasets
#     data = pd.Series(temp.values.ravel())

# # Plot for comparison
#     plt.figure(figsize=(12,8))
#     ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# # Save plot limits
#     dataYLim = ax.get_ylim()

# # Find best fit distribution
#     best_distibutions = best_fit_distribution(data, 200, ax)
#     best_dist = best_distibutions[0]

# # Update plots
#     ax.set_ylim(dataYLim)
#     ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
#     ax.set_xlabel(u'Temp (°C)')
#     ax.set_ylabel('Frequency')

# # Make PDF with best params 
#     pdf = make_pdf(best_dist[0], best_dist[1])

# # Display
#     plt.figure(figsize=(12,8))
#     ax = pdf.plot(lw=2, label='PDF', legend=True)
#     data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

#     param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
#     param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
#     dist_str = '{}({})'.format(best_dist[0].name, param_str)

#     ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
#     ax.set_xlabel(u'Temp. (°C)')
#     ax.set_ylabel('Frequency')
#     bestFit.append(dist_str)


# In[56]:


train.columns


# In[57]:


type(data)


# In[58]:


type(train.set_index('switch_id'))


# In[ ]:





# In[60]:


type(data)


# In[61]:


train['received_packets']


# In[ ]:





# In[62]:


pdfIndex=[2,3,4,5,6,11,12,13,14,21,22,23,24,25,28,29,30]
counter=1
num=[]
for index in pdfIndex:
     num.append(train.columns[index])


# In[63]:


num


# In[64]:


temp=train[train.columns[index]]
type(temp)


# In[ ]:





# In[ ]:





# In[67]:


trainBestFit


# In[68]:





# In[70]:


Best=pd.read_csv("D:/GUC/Sem5/(NETW504) Random Signals and Noise (466)/Project/my_csv.csv")


# In[ ]:


Best


# In[ ]:


trainAttack=train.loc[train.binary_label=='Attack']


# In[ ]:


trainNormal=train.loc[train.binary_label=='Normal']


# In[ ]:


# %matplotlib inline

# import warnings
# import numpy as np
# import pandas as pd
# import scipy.stats as st
# import statsmodels.api as sm
# from scipy.stats._continuous_distns import _distn_names
# import matplotlib
# import matplotlib.pyplot as plt

# matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
# matplotlib.style.use('ggplot')

# # Create models from data
# def best_fit_distribution(data, bins=500, ax=None):
#     """Model data by finding best fit distribution to data"""
#     # Get histogram of original data
#     y, x = np.histogram(data, bins=bins, density=True)
#     x = (x + np.roll(x, -1))[:-1] / 2.0

#     # Best holders
#     best_distributions = []

#     # Estimate distribution parameters from data
#     for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

#         print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

#         distribution = getattr(st, distribution)

#         # Try to fit the distribution
#         try:
#             # Ignore warnings from data that can't be fit
#             with warnings.catch_warnings():
#                 warnings.filterwarnings('ignore')
                
#                 # fit dist to data
#                 params = distribution.fit(data)

#                 # Separate parts of parameters
#                 arg = params[:-2]
#                 loc = params[-2]
#                 scale = params[-1]
                
#                 # Calculate fitted PDF and error with fit in distribution
#                 pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
#                 sse = np.sum(np.power(y - pdf, 2.0))
                
#                 # if axis pass in add to plot
#                 try:
#                     if ax:
#                         pd.Series(pdf, x).plot(ax=ax)
#                     end
#                 except Exception:
#                     pass

#                 # identify if this distribution is better
#                 best_distributions.append((distribution, params, sse))
        
#         except Exception:
#             pass

    
#     return sorted(best_distributions, key=lambda x:x[2])

# def make_pdf(dist, params, size=10000):
#     """Generate distributions's Probability Distribution Function """

#     # Separate parts of parameters
#     arg = params[:-2]
#     loc = params[-2]
#     scale = params[-1]

#     # Get sane start and end points of distribution
#     start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
#     end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

#     # Build PDF and turn into pandas Series
#     x = np.linspace(start, end, size)
#     y = dist.pdf(x, loc=loc, scale=scale, *arg)
#     pdf = pd.Series(y, x)

#     return pdf
# pdfIndex=[2,3,4,5,6,11,12,13,14,21,22,23,24,25,28,29,30]
# bestFitAttack=[]
# for index in pdfIndex:
#     temp=trainAttack[trainAttack.columns[index]]
# # Load data from statsmodels datasets
#     data = pd.Series(temp.values.ravel())

# # Plot for comparison
#     plt.figure(figsize=(12,8))
#     ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# # Save plot limits
#     dataYLim = ax.get_ylim()

# # Find best fit distribution
#     best_distibutions = best_fit_distribution(data, 200, ax)
#     best_dist = best_distibutions[0]

# # Update plots
#     ax.set_ylim(dataYLim)
#     ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
#     ax.set_xlabel(u'Temp (°C)')
#     ax.set_ylabel('Frequency')

# # Make PDF with best params 
#     pdf = make_pdf(best_dist[0], best_dist[1])

# # Display
#     plt.figure(figsize=(12,8))
#     ax = pdf.plot(lw=2, label='PDF', legend=True)
#     data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

#     param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
#     param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
#     dist_str = '{}({})'.format(best_dist[0].name, param_str)

#     ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
#     ax.set_xlabel(u'Temp. (°C)')
#     ax.set_ylabel('Frequency')
#     bestFitAttack.append(dist_str)


# In[ ]:


trainAttackBestFit=pd.Series(bestFitAttack,index=num)


# In[ ]:


# %matplotlib inline

# import warnings
# import numpy as np
# import pandas as pd
# import scipy.stats as st
# import statsmodels.api as sm
# from scipy.stats._continuous_distns import _distn_names
# import matplotlib
# import matplotlib.pyplot as plt

# matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
# matplotlib.style.use('ggplot')

# # Create models from data
# def best_fit_distribution(data, bins=500, ax=None):
#     """Model data by finding best fit distribution to data"""
#     # Get histogram of original data
#     y, x = np.histogram(data, bins=bins, density=True)
#     x = (x + np.roll(x, -1))[:-1] / 2.0

#     # Best holders
#     best_distributions = []

#     # Estimate distribution parameters from data
#     for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

#         print("{:>3} / {:<3}: {}".format( ii+1, len(_distn_names), distribution ))

#         distribution = getattr(st, distribution)

#         # Try to fit the distribution
#         try:
#             # Ignore warnings from data that can't be fit
#             with warnings.catch_warnings():
#                 warnings.filterwarnings('ignore')
                
#                 # fit dist to data
#                 params = distribution.fit(data)

#                 # Separate parts of parameters
#                 arg = params[:-2]
#                 loc = params[-2]
#                 scale = params[-1]
                
#                 # Calculate fitted PDF and error with fit in distribution
#                 pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
#                 sse = np.sum(np.power(y - pdf, 2.0))
                
#                 # if axis pass in add to plot
#                 try:
#                     if ax:
#                         pd.Series(pdf, x).plot(ax=ax)
#                     end
#                 except Exception:
#                     pass

#                 # identify if this distribution is better
#                 best_distributions.append((distribution, params, sse))
        
#         except Exception:
#             pass

    
#     return sorted(best_distributions, key=lambda x:x[2])

# def make_pdf(dist, params, size=10000):
#     """Generate distributions's Probability Distribution Function """

#     # Separate parts of parameters
#     arg = params[:-2]
#     loc = params[-2]
#     scale = params[-1]

#     # Get sane start and end points of distribution
#     start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
#     end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

#     # Build PDF and turn into pandas Series
#     x = np.linspace(start, end, size)
#     y = dist.pdf(x, loc=loc, scale=scale, *arg)
#     pdf = pd.Series(y, x)

#     return pdf
# pdfIndex=[2,3,4,5,6,11,12,13,14,21,22,23,24,25,28,29,30]
# bestFitNormal=[]
# for index in pdfIndex:
#     temp=trainNormal[trainNormal.columns[index]]
# # Load data from statsmodels datasets
#     data = pd.Series(temp.values.ravel())

# # Plot for comparison
#     plt.figure(figsize=(12,8))
#     ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# # Save plot limits
#     dataYLim = ax.get_ylim()

# # Find best fit distribution
#     best_distibutions = best_fit_distribution(data, 200, ax)
#     best_dist = best_distibutions[0]

# # Update plots
#     ax.set_ylim(dataYLim)
#     ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
#     ax.set_xlabel(u'Temp (°C)')
#     ax.set_ylabel('Frequency')

# # Make PDF with best params 
#     pdf = make_pdf(best_dist[0], best_dist[1])

# # Display
#     plt.figure(figsize=(12,8))
#     ax = pdf.plot(lw=2, label='PDF', legend=True)
#     data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

#     param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
#     param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
#     dist_str = '{}({})'.format(best_dist[0].name, param_str)

#     ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
#     ax.set_xlabel(u'Temp. (°C)')
#     ax.set_ylabel('Frequency')
#     bestFitNormal.append(dist_str)


# In[ ]:


trainNormalBestFit=pd.Series(bestFitNormal,index=num)


# In[ ]:


trainNormalBestFit


# In[ ]:


trainAttackBestFit


# In[ ]:


BestAttack=pd.read_csv("D:/GUC/Sem5/(NETW504) Random Signals and Noise (466)/Project/my_csv1.csv")


# In[ ]:


BestNormal=pd.read_csv("D:/GUC/Sem5/(NETW504) Random Signals and Noise (466)/Project/my_csv1.csv")


# In[ ]:


BestAttack


# In[ ]:


BestNormal


# In[ ]:


Best


# In[ ]:


pmfIndex=[0,1,7,8,9,10,15,16,17,18,19,20,26,27,31,32,33]
pmfField=[]
pmfProb=[]
pmfColName=[]
counter=1
dfOriginal=train
for index in pmfIndex:
    field1 = dfOriginal[dfOriginal.columns[index]].unique()
    prob1 = [0] * len(field1)
    for i in range(len(field1)):
        for data in dfOriginal[dfOriginal.columns[index]]:
            if(data == field1[i]):
                prob1[i] += 1

    for y in range(len(prob1)):
        prob1[y] = prob1[y]/len(dfOriginal)
    pmfColName.append(dfOriginal.columns[index])
    pmfField.append(field1)
    pmfProb.append(prob1)
    counter+=1


# In[ ]:


pmfAll=[]
for index in range(len(pmfField)):
    t1=pmfColName[index]
    t2=pmfField[index]
    t3=pmfProb[index]
    temp=[t1,t2,t3]
    pmfAll.append(temp)


# In[ ]:


pmfIndex=[0,1,7,8,9,10,15,16,17,18,19,20,26,27,31,32,33]
pmfField=[]
pmfProb=[]
pmfColName=[]
counter=1
dfOriginal=trainAttack
for index in pmfIndex:
    field1 = dfOriginal[dfOriginal.columns[index]].unique()
    prob1 = [0] * len(field1)
    for i in range(len(field1)):
        for data in dfOriginal[dfOriginal.columns[index]]:
            if(data == field1[i]):
                prob1[i] += 1

    for y in range(len(prob1)):
        prob1[y] = prob1[y]/len(dfOriginal)
    pmfColName.append(dfOriginal.columns[index])
    pmfField.append(field1)
    pmfProb.append(prob1)
    counter+=1


# In[ ]:


pmfAttack=[]
for index in range(len(pmfField)):
    t1=pmfColName[index]
    t2=pmfField[index]
    t3=pmfProb[index]
    temp=[t1,t2,t3]
    pmfAttack.append(temp)


# In[ ]:


pmfIndex=[0,1,7,8,9,10,15,16,17,18,19,20,26,27,31,32,33]
pmfField=[]
pmfProb=[]
pmfColName=[]
counter=1
dfOriginal=trainNormal
for index in pmfIndex:
    field1 = dfOriginal[dfOriginal.columns[index]].unique()
    prob1 = [0] * len(field1)
    for i in range(len(field1)):
        for data in dfOriginal[dfOriginal.columns[index]]:
            if(data == field1[i]):
                prob1[i] += 1

    for y in range(len(prob1)):
        prob1[y] = prob1[y]/len(dfOriginal)
    pmfColName.append(dfOriginal.columns[index])
    pmfField.append(field1)
    pmfProb.append(prob1)
    counter+=1


# In[ ]:


pmfNormal=[]
for index in range(len(pmfField)):
    t1=pmfColName[index]
    t2=pmfField[index]
    t3=pmfProb[index]
    temp=[t1,t2,t3]
    pmfNormal.append(temp)


# In[ ]:





# In[ ]:


trainNormalBestFit=pd.Series(bestFitNormal,index=num)


# In[ ]:


train


# In[ ]:


BestAttack


# In[ ]:


test


# In[ ]:


predic=test.iloc[0]


# In[ ]:


predic


# In[ ]:





# In[ ]:


BestAttack


# In[ ]:


predic


# In[ ]:


predictionProb=[]
predictionProb.append(pmfAll[0][2][2])
predictionProb.append(pmfAll[1][2][2])


# In[ ]:


from scipy.stats import mielke
mielke.pdf(x=25230955, k=0.97, s=0.36, loc=-315.05, scale=54657.44)


# In[ ]:


pmfAll[1][2][2]


# In[ ]:


predictionProb


# In[ ]:


from scipy.stats import mielke
skewcauchy	a=1.00	loc=9.00	scale=455.89	NaN


# In[ ]:


from scipy.stats import skewcauchy
skewcauchy.pdf(x=947,loc=9,scale=455.89,a = 1)


# In[ ]:


BestAttack


# In[ ]:


temp=BestAttack[0]


# In[ ]:


temp


# In[ ]:


dist=temp[1]
params=temp[2:].dropna()


# In[ ]:


dist


# In[ ]:


params


# In[ ]:


arg=params[:-2]
loc=params[-2]
scale=params[-1]


# In[ ]:


arg


# In[ ]:


loc


# In[ ]:


scale


# In[ ]:





# In[ ]:


import scipy.stats as st
dist=getattr(st,dist)


# In[ ]:


dist.pdf(947,loc=loc,scale=scale,*arg)


# In[ ]:


print(*arg)


# In[ ]:




