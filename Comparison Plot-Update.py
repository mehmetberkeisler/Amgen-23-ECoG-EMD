#!/usr/bin/env python
# coding: utf-8

# 4 different methods are implemented.
# 1-EMD
# 2-EEMD
# 3-CEEMDAN
# 4-WMD
# 5-EWT
# 

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import PyEMD
from PyEMD import EMD, EEMD, CEEMDAN
import scipy
from vmdpy import VMD
import  ewtpy


# In[14]:


def fEMD(y_sum, y_truth):
    emd = EMD()
    imfs = emd(y_sum)

    # find the IMF with the largest correlation
    correlations = []
    for i, imf in enumerate(imfs):
        correlation = np.corrcoef(imf, y_truth)[0, 1]
        correlations.append(correlation)

    # index of the IMF with the largest correlation
    max_correlation_index = np.argmax(correlations)
    return correlations[max_correlation_index], imfs[max_correlation_index]


# In[15]:


def fEEMD(y_sum, y_truth):
    eemd = EEMD()
    imfs = eemd(y_sum)

    # find the IMF with the largest correlation
    correlations = []
    for i, imf in enumerate(imfs):
        correlation = np.corrcoef(imf, y_truth)[0, 1]
        correlations.append(correlation)

    # index of the IMF with the largest correlation
    max_correlation_index = np.argmax(correlations)
    return correlations[max_correlation_index], imfs[max_correlation_index]


# In[16]:


def fCEEMDAN(y_sum, y_truth):
    ceemdan = CEEMDAN()
    imfs = ceemdan(y_sum)

    # find the IMF with the largest correlation
    correlations = []
    for i, imf in enumerate(imfs):
        correlation = np.corrcoef(imf, y_truth)[0, 1]
        correlations.append(correlation)

    # index of the IMF with the largest correlation
    max_correlation_index = np.argmax(correlations)
    return correlations[max_correlation_index], imfs[max_correlation_index]


# In[17]:


def fWMD(y_sum, y_truth):
   #. some sample parameters for VMD  
   alpha = 5000      # moderate bandwidth constraint  
   tau = 0           # noise-tolerance (no strict fidelity enforcement)  
   K = 7              # number of modes  
   DC = 0             # no DC part imposed  
   init = 1           # initialize omegas uniformly  
   tol = 1e-7

   u, u_hat, omega = VMD(y_sum, alpha, tau, K, DC, init, tol)


   # find the IMF with the largest correlation
   correlations = []
   for i in range(K):
       correlation = np.corrcoef(u[i], y_truth)[0, 1]
       correlations.append(correlation)

   # index of the IMF with the largest correlation
   max_correlation_index = np.argmax(correlations)
   return correlations[max_correlation_index], u[max_correlation_index]


# In[18]:


def fEWT(y_sum, y_truth):
    
    N=7
    ewt,  mfb ,boundaries = ewtpy.EWT1D(y_sum, N)
    
    
    # find the IMF with the largest correlation
    correlations = []
    for i in range(ewt.shape[1]):
        correlation = np.corrcoef(ewt[:,i], y_truth)[0, 1]
        correlations.append(correlation)

    # index of the IMF with the largest correlation
    max_correlation_index = np.argmax(correlations)
    return correlations[max_correlation_index] ,ewt[:,max_correlation_index] 


# 
# The VMD function gives 3 output variables. u contains the decomposed signals, omega contains the frequency information. u_hat is an array that contains complex values which is used in computing the omega.
# 
# Put more expl.
# 

# In[19]:


#Idea: Compare best performed IMF's corr values of EMD-EEMD-CEEMDAN to see which one is the best


# In[20]:


def gaussian_wave(mu, sigma, x):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


# In[21]:


def Parameters(mu1,sigma1,mu2,sigma2,mu3,sigma3):

    x = np.linspace(0, 2, num=500)  

    y1 = gaussian_wave(mu1, sigma1, x)
    y2 = gaussian_wave(mu2, sigma2, x)/80
    y3 = gaussian_wave(mu3, sigma3, x)/80


    y_sum = y1 + y2 + y3
    y_truth= y2 + y3

    rescale_factor = 1/y_sum.std()

    y1 *= rescale_factor
    y2 *= rescale_factor
    y3 *= rescale_factor
    y_sum *= rescale_factor
    y_truth *= rescale_factor
    
    y1 = y1
    y_sum = y1 + y2 + y3
    
    #plotting

    plt.plot(x, y_sum, label='The Sum')
    plt.title("Gaussian Waves")
    plt.xlabel('Time(s)')
    plt.ylabel('Power(mV)')

    plt.plot(x,y_truth,label='Ground Truth')

      
    plt.plot(x,y1,label='Slow Drift')

    plt.legend()
    plt.show()
    
    
    return y_sum, y_truth


# In[25]:


# Set the number of random parameter combinations to try
num_combinations = 10

# Perform the parameter sweep
results_emd = []
results_eemd = []
results_ceemdan = []
results_wmd = []
results_ewt = []

results_mu1 = []
results_sigma1 = []
results_mu2 = []
results_sigma2 = []
results_mu3 = []
results_sigma3 = []

best_imf_emd = []
best_imf_eemd = []
best_imf_ceemdan = []
best_imf_wmd = []
best_imf_ewt = []

# Define the parameter ranges
mu1_range = [0.5, 1.5]
sigma1_range = [0.5, 1.5]
mu2_range = [0.4, 0.8]
sigma2_range = [0.05, 0.15]
mu3_range = [1.2, 1.6]
sigma3_range = [0.05, 0.15]

# Perform the parameter sweep
results = []
for _ in range(num_combinations):
    # Randomly sample parameter values within the specified ranges
    mu1 = np.random.uniform(*mu1_range)
    sigma1 = np.random.uniform(*sigma1_range)
    mu2 = np.random.uniform(*mu2_range)
    sigma2 = np.random.uniform(*sigma2_range)
    mu3 = np.random.uniform(*mu3_range)
    sigma3 = np.random.uniform(*sigma3_range)
        
    y_sum, y_truth = Parameters(mu1, sigma1, mu2, sigma2, mu3, sigma3)
    
    result_emd, best_imf_emd = fEMD(y_sum, y_truth)
    result_eemd, best_imf_eemd = fEEMD(y_sum, y_truth)
    result_ceemdan, best_imf_ceemdan = fCEEMDAN(y_sum, y_truth)
    result_wmd, best_imf_wmd = fWMD(y_sum, y_truth)
    result_ewt, best_imf_ewt = fEWT(y_sum, y_truth)    
    
    print("EMD:", result_emd)
    print("EEMD:" ,result_eemd)
    print("CEEMDAN:",result_ceemdan)
    print("WMD: " ,result_wmd)
    print("EWT: " ,result_ewt)
    
    results_emd.append(result_emd)
    results_eemd.append(result_eemd)
    results_ceemdan.append(result_ceemdan)
    results_wmd.append(result_wmd)
    results_ewt.append(result_ewt)
    
    results_mu1.append(mu1)
    results_sigma1.append(sigma1)
    results_mu2.append(mu2)
    results_sigma2.append(sigma2)
    results_mu3.append(mu3)
    results_sigma3.append(sigma3)

    plt.title("Gaussian Waves")
    plt.xlabel('Time(s)')
    plt.ylabel('Power(mV)')

    plt.plot(x,y_truth,label='Ground Truth',linewidth=4, color='black')

    plt.plot(x, best_imf_wmd, label='Best WMD')
    plt.plot(x, best_imf_ewt, label='Best EWT')
    plt.plot(x, best_imf_emd, label='Best EMD')
    plt.plot(x, best_imf_eemd, label='Best EEMD')
    plt.plot(x, best_imf_ceemdan, label='Best CEEMDAN')


    plt.legend()
    plt.show()


# In[27]:


print(np.mean(results_emd))  
print(np.mean(results_eemd)) 
print(np.mean(results_ceemdan)) 
print(np.mean(results_wmd)) 
print(np.mean(results_ewt)) 


# In[28]:


# Assuming your results are stored in lists.
results_all = [results_emd, results_eemd, results_ceemdan, results_wmd, results_ewt]
labels_all = ['EMD', 'EEMD', 'CEEMDAN', 'WMD', 'EWT']

x = np.diff([results_mu2, results_mu3], axis=0) #distance of the fast
x = np.ravel(x)  # Flatten x to a 1D array

y = np.mean( [results_sigma2, results_sigma3], axis=0) # 1/amplitude of the fast
y = np.ravel(y)  # Flatten y to a 1D array

fig, axs = plt.subplots(len(results_all), 2, figsize=(20, 30))

for i, results in enumerate(results_all):
    results = np.ravel(results)  # Flatten results to a 1D array
    h1 = axs[i][0].hist2d(x, results, bins=7, cmap='viridis')
    h2 = axs[i][1].hist2d(y, results, bins=7, cmap='viridis')
    
    fig.colorbar(h1[3], ax=axs[i][0])
    fig.colorbar(h2[3], ax=axs[i][1])
    
    axs[i][0].set_xlabel('Distance of fast waves')
    axs[i][0].set_ylabel('Performance')
    axs[i][0].set_title(labels_all[i])

    axs[i][1].set_xlabel('Flatness of fast waves')
    axs[i][1].set_ylabel('Performance')
    axs[i][1].set_title(labels_all[i])

plt.tight_layout()
plt.show()


# In[29]:


# Assuming your results are stored in lists.
results_all = [results_emd, results_eemd, results_ceemdan, results_wmd, results_ewt]
labels_all = ['EMD', 'EEMD', 'CEEMDAN', 'WMD', 'EWT']

x = results_mu1  #mean of the slow
y = results_sigma1 # 1/amplitude of the slow

fig, axs = plt.subplots(len(results_all), 2, figsize=(20, 30))

for i, results in enumerate(results_all):
    results = np.ravel(results)  # Flatten results to a 1D array
    h1 = axs[i][0].hist2d(x, results, bins=7, cmap='viridis')
    h2 = axs[i][1].hist2d(y, results, bins=7, cmap='viridis')
    
    fig.colorbar(h1[3], ax=axs[i][0])
    fig.colorbar(h2[3], ax=axs[i][1])
    
    axs[i][0].set_xlabel('Mean of the slow drift')
    axs[i][0].set_ylabel('Performance')
    axs[i][0].set_title(labels_all[i])

    axs[i][1].set_xlabel('Flatness of the slow drift')
    axs[i][1].set_ylabel('Performance')
    axs[i][1].set_title(labels_all[i])

plt.tight_layout()
plt.show()


# In[ ]:




