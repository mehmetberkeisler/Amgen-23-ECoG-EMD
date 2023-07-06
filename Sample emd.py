#!/usr/bin/env python
# coding: utf-8

# 4 different methods are implemented.
# 1-EMD
# 2-EEMD
# 3-CEEMDAN
# 4-WMD
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import PyEMD
from PyEMD import EMD, EEMD, CEEMDAN
import scipy


# In[2]:


def gaussian_wave(mu, sigma, x):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


# Wave1
mu1 = 1  # mean (time point)
sigma1 = 1  # sd (duration=2*sigma ms)

# Wave2 
mu2 = 0.6  # mean (time point)
sigma2 = 0.1  # sd (duration=2*sigma ms)

# Wave3
mu3 = 1.4  # mean (time point)
sigma3 = 0.1  # sd (duration=2*sigma ms)



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

y1 = y1 - 4
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


# In[ ]:





# In[3]:


#EMD


# In[ ]:





# In[4]:


emd = EMD()
imfs = emd(y_sum)

# find the IMF with the largest correlation
correlations = []
for i, imf in enumerate(imfs):
    correlation = np.corrcoef(imf, y_truth)[0, 1]
    correlations.append(correlation)

# index of the IMF with the largest correlation
max_correlation_index = np.argmax(correlations)

print("IMF with the largest correlation is imfs[",max_correlation_index+1,"]")

# plot the IMFs
plt.figure(figsize=(16, 10))
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs), 1, i+1)
    plt.plot(x, imf, label='IMF {}'.format(i+1))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Intrinsic Mode Function (IMF {})'.format(i+1))
    plt.legend()

    # correlation coefficient
    correlation = correlations[i]
    text = 'Correlation: {:.2f}'.format(correlation)

    # text annotation
    plt.text(0.1, 0.9, text, size=12, color='blue', ha='right', va='top', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()



# In[5]:


eemd = EEMD()
imfs = eemd(y_sum)

# find the IMF with the largest correlation
correlations = []
for i, imf in enumerate(imfs):
    correlation = np.corrcoef(imf, y_truth)[0, 1]
    correlations.append(correlation)

# index of the IMF with the largest correlation
max_correlation_index = np.argmax(correlations)

print("IMF with the largest correlation is imfs[",max_correlation_index+1,"]")

# plot the IMFs
plt.figure(figsize=(16, 10))
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs), 1, i+1)
    plt.plot(x, imf, label='IMF {}'.format(i+1))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Intrinsic Mode Function (IMF {})'.format(i+1))
    plt.legend()

    # correlation coefficient
    correlation = correlations[i]
    text = 'Correlation: {:.2f}'.format(correlation)

    # text annotation
    plt.text(0.1, 0.9, text, size=12, color='blue', ha='right', va='top', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()


# In[6]:


#CEEMDAN 


# In[7]:


ceemdan = CEEMDAN()
imfs = ceemdan(y_sum)

# find the IMF with the largest correlation
correlations = []
for i, imf in enumerate(imfs):
    correlation = np.corrcoef(imf, y_truth)[0, 1]
    correlations.append(correlation)

# index of the IMF with the largest correlation
max_correlation_index = np.argmax(correlations)
print(correlations[max_correlation_index])

print("IMF with the largest correlation is imfs[",max_correlation_index+1,"]")

# plot the IMFs
plt.figure(figsize=(16, 10))
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs), 1, i+1)
    plt.plot(x, imf, label='IMF {}'.format(i+1))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Intrinsic Mode Function (IMF {})'.format(i+1))
    plt.legend()

    # correlation coefficient
    correlation = correlations[i]
    text = 'Correlation: {:.2f}'.format(correlation)

    # text annotation
    plt.text(0.1, 0.9, text, size=12, color='blue', ha='right', va='top', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()


# In[ ]:





# In[8]:





# In[ ]:





# In[ ]:


#variational mode decomposition


# In[10]:


from vmdpy import VMD


# In[39]:


#. some sample parameters for VMD  
alpha = 5000      # moderate bandwidth constraint  
tau = 0           # noise-tolerance (no strict fidelity enforcement)  
K = 4              # number of modes  
DC = 0             # no DC part imposed  
init = 1           # initialize omegas uniformly  
tol = 1e-7


# In[40]:


u, u_hat, omega = VMD(y_sum, alpha, tau, K, DC, init, tol)


# In[ ]:


for i in range(K):   
    plt.plot(u[i])
    plt.figure()
    


# 
# The VMD function gives 3 output variables. u contains the decomposed signals, omega contains the frequency information. u_hat is an array that contains complex values which is used in computing the omega.
# 

# In[54]:


# find the IMF with the largest correlation
correlations = []
for i, imf in enumerate(u):
    correlation = np.corrcoef(u[i], y_truth)[0, 1]
    correlations.append(correlation)

# index of the IMF with the largest correlation
max_correlation_index = np.argmax(correlations)
print(correlations[max_correlation_index])

print("WMD with the largest correlation is wmd[", max_correlation_index+1, "]")

# plot the IMFs
plt.figure(figsize=(16, 10))

for i in range(K):   
    plt.figure()
    plt.plot(x, u[i])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('WMD (WMD {})'.format(i+1))

    # correlation coefficient
    correlation = correlations[i]
    text = 'Correlation: {:.2f}'.format(correlation)

    # text annotation
    plt.text(0.2, 0.9, text, size=9, color='blue', ha='right', va='top', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()


# In[ ]:


#Idea: Compare best performed IMF's corr values of EMD-EEMD-CEEMDAN to see which one is the best


# In[ ]:





# In[ ]:





# In[ ]:


#there are 2 approaches: 1.extrema points(mu values) 2.correlation
'''
import peakutils
maxima_indices = peakutils.indexes(imfs[-1], thres=0.5, min_dist=10)
print("The mean of the slow wave is", maxima_indices)
maxima_indices = peakutils.indexes(imfs[-2], thres=0.5, min_dist=10)
print("The mean of the fast waves are", maxima_indices)
'''

