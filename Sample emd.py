#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import PyEMD
from PyEMD import EMD, CEEMDAN

def gaussian_wave(mu, sigma, x):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


# Wave1
mu1 = 400  # mean
sigma1 = 180  # sd

# Wave2
mu2 = 550  # mean
sigma2 = 40  # sd

# Wave3
mu3 = 250  # mean
sigma3 = 40  # sd



x = np.linspace(0, 1000, num=1000)  

y1 = gaussian_wave(mu1, sigma1, x)
y2 = gaussian_wave(mu2, sigma2, x)/15
y3 = gaussian_wave(mu3, sigma3, x)/15


y_sum = y1 + y2 + y3
rescale_factor = 1/y_sum.std()

y1 *= rescale_factor
y2 *= rescale_factor
y3 *= rescale_factor
y_sum *= rescale_factor

# Plotting

plt.plot(x, y_sum, label='The Sum')
plt.title("Gaussian Waves")

plt.legend()
plt.show()


# In[50]:


sample_rate = 100
seconds = 10
num_samples = sample_rate*seconds


# In[51]:


# Perform CEEMDAN decomposition
ceemdan = CEEMDAN()
imfs = ceemdan(y_sum)

# Plot the decomposed IMFs
plt.figure(figsize=(8, 6))
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs), 1, i+1)
    plt.plot(x, imf, label='IMF {}'.format(i+1))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Intrinsic Mode Function (IMF {})'.format(i+1))
    plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


#there are 2 approaches: 1.extrema points(mu values) 2.correlation


# In[ ]:





# In[62]:


import scipy


# In[66]:


scipy.signal.find_peaks(imfs[5])


# In[68]:


scipy.signal.find_peaks(imfs[4])


# In[ ]:





# In[ ]:




