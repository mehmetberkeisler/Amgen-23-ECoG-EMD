#!/usr/bin/env python
# coding: utf-8

# 4 different methods are implemented.
# 1-EMD
# 2-EEMD
# 3-CEEMDAN
# 4-WMD
# 5-EWT
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import PyEMD
from PyEMD import EMD, EEMD, CEEMDAN
import scipy
from vmdpy import VMD
import  ewtpy
import peakutils
from scipy.signal import find_peaks


# In[2]:


def fEMD(y_sum, y_truth):
    emd = EMD()
    imfs = emd(y_sum)

    # find the IMF with the largest correlation
    distances = []
    for i, imf in enumerate(imfs):
        peaks_imf, _ = find_peaks(imf)
        peaks_truth, _ = find_peaks(y_truth)
        distance = calculate_similarity(peaks_imf, peaks_truth)
        distances.append(distance)

    # index of the IMF with the minimal distance
    min_distance_index = np.argmin(distances)
    return distances[min_distance_index]


# In[3]:


def fEEMD(y_sum, y_truth):
    eemd = EEMD()
    imfs = eemd(y_sum)

    # find the IMF with the largest correlation
    distances = []
    for i, imf in enumerate(imfs):
        peaks_imf, _ = find_peaks(imf)
        peaks_truth, _ = find_peaks(y_truth)
        distance = calculate_similarity(peaks_imf, peaks_truth)
        distances.append(distance)

    # index of the IMF with the minimal distance
    min_distance_index = np.argmin(distances)
    return distances[min_distance_index]


# In[4]:


def fCEEMDAN(y_sum, y_truth):
    ceemdan = CEEMDAN()
    imfs = ceemdan(y_sum)
    
    # find the IMF with the largest correlation
    distances = []
    for i, imf in enumerate(imfs):
        peaks_imf, _ = find_peaks(imf)
        peaks_truth, _ = find_peaks(y_truth)
        distance = calculate_similarity(peaks_imf, peaks_truth)
        distances.append(distance)

    # index of the IMF with the minimal distance
    min_distance_index = np.argmin(distances)
    return distances[min_distance_index]


# In[5]:


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
   distances = []
   for i in range(K):
       peaks_imf, _ = find_peaks(u[i])
       peaks_truth, _ = find_peaks(y_truth)
       distance = calculate_similarity(peaks_imf, peaks_truth)
       distances.append(distance)

   # index of the IMF with the minimal distance
   min_distance_index = np.argmin(distances)
   return distances[min_distance_index]


# In[6]:


def fEWT(y_sum, y_truth):
    
    N=7
    ewt,  mfb ,boundaries = ewtpy.EWT1D(y_sum, N)
    
    
    # find the IMF with the largest correlation
    distances = []
    for i in range(ewt.shape[1]):
        peaks_imf, _ = find_peaks(ewt[:,i])
        peaks_truth, _ = find_peaks(y_truth)
        distance = calculate_similarity(peaks_imf, peaks_truth)
        distances.append(distance)

    # index of the IMF with the minimal distance
    min_distance_index = np.argmin(distances)
    return distances[min_distance_index]


# 
# The VMD function gives 3 output variables. u contains the decomposed signals, omega contains the frequency information. u_hat is an array that contains complex values which is used in computing the omega.
# 
# Put more expl.
# 

# In[7]:


#Idea: Compare best performed IMF's corr values of EMD-EEMD-CEEMDAN to see which one is the best


# In[8]:


# Function to calculate minimum distance for each peak in first dataset to any peak in second dataset
def calculate_distances(peaks1, peaks2):
    min_distances = [np.min(np.abs(peaks1[i] - peaks2)) for i in range(len(peaks1))]
    return min_distances

# Function to calculate average minimum distance as a measure of similarity
def calculate_similarity(peaks1, peaks2):
    min_distances = calculate_distances(peaks1, peaks2)
    if len(min_distances) > 0:
        average_min_distance = np.mean(min_distances)
    else:
        average_min_distance = np.inf  # if there are no peaks, set similarity to infinity
    return average_min_distance




# In[9]:


def gaussian_wave(mu, sigma, x):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)


# In[10]:


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


# In[11]:


# Define the parameter ranges
mu1_range = [0.5, 1.5]
sigma1_range = [0.5, 1.5]
mu2_range = [0.4, 0.8]
sigma2_range = [0.05, 0.15]
mu3_range = [1.2, 1.6]
sigma3_range = [0.05, 0.15]

# Set the number of random parameter combinations to try
num_combinations = 1000
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
    
    result_emd = fEMD(y_sum, y_truth)
    result_eemd = fEEMD(y_sum, y_truth)
    result_ceemdan = fCEEMDAN(y_sum, y_truth)
    result_wmd = fWMD(y_sum, y_truth)
    result_ewt = fEWT(y_sum, y_truth)    

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


# In[12]:


# Plot the results
x = range(1, num_combinations + 1)
plt.figure(figsize=(10,6))
plt.plot(x, results_emd, label='EMD')
plt.plot(x, results_eemd, label='EEMD')
plt.plot(x, results_ceemdan, label='CEEMDAN')
plt.plot(x, results_wmd, label='WMD')
plt.plot(x, results_ewt, label='EWT')

plt.xlabel('Sample')
plt.ylabel('Result')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10,6))

plt.plot(x, results_mu1, label='mean of slow drift')
plt.plot(x, results_sigma1, label='sigma of slow drift')
plt.plot(x, results_mu2, label='mean of fast drift-1')
plt.plot(x, results_sigma2, label='sigma of fast drift-1')
plt.plot(x, results_mu3, label='mean of fast drift-2')
plt.plot(x, results_sigma3, label='sigma of fast drift-2')

plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


print(np.mean(results_emd))  
print(np.mean(results_eemd)) 
print(np.mean(results_ceemdan)) 
print(np.mean(results_wmd)) 
print(np.mean(results_ewt)) 


# In[14]:


from mpl_toolkits.mplot3d import Axes3D

# Assuming your results are stored in lists.
results_all = [results_emd, results_eemd, results_ceemdan, results_wmd, results_ewt]
labels_all = ['EMD', 'EEMD', 'CEEMDAN', 'WMD', 'EWT']

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

x = np.diff([results_mu2, results_mu3], axis=0) #mean of the fast

y = np.mean( [results_sigma2, results_sigma3], axis=0) # 1/amplitude of the fast

for i, results in enumerate(results_all):
    ax.scatter(x, y, results, marker='o', label=labels_all[i], cmap = 'hot')

ax.set_xlabel('Distance between fast waves')
ax.set_ylabel('Flatness of fast waves')
ax.set_zlabel('Peak discrepancy')
plt.legend()
plt.grid(True)
plt.show()


# In[18]:


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
    h1 = axs[i][0].hist2d(x, results, bins=10, cmap='viridis')
    h2 = axs[i][1].hist2d(y, results, bins=10, cmap='viridis')
    
    fig.colorbar(h1[3], ax=axs[i][0])
    fig.colorbar(h2[3], ax=axs[i][1])
    
    axs[i][0].set_xlabel('Distance of fast waves')
    axs[i][0].set_ylabel('Peak discrepancy ')
    axs[i][0].set_title(labels_all[i])

    axs[i][1].set_xlabel('Flatness of fast waves')
    axs[i][1].set_ylabel('Peak discrepancy')
    axs[i][1].set_title(labels_all[i])

plt.tight_layout()
plt.show()


# In[16]:


results_all = [results_emd, results_eemd, results_ceemdan, results_wmd, results_ewt]
labels_all = ['EMD', 'EEMD', 'CEEMDAN', 'WMD', 'EWT']

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

x = ([results_mu1])  #mean of the slow
y = ([results_sigma1]) # 1/amplitude of the slow

for i, results in enumerate(results_all):
    ax.scatter(x, y, results, label=labels_all[i])

ax.set_xlabel('Mean of the Slow Drift')
ax.set_ylabel('Flatness of Slow Drift')
ax.set_zlabel('Peak discrepancy')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


# Assuming your results are stored in lists.
results_all = [results_emd, results_eemd, results_ceemdan, results_wmd, results_ewt]
labels_all = ['EMD', 'EEMD', 'CEEMDAN', 'WMD', 'EWT']

x = results_mu1  #mean of the slow
y = results_sigma1 # 1/amplitude of the slow

fig, axs = plt.subplots(len(results_all), 2, figsize=(20, 30))

for i, results in enumerate(results_all):
    results = np.ravel(results)  # Flatten results to a 1D array
    h1 = axs[i][0].hist2d(x, results, bins=10, cmap='viridis')
    h2 = axs[i][1].hist2d(y, results, bins=10, cmap='viridis')
    
    fig.colorbar(h1[3], ax=axs[i][0])
    fig.colorbar(h2[3], ax=axs[i][1])
    
    axs[i][0].set_xlabel('Mean of the slow drift')
    axs[i][0].set_ylabel('Peak discrepancy')
    axs[i][0].set_title(labels_all[i])

    axs[i][1].set_xlabel('Flatness of the slow drift')
    axs[i][1].set_ylabel('Peak discrepancy')
    axs[i][1].set_title(labels_all[i])

plt.tight_layout()
plt.show()


# In[28]:


from sklearn.decomposition import PCA

results = np.array([results_emd, results_eemd, results_ceemdan, results_wmd, results_ewt])

pca = PCA(n_components=2)
results_pca = pca.fit_transform(results)

plt.figure(figsize=(5, 3))
plt.scatter(results_pca[:, 0], results_pca[:, 1], marker='o', c=range(len(results)), cmap='viridis')

for i, label in enumerate(['EMD', 'EEMD', 'CEEMDAN', 'WMD', 'EWT']):
    plt.annotate(label, (results_pca[i, 0], results_pca[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.grid(True)
plt.show()


# In[ ]:




