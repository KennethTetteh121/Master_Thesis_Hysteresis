#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Given parameters for Van Genuchten equation
# Model Constraints: alpha_drying = 2 * alpha_wetting, theta_r_drying = theta_r_wetting = theta_r, n_drying = n_wetting
theta_s_wetting = 0.42
theta_s_drying = 0.53
theta_m_drying = 0.566
theta_r = 0
alpha = 0.01302896
n = 1.345
m = 1 - (1/n)

# Computing the water saturation function (Se)
def calculate_Se(h_values, alpha, n, m):
    S = (alpha * np.abs(h_values))**n
    Se_values = (1 + S)**(-m)
    return Se_values

# Function to calculate the van Genuchten retention
def calculate_Theta_VG_Wetting(Se_values, theta_s_wetting, theta_r):
    Theta_VG_Wetting = Se_values * (theta_s_wetting - theta_r) + theta_r
    return Theta_VG_Wetting

# Function for computing conceptual Mualem hysteresis
def calculate_Theta_Mualem_Drying(Theta_VG_Wetting, theta_s_wetting, theta_r):
    Theta_Mualem_Drying = (((2 * theta_s_wetting) - Theta_VG_Wetting - theta_r) * ((Theta_VG_Wetting - theta_r) / (theta_s_wetting - theta_r))) + theta_r
    return Theta_Mualem_Drying

# Generating function for matric potential, h (cm); starting at -1 cm and ending at - 34156133.22 cm at steps of 1.05
# start at p
p = 1

# End at Max_value 34156133 cm (pF 7.5)
max_value = 34156133.22

# Listing and storing h values, van Genuchten retention values, and pF values
h_values = [0]
Theta_VG_Wetting = [0]
pF_values_wetting = [0]

# Lists to store Theta_Mualem_Drying values and pF values
Theta_Mualem_Drying = [0]
pF_values_drying = [0]

while p < max_value:
    # Append the current h value
    h_values.append(-p)  # regenerate h_values as negative

    # Calculate van Genuchten retention
    Se_values = calculate_Se(h_values, alpha, n, m)
    Theta_VG = calculate_Theta_VG_Wetting(Se_values[-1], theta_s_wetting, theta_r)
    Theta_VG_Wetting.append(Theta_VG)

    # Calculate pF values for wetting
    pF_wetting = np.log10(p) if p > 0 else 0
    pF_values_wetting.append(pF_wetting)

    # Calculate Theta_Mualem_Drying directly using Theta_VG_Wetting
    Theta_Mualem_Drying_value = calculate_Theta_Mualem_Drying(Theta_VG, theta_s_wetting, theta_r)
    Theta_Mualem_Drying.append(Theta_Mualem_Drying_value)

    # Calculate pF values for drying
    pF_drying = np.log10(p) if p > 0 else 0
    pF_values_drying.append(pF_drying)

    # Displaying the predictions
    print(f"h = {-p:.6f}, Theta_VG_Wetting = {Theta_VG:.6f}, Theta_Mualem_Drying = {Theta_Mualem_Drying_value:.6f}")

    # Increase p
    p *= 1.05

# Given parameter for drying
theta_s_drying = 0.53

# Compute Theta values for Mualem_hysteresis
Theta_Mualem_Drying = (((2 * theta_s_wetting) - np.array(Theta_VG_Wetting) - theta_r) *
                       ((np.array(Theta_VG_Wetting) - theta_r) / (theta_s_wetting - theta_r))) + theta_r

# Predict Theta values for modified Mualem Hysteresis (call it Mualem Modified 1)
Theta_Mualem_Drying_Modified_1 = (((2 * theta_s_drying) - np.array(Theta_VG_Wetting) - theta_r) *
                                  ((np.array(Theta_VG_Wetting) - theta_r) / (theta_s_drying - theta_r))) + theta_r

# Predict Theta values for second modified Mualem Hysteresis (call it Mualem Modified 2)
Theta_Mualem_Drying_Modified_2 = (((2*0.566)-np.array(Theta_VG_Wetting)-theta_r)*((np.array(Theta_VG_Wetting)-theta_r)/(0.566-theta_r)))+theta_r

# Provided data for pFHyprop and WCHyprop
pFHyprop = [0.345, 0.259, 0.247, 0.295, 0.369, 0.52, 0.671, 0.721, 0.628, 0.574, 0.637, 0.681, 0.756, 0.731, 0.723,
            0.846, 0.922, 0.971, 1.046, 1.079, 1.114, 1.165, 1.156, 1.162, 1.245, 1.326, 1.375, 1.423, 1.467, 1.508,
            1.551, 1.593, 1.636, 1.679, 1.717, 1.754, 1.789, 1.823, 1.855, 1.884, 1.913, 1.941, 1.968, 1.998, 2.027,
            2.056, 2.085, 2.113, 2.142, 2.17, 2.196, 2.222, 2.247, 2.271, 2.294, 2.315, 2.336, 2.356, 2.374, 2.391,
            2.408, 2.424, 2.442, 2.458, 2.471, 2.483, 2.495, 2.507, 2.518, 2.53, 2.542, 2.555, 2.566, 2.576, 2.586,
            2.597, 2.61, 2.621, 2.631, 2.641, 2.65, 2.659, 2.669, 2.681, 2.692, 2.702, 2.713, 2.723, 2.732, 2.742,
            2.751, 2.761, 2.771, 2.782, 2.794, 2.807, 2.82, 2.834, 2.848]

WCHyprop = [0.534, 0.5339, 0.5338, 0.5336, 0.5334, 0.5331, 0.5327, 0.5323, 0.5318, 0.5312, 0.5306, 0.5299, 0.5292,
            0.5284, 0.5276, 0.5267, 0.5257, 0.5246, 0.5235, 0.5224, 0.5212, 0.5199, 0.5185, 0.5171, 0.5157, 0.5141,
            0.5125, 0.5109, 0.5092, 0.5074, 0.5056, 0.5037, 0.5017, 0.4997, 0.4976, 0.4955, 0.4933, 0.491, 0.4886,
            0.4862, 0.4837, 0.4811, 0.4785, 0.4759, 0.4733, 0.4706, 0.4679, 0.4652, 0.4624, 0.4596, 0.4568, 0.4539,
            0.451, 0.4481, 0.4451, 0.4421, 0.4392, 0.4362, 0.4332, 0.4302, 0.4273, 0.4243, 0.4214, 0.4184, 0.4154,
            0.4124, 0.4093, 0.4062, 0.4031, 0.4, 0.3968, 0.3936, 0.3904, 0.3872, 0.3839, 0.3806, 0.3773, 0.374,
            0.3706, 0.3672, 0.3638, 0.3603, 0.3568, 0.3533, 0.3498, 0.3462, 0.3426, 0.3389, 0.3352, 0.3315, 0.3277,
            0.3239, 0.3201, 0.3163, 0.3124, 0.3085, 0.3045, 0.3003, 0.2961]
pFHypropAE = [3.535]
WCHypropAE = [0.2187]

# Additional data for pFd and WCd
pFd = [3.5, 3.64, 3.9, 4.73, 5.77, 6.11, 6.31, 4.32, 4.77, 5.16, 5.41, 5.78]
WCd = [0.232528132, 0.185468867, 0.141177794, 0.082123031, 0.036909227, 0.01375919, 0,
        0.111300771, 0.075254499, 0.054385604, 0.046796915, 0.03351671]

pFw = [6.14, 4.84, 4.7, 4.15, 3.52, 5.76, 4.43, 4.12, 3.67, 5.92, 5.07, 4.09, 3.77, 6.14, 4.53, 3.94]
WCw = [0.010721078, 0.04653433, 0.05005814, 0.07630273, 0.121072478, 0.021372719, 0.062217899, 0.077979713, 0.109948553,
       0.018666134, 0.037232594, 0.077708998,
       0.095174247, 0.011081081, 0.056736902, 0.08614786]

pFSB = [2, 2, 2, 1, 1, 1]
WCSB = [0.315882314, 0.311876503, 0.326098, 0.405235376, 0.412639676, 0.4018334]

pFWind = [2.46686762, 2.301029996, 2]
WCWind = [0.2437, 0.271952152, 0.299486546]

# Plotting
plt.figure(figsize=(12, 8))
# Scatter plot
plt.scatter(WCHyprop, pFHyprop, color='brown', marker='o', facecolor='none', label='Drying Hyprop', s=100)
plt.scatter(WCHypropAE, pFHypropAE, color='brown', marker='x', label='AEP Hyprop', s=100)
plt.scatter(WCd, pFd, color='red', marker='o', label='Drying WP4C', s=100)
plt.scatter(WCw, pFw, color='blue', marker='o', label='Wetting WP4C', s=100)
plt.scatter(WCSB, pFSB, color='orange', marker='s', facecolor='none', edgecolor='blue', label='Wetting HWC SB', s=100)
plt.scatter(WCWind, pFWind, color='green', marker='*', s=150, facecolor='none', label='Wetting HWC CP')

# Plot wetting, drying, and modified drying curves on the same graph
plt.plot(Theta_VG_Wetting[1:], pF_values_wetting[1:], label='Wetting Predicted VG')
plt.plot(Theta_Mualem_Drying[1:], pF_values_drying[1:], label='Drying Original Mualem')
plt.plot(Theta_Mualem_Drying_Modified_1[1:], pF_values_drying[1:], label='Mualem Modified 1')
plt.plot(Theta_Mualem_Drying_Modified_2[1:], pF_values_drying[1:], label='Mualem Modified 2')

plt.xlabel('Volumetric Water Content, θ  (cm³ cm⁻³)', fontsize=12)
plt.ylabel('pF (log |h cm |)', fontsize=12)
plt.title("Exploring Mualem (1977) Hysteresis model under Kool & Parker's (1987) constraints with modifications according to Vogel (1988)")
plt.xlim(-0.02, 0.62)
plt.ylim(-0.02, 7.7)
plt.legend()

# Other plotting code...

# Use Markdown for italic text in a code cell
plt.text(0.25, 6.5, 'Tetteh et al. (2023)', fontsize=14, ha='center', va='center')
plt.text(0.25, 5.2, 'Model Simplification' , fontsize=14, ha='center', va='center')
plt.text(0.25, 4.8, r'$\alpha^{w} = 2\alpha^{d} = 0.013 \, \mathrm{cm}^{-1}$', fontsize=12, ha='center', va='center')
plt.text(0.25, 4.4, r'$n^{w} = n^{d} = 1.345$', fontsize=12, ha='center', va='center')
plt.text(0.25, 4.0, r'$\theta_{r}^{w} = \theta_{r}^{d} = 0 \, \mathrm{cm}^3 \, \mathrm{cm}^{-3}$', fontsize=12, ha='center', va='center')

plt.show()




# In[ ]:




