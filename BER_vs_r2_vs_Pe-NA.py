import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
from scipy.stats import norm
from scipy.integrate import quad
import math
import pandas as pd
from scipy.stats import binom
import math
plt.rcParams['figure.dpi'] = 200
##=======pre define function===============
# Function to add a column to the DataFrame and save to CSV
def add_column_to_csv(column_name, column_data, filename):
    global df
    # Add the new column to the DataFrame
    df[column_name] = column_data

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index = False)

def Q_function(z):
    '''
    Q-function in probability theory.
    '''
    # Define the integrand for the Q function
    def integrand(y):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-y**2 / 2)

    # Perform the integration from z to infinity
    result, _ = quad(integrand, z, np.inf)
    return result

def h(delta):
    """
    Calculate the binary entropy function h(delta) for BSC.
    """
    if delta == 0 or delta == 1:
        return 0
    return -delta * np.log2(delta) - (1 - delta) * np.log2(1 - delta)
def func_cal_z(delta, n, k):
    """Calculate z based on the given formula."""
    # Calculate the entropy
    entropy = h(delta)

    # Calculate the term O(1) (here we will assume it's small enough to be ignored, or you can define a specific value)
    O_1 = 0  # You can replace this with an appropriate constant if needed

    # Calculate z
    numerator = n * (1 - entropy) - k + (1 / 2) * np.log2(n) + O_1
    denominator = np.sqrt(n * delta * (1 - delta)) * np.log2((1 - delta) / delta)

    return numerator / denominator

# ===============file path==================#
numeric_result_path = f'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/BER_in_Second-layer_vs_r1/FER_vs_R1_data_theory.csv'
graphic_result_path = f'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/BER_in_Second-layer_vs_r1/FER_vs_R1_data_theory'
df = pd.DataFrame()

# Parameter
m = 160 #Length of oligo/nt
k_2 = 2*m
n_0 = 112 # total redundancy PC of two layer of BCH
k_1 = 14 #length of the first message (index)
PE = [0.02,0.03,0.05,0.08,0.1]
max_r_n1 = 105 # Set maximum redundancy bit
markers = ['^','o','s','p','v']
colors = ['#EE5940','#F2AF30','#579B85','#2093AE','#A06C7D']
# Redundancy in two layer.
R_1 = np.linspace(5,105,10)
R_2 = [(n_0 - r1) for r1 in R_1 ]
add_column_to_csv('r_2', R_2, numeric_result_path)
for ip, pe in enumerate(PE):
    BERs = []
    for r_1 in R_1:
        n_2 = k_2 + (max_r_n1-r_1)
        # pe to crossover probability in BSC
        delta = 0.38*pe
        z = func_cal_z(delta, n_2, k_2)
        ber = Q_function(z)
        BERs.append(ber)
    add_column_to_csv(f'BER(Pe = {pe})', BERs, numeric_result_path)
    plt.plot(R_2, BERs,linestyle = 'dashed', linewidth = 2, color = colors[ip], marker = markers[ip], markerfacecolor='none', label = f'$P_e$ = {pe}', markersize=5) # Plot the curve of success rate vs redundancy.
plt.legend(fontsize = 12)
# plt.title(f'Redundancy parity (m = {m} nt)')
plt.xlabel('$r_2$',fontsize=14)
plt.ylabel('BER',fontsize=14)
plt.yscale('log')
plt.tick_params(axis="x", which = "both", direction="in")
plt.tick_params(axis="y", which = "both", direction="in")
plt.xticks(fontsize=12)  # Set font size for x-axis ticks
plt.yticks(fontsize=12)  # Set font size for y-axis ticks
plt.grid(True, which='both', linestyle=':', linewidth=0.2)
plt.savefig(graphic_result_path)
esp_file_path = 'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/BER_in_Second-layer_vs_r1/4_BER_vs_R1_data_theory.eps'
plt.savefig(esp_file_path, bbox_inches='tight',  format='eps')

