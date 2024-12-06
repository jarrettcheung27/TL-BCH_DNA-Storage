import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
from scipy.stats import norm
from scipy.integrate import quad
import math
import pandas as pd
from scipy.stats import binom
import math
import scipy.special as sp
plt.rcParams['figure.dpi'] = 200
##=======pre define function===============
# Function to calculate t_1
def calculate_t1(r_n1, k_1):
    # Apply the equation for t_1
    t_1 = np.floor(r_n1 / np.ceil(np.log2(k_1 + r_n1 + 1)))
    return t_1

# Function to calculate P(N_{e1} > t_1)
def calculate_Frame_error_prob(r_n1, m, P_e, t_1):
    # Step 1: Calculate the threshold for the summation
    n_threshold = np.ceil((3/4) * t_1).astype(int)

    # Step 2: Initialize the probability to 0
    prob = 0

    # Step 3: Perform the summation from n_threshold to r_n1 + M_1
    for n in range(n_threshold, int(np.ceil(0.5*r_n1 + m)) + 1):
        # Calculate the binomial coefficient
        binom_coeff = sp.comb(int(np.ceil(0.5*r_n1 + m)), n)
        # Calculate the term P_e^n * (1 - P_e)^(r_n1 + M_1 - n)
        term = binom_coeff * (P_e ** n) * ((1 - P_e) ** (int(np.ceil(0.5*r_n1 + m)) - n))
        # Add this term to the total probability
        prob += term

    return prob

# Function to add a column to the DataFrame and save to CSV
def add_column_to_csv(column_name, column_data, filename):
    global df
    # Add the new column to the DataFrame
    df[column_name] = column_data

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index = False)
# Parameter
m = 7 #Length of oligo/nt
k_2 = 2*m
r_n = 112 # total redundancy PC of two layer of BCH
k_1 = 14 #length of the first message (index)
# PE = [0.01, 0.02, 0.03, 0.05, 0.10]
PE = [0.02, 0.03, 0.05, 0.08, 0.1]
# PE = [0.01]
m_bch = np.ceil(math.log(k_2, 2)) # The Galois number
# m_bch = np.ceil(math.log(k_2, 2)) + 1 # The Galois number
max_r_n1 = 105 # Set maximum redundancy bit
df = pd.DataFrame()
numeric_result_path = f'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/FER_in_First_Layer_Vs_Redundancy1/Theory_result-1.csv'
# R_n1 = list(range(30, max_r_n1 + 1)) # an array of redundancy for 1 to max_redundancy
# R = [(5,108), (15,99),(24,99),(33,81),(39,72),(49,63),(56,54),(70,36),(84,27),(91,18),(105,9)]
R = [(5,108),(24,99),(33,81),(39,72),(49,63),(70,36),(84,27),(91,18),(105,9)]
R_n1 = [r[0] for r in R]
# R_n1 = np.linspace(5, max_r_n1, 100)
T_1 = [calculate_t1(r_n1, k_1) for r_n1 in R_n1]
add_column_to_csv('r_n1', R_n1, numeric_result_path)
# redundant bit number Vs success rate
add_column_to_csv('t', T_1, numeric_result_path)
# print('m_bch:', m_bch, ', t_max:', t_max)
markers = ['^','o','s','p','v']
colors = ['#EE5940','#F2AF30','#579B85','#2093AE','#A06C7D']
'''
# t Vs success rate
for Pe in PE:
    P = []
    for t in T:
        z = (t - (4/3)*m*Pe) / ((4/3)*np.sqrt(m*Pe*(1 - Pe)))
        # Calculate the probability
        p = norm.cdf(z)
        P.append(p)
    plt.plot(T, P, label = f'Pe = {Pe}', markersize=5) # Plot the curve of Probability of error bits more than t vs t.
plt.legend()
plt.title(f'Recovery success rate in different t (m = {m} nt)')
plt.xlabel('Error-correcting capability t')
plt.ylabel('Success Rate')
# plt.show()
file_path = 'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Graphical _Result/Normal Approximation/FER_in_First_Layer_Vs_Redundancy1/Information_part-theory'
plt.savefig(file_path)
'''

'''
for Pe in PE:
    FER = []
    for t in T:
        z = (t - (4/3)*m*Pe) / ((4/3)*np.sqrt(m*Pe*(1 - Pe)))
        # Calculate the probability
        p = norm.cdf(z)
        FER.append(1-p)
    add_column_to_csv(f'Success Rate(Pe = {Pe})', FER, numeric_result_path)
    plt.plot(reduns, FER, label = f'Pe = {Pe}', markersize=5) # Plot the curve of success rate vs redundancy.
'''

# bionomial distribution
for ip,Pe in enumerate(PE):
    FERs = []
    for i, r_n1 in enumerate(R_n1):
        # Calculate the probability
        t_1 = T_1[i]
        n_B1 = r_n1 + k_1 # Code length of the second layer
        Prob_FER = calculate_Frame_error_prob(r_n1, m , Pe, t_1)
        FERs.append(Prob_FER)
    add_column_to_csv(f'FER(Pe = {Pe})', FERs, numeric_result_path)
    plt.plot(R_n1, FERs,linestyle = 'dashed', linewidth = 2,color = colors[ip], marker = markers[ip], markerfacecolor='none', label = f'$P_e$ = {Pe}') # Plot the curve of success rate vs redundancy.
plt.legend(fontsize = 12)
# plt.title(f'Redundancy parity (m = {m} nt)')
plt.xlabel('$r_1$',fontsize=14)
plt.ylabel('FER',fontsize=14)
plt.yscale('log')
plt.tick_params(axis="x", which = "both", direction="in")
plt.tick_params(axis="y", which = "both", direction="in")
plt.xticks(fontsize=12)  # Set font size for x-axis ticks
plt.yticks(fontsize=12)  # Set font size for y-axis ticks
plt.grid(True, which='both', axis='both', linestyle=':', linewidth=0.2)
# plt.ylim([10e-7,0])
# plt.show()
file_path = f'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/FER_in_First_Layer_Vs_Redundancy1/4_FER_vs_R1_index_theory'
plt.savefig(file_path)
eps_file_path = f'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/FER_in_First_Layer_Vs_Redundancy1/4_FER_vs_R1_index_theory.eps'
plt.savefig(eps_file_path, bbox_inches='tight', format='eps')

