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
##=======pre define function===============
# Function to calculate t_1
def calculate_t(r, k):
    # Apply the equation for t_1
    t = np.floor(r / np.ceil(np.log2(k + r + 1)))
    return t
def calc_FER_of_index(code_length, P_e, t_1):
    '''
    Calculate the FER of index
    n: code length of index / bits.
    P_e: base edit prabability.
    t_1: error-correcting capability / bits

    prob: FER of index
    '''
    # Tranform code length from bit to base.
    M = np.ceil(code_length/2).astype(int)
    # Step 1: Calculate the threshold for the summation
    n_threshold = np.ceil((3/4) * t_1).astype(int)

    # Step 2: Initialize the probability to 0
    prob = 0

    # Step 3: Perform the summation from n_threshold to r_n1 + M_1
    for n in range(n_threshold, M + 1):
        # Calculate the term P_e^n * (1 - P_e)^(r_n1 + M_1 - n)
        term = sp.comb(M, n) * (P_e ** n) * ((1 - P_e) ** (M - n))
        # Add this term to the total probability
        prob += term
    return prob
# Define the function to be integrated
def integrand(x, n, P_e):

    norm_x = (x - (4 / 3) * n * P_e) / ((4 / 3) * np.sqrt(n * P_e * (1 - P_e)))
    # Standard normal PDF (phi(x))
    phi_x = norm.pdf(norm_x)

    return phi_x * x

# Function to calculate error bit in information bit
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


def pmf_k(k, d_syn, P_e):
    """Calculate the PMF of K."""
    p_k = 1 - 0.43 * P_e
    return binom.pmf(k, d_syn, p_k)

def pmf_q(q, k, d_seq, d_syn, P_e):
    """Calculate the PMF of Q."""
    p_q = d_seq / (d_syn * (1 - 0.43 * P_e))
    return binom.pmf(q, k, p_q)

def pmf_m_k(m_k, q, E_1):
    """Calculate the PMF of M_k."""
    return binom.pmf(m_k, q, 1 - E_1)

def prob_x_greater_than_half(m_k, epsilon):
    """Calculate P(X > m_k / 2) for a given m_k."""
    threshold = m_k / 2
    return 1 - binom.cdf(threshold, m_k, epsilon)


def cal_ps_tilde(d_syn, d_seq, P_e, E_1, epsilon):
    """Calculate the total probability P(X > M_k / 2) considering K and Q as random variables."""
    ps_tilde = 0.0

    for k in range(0, d_syn + 1):
        pmf_k_value = pmf_k(k, d_syn, P_e)

        for q in range(0, k + 1):
            pmf_q_value = pmf_q(q, k, d_seq, d_syn, P_e)

            for m_k in range(0, q + 1):
                pmf_m_k_value = pmf_m_k(m_k, q, E_1)
                prob_value = prob_x_greater_than_half(m_k, epsilon)
                ps_tilde += pmf_k_value * pmf_q_value * pmf_m_k_value * prob_value

    return ps_tilde

# Function to add a column to the DataFrame and save to CSV
def add_column_to_csv(column_name, column_data, filename):
    global df
    # Add the new column to the DataFrame
    df[column_name] = column_data

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index = False)

# Parameter
d_seq = 15
d_syn = 30 # Aynthesis number.
average_copy = d_syn # Average_copy before sequenceing.
m1 = 7 # Length of index/nt
m2 = 160 # Length of oligo/nt
k_1 = 2*m1 #length of the first message (index)
k_2 = 2*m2
# r_n = 112 # total redundancy PC of two layer of BCH

df = pd.DataFrame()
# numeric_result_path = 'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/FER_Vs_R1/Theory_result-R132-d15-v1.csv'

# Total R ~ 92
# R = [(5,90),(10,81),(15,72),(24,63),(39,54),(45,45),(47,45),(77,13),(84,14)]
R = [(5,90),(10,81),(15,72),(24,63),(47,45)]
T_1 = [1,2,3,4,11]
T_2 = [10,9,8,7,5]
PE1 = np.linspace(0.035,0.08,10)
PE2 = np.linspace(0.04,0.075,10)
PE3 = np.linspace(0.035,0.065,10)
PE4 = np.linspace(0.03,0.055,10)
PE5 = np.linspace(0.01,0.04,10)
PE_set = [PE1,PE2,PE3,PE4,PE5]

# Total R ~ 112
# R = [(5,108),(15,99),(33,81),(56,54),(84,27)]
# T_1 = [1,3,6,9,14]
# T_2 = [12,11,9,6,3]
# PE1 = np.linspace(0.04,0.095,10)
# PE2 = np.linspace(0.05,0.095,10)
# PE3 = np.linspace(0.04,0.075,10)
# PE4 = np.linspace(0.025,0.05,10)
# PE5 = np.linspace(0.005,0.025,10)
# PE_set = [PE1,PE2,PE3,PE4,PE5]

# Total R ~ 132
# R = [(5,126),(15,117),(24,108),(45,90),(77,54)]
# T_1 = [1,3,4,10,13]
# T_2 = [14,13,12,10,6]
# PE1 = np.linspace(0.04,0.105,10)
# PE2 = np.linspace(0.04,0.11,10)
# PE3 = np.linspace(0.03,0.1,10)
# PE4 = np.linspace(0.02,0.08,10)
# PE5 = np.linspace(0.01,0.05,10)
# PE_set = [PE1,PE2,PE3,PE4,PE5]

R_n1 = [r[0] for r in R]
R_n2 = [r[1] for r in R]
# add_column_to_csv('PE', PE, numeric_result_path)
markers = ['^','o','s','p','v']
colors = ['#EE5940','#F2AF30','#579B85','#2093AE','#A06C7D']
# bionomial distribution
for i, r_n1 in enumerate(R_n1):
    PE = PE_set[i]
    FERs = []
    for Pe in PE:
        # Calculate the probability
        # Base-level-error
        Ps = 0.57 * Pe
        # Sequence-level-error
        Pl = 0.43 * Pe
        t_1 = T_1[i]
        t_2 = T_2[i]
        n_1 = r_n1 + k_1 # Code length of the first layer
        n_2 = R_n2[i] + k_2  # Code length of the second layer
        # Calculate the probability of error index.
        E1 = calc_FER_of_index(n_1, Ps, t_1)
        # Calculate ber in information bits.
        delta = 0.38*Pe
        z = func_cal_z(delta, n_2, k_2)
        E2 = Q_function(z)
        # Calculate \tilde{p_e} according to E1 and E2
        Ps_tilde = cal_ps_tilde(d_syn, d_seq, Pe, E1, E2)
        FER = 1 - (1 - Ps_tilde)**k_2
        # Calculate of probability of sequence with error bits using binomial.
        FERs.append(FER)
    # add_column_to_csv(f'FER(R = ({R_n1[i]},{R_n2[i]})', FERs, numeric_result_path)
    plt.plot(PE, FERs,  linestyle = 'dashed', linewidth = 2,color = colors[i], marker = markers[i], markerfacecolor='none', label = f'($r_1$,$r_2$) = ({R_n1[i]},{R_n2[i]})') # Plot the curve of success rate vs redundancy.
plt.legend(fontsize = 12, loc = 'upper left')
# plt.title('FER verse $P_e$'+f', d_seq = {d_seq}, d_syn = {d_syn}')
plt.xlabel('$P_e$',fontsize=14)
plt.ylabel('FER',fontsize=14)
plt.tick_params(axis="x", which = "both", direction="in")
plt.tick_params(axis="y", which = "both", direction="in")
plt.xticks(fontsize=12)  # Set font size for x-axis ticks
plt.yticks(fontsize=12)  # Set font size for y-axis ticks
plt.yscale('log')
plt.grid(True, which='both', linestyle=':', linewidth=0.1)
plt.ylim([10e-6,0])
# plt.show()
file_path = f'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/FER_Vs_R1/FER_vs_R92_d{d_seq}_theory'
plt.savefig(file_path)
esp_file_path = f'D:/DeSP-main/Data/Simulation Result of LDPC-RS Code for DNA Storage/Normal_Approximation/FER_Vs_R1/5_FER_vs_R92_d{d_seq}_theory.eps'
plt.savefig(esp_file_path, bbox_inches='tight', format='eps')

