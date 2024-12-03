# Description
'''
update LDPC encoder and decoder to nr-LDPC.
Calculate LLR according to voting score (Proportion of 1).
LLR in loss sequence are set to 0.5.
'''
from Model.Model import *
from Model.config import DEFAULT_PASSER, TM_NGS, TM_NNP

from Encode.Helper_Functions import preprocess, dna_to_int_array, load_dna
from Encode.DNAFountain import DNAFountain, Glass
from Analysis.Analysis import save_simu_result, dna_chunk, inspect_distribution, examine_strand
from Encode.InterLdpcEncode import InterLdpcEncoder, sequences_padding, Change_Direction
from Encode.Helper_Functions import bin_to_dna, dna_to_byte, dna_to_bin_array, int_to_binary_array, save_dna_files

import matplotlib.pyplot as plt
import scipy
import csv
import matlab.engine

import numpy as np
import math
import logging
import random

import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)
# plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300
np.set_printoptions(threshold=np.inf)

# %load_ext autoreload
# %autoreload

##=============Simulation parameter=======
Prob_E = np.linspace(0.118, 0.11, num = 5)
# Prob_E = [0.11]
seq_depth = 15
r_1 = 15 # Redundancy for index
r_2 = 99 # Redundancy for data
sample_num = 320
# sample_num = 2
##=============File path==================

# msg_filepath = 'D:\Desp-main\Data\Desp_Output_Data\message.csv'
# encode_filepath = f"D:\Desp-main\Data\Desp_Output_Data\encode_dnas_two-layer.txt"
# simu_filepath = 'D:\Desp-main\Data\Desp_Output_Data\simu_dnas_Two.txt'
# vscore_filepath = 'D:\Desp-main\Data\Desp_Output_Data\Voting_result.csv'
numeric_result_path = 'Data/Simulation Result of LDPC-RS Code for DNA Storage/Two Layer Scheme/Simulation-8/Simulation-1.csv'
##=====================Creat file for simulation result========
# Creat a CSV file for simulation result
header = [['r_1','r_2','Seq_depth', 'PE', 'BER', 'FER']]  # Header for the .csv file.
with open(numeric_result_path, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the list of lists to the CSV file
    writer.writerows(header)

##===============Coding argument======================
# DNA size
k_2 = 320  # length of data

# LDPC code parameter
n_0 = 8320 # Code length
k_0 = 7040 # Message length

print(f'LDPC code argument: n = {n_0}, k = {k_0}, code rate: {k_0/n_0: .2f}.')
index_length = int(math.ceil(np.log2(n_0)))  # bits
print(f'k_1 =  {index_length} bits, k_2 =  {k_2} bits.')

# BCH code parameter

# Parameter of BCH code for index.
k_1 = index_length
n_1 = k_1 + r_1  # Codeword length of BCH

# Parameter of BCH code for information bit.
n_2 = k_2 + r_2  # Codeword length of BCH

for ie, PE in enumerate(Prob_E):
    print(f'\n==========Experiemnt {ie+1}/{len(Prob_E)}===========\n')
    # =================Input data from file=====================
    '''
    #Files path
    file_name = 'Jnu.jpg'
    #对输入的数据进行分割
    data_segment = InterLdpcEncoder(file_name, chunk_size = chunk_size, encode = False)

    # print(f'first segment in byte: {data_segment.data[0]}')

    #Convert byte data to bit string.
    data_segment.bytes2bits()
    segments = data_segment.bit_sequences
    # print(f'first segment in bit: {bit_segments[0]}')
    k_ldpc = len(segments)
    print(f'Segment length: {len(segments[0])} bit')
    print(f'Segments number: {len(segments)}')
    '''
    ##================Generate random data===================
    print('Generating random bitstream...')
    msgs = []
    # Generate k_0 random bitstream of length 320.
    for i in range(k_0):
        msg = np.array([random.randint(0, 1) for _ in range(k_2)])
        msgs.append(msg)
    msgs = np.array(msgs)

    ## Save random bitstream to .csv file.
    # with open(msg_filepath, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(msgs)

    ## Read message from .csv file
    # with open(msg_filepath, mode='r') as file:
    #     reader = csv.reader(file)
    #     msgs = np.array([np.array(list(map(int, row))) for row in reader])

    ##=================LDPC encode===================
    print('LDPC encoding...')
    eng = matlab.engine.start_matlab()
    eng.cd(r'Codec-5g-nrldpc-BCH', nargout=0)
    # 转换垂直编码
    v_msgs = msgs.T  # Message in vertical direction.
    cwr1 = eng.nrldpc_Enc(v_msgs)  # LDPC encode

    ##================Generating index===================
    ids = []
    # Generate indices
    for id in range(n_0):
        ids.append(int_to_binary_array(id, index_length))
    ids = np.array(ids)

    ##================Two layer BCH encode in matlab module=================
    print('BCH encoding...')
    # Encode index by BCH encoder.
    cwr_ids = eng.BCH_Encoder(n_1, k_1, n_0, ids)

    # Encode information bit by BCH encoder.
    cwr1 = np.array(cwr1).T  # 转换为水平编码
    cwr_data = eng.BCH_Encoder(n_2, k_2, n_0, cwr1)

    ##================Concantenate Codewords of index and information bit=============
    # print('Concatenating  codewords of indices and information bit.')
    cwr2 = np.concatenate((cwr_ids, cwr_data), axis=1)

    ##===================Convert binary data to DNA=======================
    print('Binary to DNA...')
    dnas = []
    for cw in cwr2:
        dna = bin_to_dna(''.join(str(int(cw[i])) for i in range(n_1 + n_2)))
        dnas.append(dna)
    ##==================Save DNA template to .txt==========================
    '''
    # Open the file in write mode
    with open(encode_filepath, "w") as file:
        # Write each string in the list to the file
        for line in dnas:
            file.write(line + "\n")
    print(f"The encoded dna has been saved to {encode_filepath}.")
    '''
    ##================Read encoded DNA template from .txt==========
    '''
    print('Reading encoded DNA from', encode_filepath)
    with open(encode_filepath, 'r') as file:
    # Read all lines from the file and add them to a list
        lines = file.readlines()
    # Strip newline characters from each line
    dnas = [line.strip() for line in lines]
    '''
    ##======================DNA Storage Channel===========================
    print(f'>-------------------------------<')
    print(f'Channel: Pe = {PE}, d_seq = {seq_depth}.')
    # load dna
    # in_dnas = load_dna('files/jnu.dna') #Load from files
    in_dnas = dnas
    # Argument setting
    arg = DEFAULT_PASSER
    arg.syn_number = 30
    arg.syn_yield = 1
    # arg.syn_sub_prob = 0.57 * Pe / 6
    arg.syn_ins_prob = 0
    arg.syn_del_prob = 0

    arg.decay_er = 0
    arg.decay_loss_rate = 0.43 * PE

    arg.pcrc = 2
    arg.pcrp = 0.8
    arg.pcrBias = 0

    arg.seq_depth = seq_depth
    ps_seq = 0.57 * PE / 3  # 测序阶段单向替换概率
    arg.seq_TM = genTm(ps_seq)
    # arg.seq_TM = TM_NGS
    # Synthesis
    print('Synthesis is procceding...')
    SYN = Synthesizer(arg)  # set parameter for synthesis
    dnas_syn = SYN(in_dnas)
    random.shuffle(dnas_syn)  # Randomize the order of DNAs
    # Decay
    print('Decay is procceding...')
    DEC = Decayer(arg)
    dnas_dec = DEC(dnas_syn)
    # PCRer
    # print('PCR is procceding...')
    # PCR = PCRer(arg = arg)
    # dnas_pcr = PCR(dnas_dec)
    # Sequencing
    print('Sequencing is procceding...')
    SEQ = Sequencer(arg)
    dnas_seq = SEQ(dnas_dec)
    # Extract every DNA after the simulation  pipeline
    dnas_sim_result = []
    for dna_set in dnas_seq:
        for dna_error_profile in dna_set['re']:
            for i in range(dna_error_profile[0]):
                dnas_sim_result.append(dna_error_profile[2])

    # Open the file in write mode
    # with open(simu_filepath, "w") as file:
    #     # Write each string in the list to the file
    #     for line in dnas_sim_result:
    #         file.write(line + "\n")

    ## Check the length of every DNA
    dna_length = int(np.ceil((n_1 + n_2) / 2))
    normized_dnas_result = []
    for i, dna in enumerate(dnas_sim_result):
        if len(dna) != dna_length:
            # print(f'Length of {i}th dna is: {len(dna)}')
            ## Modify DNA to goal length
            if len(dna) < dna_length:
                dna = dna.ljust(dna_length, 'A')  # Pad 'A' at the end to meet the goal length.
            elif len(dna) > dna_length:
                dna = dna[:dna_length]  # cut off redundant base to meet the goal length.
        normized_dnas_result.append(dna)
    # print('Abnormal length DNA has been normized.')
    print(f'>-------------------------------<')
    ##==================dna to binary data===================
    print('DNA to Binary...')
    QUANT2BIN = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
    index_bin_arrays = []
    inf_bin_arrays = []
    for i, dna in enumerate(normized_dnas_result):
        # Convert dna strings to binary array like [0,1,0,0,0,1,0,0,0,1,0].
        # input: dna string to be converted
        # output: bin_str: binary array
        # print(f'Progress: {100 * i / len(dnas_sim_result): .2f}%', end='\r')
        bin_str = ''
        for b in dna:
            bin_str = bin_str + QUANT2BIN[b]
        indices_bin_str = bin_str[:n_1]
        inf_bin_str = bin_str[-n_2:]
        index_bin_array = np.array([float(s) for s in indices_bin_str])
        inf_bin_array = np.array([float(s) for s in inf_bin_str])
        index_bin_arrays.append(index_bin_array)
        inf_bin_arrays.append(inf_bin_array)
    simu_indices_arr = np.array(index_bin_arrays)
    simu_inf_arr = np.array(inf_bin_arrays)

    ##===============Two-layer BCH decode=============================
    print('BCH decoding...')
    rx_cwr_ids = eng.BCH_Decoder(n_1, k_1, len(dnas_sim_result), simu_indices_arr)
    rx_cwr_data = eng.BCH_Decoder(n_2, k_2, len(dnas_sim_result), simu_inf_arr)

    #  Convert double datatype to string.
    # Convert indices
    segments_temp = []
    for id in rx_cwr_ids:
        segment_temp = ''.join(str(int(s)) for s in id[1:])
        segments_temp.append(segment_temp)
    indices_dec_str = segments_temp

    # Convert information part
    segments_temp = []
    for data in rx_cwr_data:
        segment_temp = ''.join(str(int(s)) for s in data[1:])
        segments_temp.append(segment_temp)
    inf_bit_dec_str = segments_temp

    ##==========================Sorting==================
    segments_temp = []
    for i, inf_bit in enumerate(inf_bit_dec_str):
        segment_temp = dict(index=0, num=0, data=' ')
        segment_temp['index'] = int(indices_dec_str[i], 2)  # Convert binary index to integer index
        segment_temp['data'] = inf_bit
        segments_temp.append(segment_temp)
    segments = segments_temp

    # join sequences with the same index
    # print('Joining sequences...')
    segments_temp = []
    for i in range(n_0):
        # print(f'Progress: {100 * i / n_0: .2f}%', end='\r')
        segment_temp = dict(index=0, num=0, data=[])
        segment_temp['index'] = i
        for segment in segments:
            if segment['index'] == i and len(segment['data']) == k_2:
                segment_temp['num'] += 1
                segment_temp['data'].append(segment['data'])
        segments_temp.append(segment_temp)
    segments = segments_temp

    ##=================Voting============
    print('voting...')
    voting_result = []
    for i, segment in enumerate(segments):
        # print(f'Progress: {100 * i / n_0: .2f}%', end='\r')
        data = []
        if segment['num'] > 1:
            for j in range(k_2):
                bit_sum = 0
                for bit_string in segment['data']:
                    bit_sum += int(bit_string[j])
                data.append(float(bit_sum / segment['num']))
        elif segment['num'] == 1:
            data = [float(bit) for bit in segment['data'][0]]
        elif segment['num'] == 0:  # Generate a random bitstream if no read result
            data = [0.5 for _ in range(k_2)]
        data = np.array(data)
        voting_result.append(data)
    voting_result = np.array(voting_result)
    # Transform the matrix for LDPC decode.
    v_score = voting_result.T

    ## Save voting score to .csv file.

    # with open(vscore_filepath, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(v_score)

    ## Read voting score data from .csv file

    # with open(vscore_filepath, mode='r') as file:
    #     reader = csv.reader(file)
    #     v_score = np.array([np.array(list(map(float, row))) for row in reader])

    ##=================LDPC decode and calculate BER=====================
    # sample_num = 5  # LDPC decoder decode first 5 sequence for pre-experiment
    # Decode every sequence and calculate BER.
    print('LDPC decoding...')
    BERs = []
    re_v_msgs = eng.nrldpc_Dec_DnaVoting(v_score[:sample_num,:])
    re_msgs = np.array(re_v_msgs).T
    # Calculate BER
    err_num = np.sum(msgs[:, :sample_num] != re_msgs)  # count different bit between message and recovered message.
    BER = err_num / (k_0 * sample_num)
    # Calculte FER
    FE = 0  # Error Frame
    for i, re_msg in enumerate(re_msgs):
        if np.sum(msgs[i, :sample_num] != re_msg) > 0:
            FE += 1
    FER = FE / k_0
    print(f'Finish! FER = {FER:.8f}, BER = {BER:.8f}')
    data = [r_1,r_2,seq_depth, PE, BER, FER]
    with open(numeric_result_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
    print("Simulation Data has been written to .csv file in ", numeric_result_path)
