#%%
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import polarLib as polar

#%%
num_of_frame = int(1e3)
LL           = [1,2,4,8]      #list size
N            = 500

rate       = 1/2
design_snr = -2
crc        = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1], dtype = np.int64)

#%%
snr_min  = 0
snr_max  = 5
snr_step = 0.3
snr_db  = np.arange(snr_min, snr_max, snr_step)
ber     = np.zeros((len(LL),len(snr_db)))
fer     = np.zeros((len(LL),len(snr_db)))

for nnn in range(len(LL)):
    print('**'*50)
    L = LL[nnn]              # List size
    k  = int(N*rate)         # number of information bits + CRC bits
    k_crc = k - len(crc) + 1 # number of information bits remaining the use of CRC.

    ############# FIND WHETHER SHORTENING IS REQUIRED #############################
    n = int(np.ceil(np.log2(N)))
    if (1<<n) > N: # if shortening is required
        N2 = 1<<n
        n2 = n
    else:          # else no shortening is required
        N2 = N
        n2 = n

    ############## Polar Code Construction #######################################

    # Find bit-channel reliabilities using Bhattacharyya parameter
    Z   = polar.constructPolarCode(N2, design_param = design_snr)
    idx_inf_plus_frozen = polar.bitReverseArray(np.arange(N), n2)
    Z_inf_frozen        = Z[idx_inf_plus_frozen]
    argsort_list        = np.argsort(Z_inf_frozen)
    idx_used            = idx_inf_plus_frozen[argsort_list]
    idx_shortened       = polar.bitReverseArray(np.arange(N,N2), n2)
    idx_frozen          = np.copy(idx_used[k:])
    idx_inf             = np.copy(idx_used[:k])
    if_information_bit  = np.zeros(N2)       # "0" denotes that this bit is frozen
    if_information_bit[idx_inf]        = 1   # "1" denotes that this bit is information
    if_information_bit[idx_shortened]  = 2   # "2" denotes that this bit is shortened
    #------------------------------------------------------------------------------


    # SIMULATION PARAMETERS
    no_error_flag = 0  #MONTE CARLO ERROR COUNTER
    np.random.seed(0)

    print('CODE RATE:'+str(int(N*rate))+'/'+str(N))
    print('CA-SCL:'+str(L))

    for ss in range(len(snr_db)):
        snr        = 10.0**(snr_db[ss]/10.0)
        sigma2     = 1/(snr*rate)
        flag_break = False
        No = 10**(-snr_db[ss]/10)

        # malloc for encoding
        u = np.zeros(N2, dtype = np.int32)
        for qq in tqdm.tqdm(range(num_of_frame)):

            # Information bit generation and encoding
            u_inf = np.random.randint(0,2,k_crc)
            u_crc = polar.CRC_calculator(u_inf,crc)
            u[idx_inf[:k_crc]] = u_inf
            u[idx_inf[k_crc:]] = u_crc

            v = polar.polarEncoder(u)

            # BPSK MODULATION
            y = 2*v - 1

            # AWGN CHANNEL
            w = 0.707*np.random.randn(N2)
            x = y + w*np.sqrt(No/rate)

            # Calculate channel LLRs
            llr_x     = -x

            # replace Inf LLR for the shortened bits
            llr_x[idx_shortened] = np.inf

            #SCL+CRC decoding
            u_hat       = polar.SCLCRCDecoder(llr_x, if_information_bit, idx_inf, k_crc, L, crc)

            # SC Decoding
            # u_hat       = polar.SCDecoder(llr_x, if_information_bit)

            n_bit_error = np.sum(u_hat[idx_inf[:k_crc]] != u_inf)

            #Count bit error rate
            if n_bit_error>0:
                ber[nnn,ss] += n_bit_error
                fer[nnn,ss] += 1

            if fer[nnn,ss]>50:
                break

        ber[nnn,ss] /= (k*qq)
        fer[nnn,ss] /= qq

        if fer[nnn,ss] == 0:
            no_error_flag += 1
        else:
            no_error_flag = 0

        if no_error_flag == 3: # if 3 consecutive SNR values gives no error, than terminate the simulation.
            print('Precision reached.')
            break
# END OF LOOP


#%%

colors = ['c','r','b','m','navy','r']
plt.figure()
#BER
plt.subplot(121)
for ii in range(len(LL)):
    plt.semilogy(snr_db, ber[ii,:], ls = '-', marker ='s', color = colors[ii], label ='SCL, L=')
plt.grid(True, ls = '--', alpha = 0.5, which = 'both')
plt.xlabel('SNR [dB]')
plt.ylabel('BER')
plt.legend()

#FER
plt.subplot(122)
for ii in range(len(LL)):
    plt.semilogy(snr_db,fer[ii,:],ls = '-',marker ='s',color = colors[ii%len(colors)],label ='SCL, L=')
plt.grid(True,ls = '--',alpha =0.5, which = 'both')
plt.xlabel('SNR [dB]')
plt.ylabel('FER')
plt.legend()


# %%

# %%

# %%
