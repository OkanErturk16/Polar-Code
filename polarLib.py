import numpy as np
import numba

#%% Polar Code Code Construction

def bitReverse(x,n_bit):
    result = 0
    for i in range(n_bit):
        result <<= 1
        result |= x & 1
        x >>= 1
    return result


def bitReverseArray(x,n_bit):

    result = 0*x
    for kk in range(len(x)):
        for _ in range(n_bit):
            result[kk] <<= 1
            result[kk] |= x[kk] & 1
            x[kk] >>= 1
    return result

def constructPolarCode(N,design_param):
    n  = int(np.log2(N))
    if (1<<int(np.log2(N))) != N:
        N2 = 1<<(n+1)
    else:
        N2 = N

    # Slice information and frozen bits
    n2 = int(np.log2(N2))

    #Calculate error probabilty for a BMC channel
    e = np.exp(-10**(design_param/10.0))

    # Calculate Bhattacharyya parameters recursively

    u_idx = np.zeros((N2//2, n2), dtype = np.int32)
    l_idx = np.zeros((N2//2, n2), dtype = np.int32)

    for ll in range(1, n2 + 1):
        up_list_iter = np.zeros(N2//2)
        dw_list_iter = np.zeros(N2//2)
        N2_iter = 1<<ll
        N2_iter_half = N2_iter>>1
        for kk in range(N2//N2_iter):
            up_list_iter[kk*N2_iter_half: (kk+1)*N2_iter_half] = np.arange(kk*N2_iter,kk*N2_iter+N2_iter_half)
            dw_list_iter[kk*N2_iter_half: (kk+1)*N2_iter_half] = np.arange(kk*N2_iter+N2_iter_half,(kk+1)*N2_iter)

        u_idx[:,ll-1] = up_list_iter
        l_idx[:,ll-1] = dw_list_iter

    Z = np.zeros((N2, n2+1), dtype = np.float32)
    Z[:,0]  = e

    for ll in range(0,n2):
        Z[u_idx[:,n2-ll-1],ll+1] = Z[l_idx[:,n2-ll-1],ll] + Z[u_idx[:,n2-ll-1],ll] - Z[l_idx[:,n2-ll-1],ll]*Z[u_idx[:,n2-ll-1],ll]
        Z[l_idx[:,n2-ll-1],ll+1] = Z[l_idx[:,n2-ll-1],ll] * Z[u_idx[:,n2-ll-1],ll]


    return Z[:,-1]

#%% Methods for the implementation of SC/SCL/SCL+CRC decoders
@numba.jit(nopython = True)
def _f_node_minsum(a,b):
    return np.sign(a*b)*np.minimum(np.abs(a),np.abs(b)) 

@numba.jit(nopython = True)
def _g_node(LLR1,LLR2,s):
    return LLR1*(1-2*s) + LLR2

@numba.jit(nopython = True)
def _B_check(ll,ii):
    return (ii//(1<<ll))%2

@numba.jit(nopython = True)
def _s_updater(ll,ii,s):
    if _B_check(ll-1,ii): #if "g" node
        s[ll,ii] = s[ll-1,ii]
    else: #if "f" node
        if s[ll-1,ii] == -1:
            _s_updater(ll-1,ii,s)
        if s[ll-1,ii+(1<<(ll-1))] == -1:
            _s_updater(ll-1,ii+(1<<(ll-1)),s)           # Burasi da ayni sebeple yazildi!
        s[ll,ii] = s[ll-1,ii] ^ s[ll-1,ii+(1<<(ll-1))]  # MAIN ALGORITM

@numba.jit(nopython = True)
def Li(ll,ii,llrs,s):
    if llrs[ll,ii] != -np.inf:
        return llrs[ll,ii]
    else:
        if _B_check(ll,ii) == 0:
            llrs[ll,ii] = _f_node_minsum(Li(ll+1,ii,llrs,s), Li(ll+1, ii+(1<<ll), llrs, s))
            return llrs[ll,ii]
        else:
            if ll>0:
                _s_updater(ll,ii-(1<<ll),s)
            llrs[ll,ii] = _g_node(Li(ll+1, ii-(1<<ll),llrs,s), Li(ll+1,ii,llrs,s), s[ll,ii-(1<<ll)])
            return llrs[ll,ii]

#%% #* Polar Encoding
@numba.jit(nopython = True)
def polarEncoder(u):
    N = len(u)
    n = int(np.log2(N))
    v = np.copy(u)
    for ll in range(1,n+1):
        N_iter = 1<<ll
        N_iter_half = N_iter//2
        for kk in range(N//N_iter):
            v[kk*N_iter:(kk*N_iter + N_iter_half)] = (v[kk*N_iter:(kk*N_iter + N_iter_half)] ^ v[(kk*N_iter + N_iter_half):(kk+1)*N_iter])
    return v


#%% #* SC Decoding
@numba.jit(nopython = True)
def SCDecoder(llr_channel,if_information_bit):
    '''
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
        llr_channel : channel LLR values `numpy.float32`
        if_information_bit: stores if the bits correspoding to the indices are information bits or not  `bool`
    '''
    # Define Polar code length and depth
    N = len(llr_channel)
    n = int(np.log2(N))

    #malloc and initialize Decoder LLRS
    llrs       = -np.inf*np.ones((n+1, 1<<n), dtype = np.float32)
    llrs[-1,:] = llr_channel

    #HARD Decisions of SCL decoder
    s = -1*np.ones((n+1,int(1<<n)),dtype = np.int8)

    # Decode bits successively
    for ii in range(N):
        if if_information_bit[ii] == 0: #frozen bit
            s[0,ii]    = 0
            llrs[0,ii] = np.inf

        else:                           #information bit
            llrs[0,ii] = Li(0,ii,llrs,s)
            s[0,ii]    = llrs[0,ii]<0
    # return the decoded bit sequence
    return s[0,:]

#%% #* SCL Decoding
@numba.jit(nopython = True)
def SCLDecoder(llr_channel,if_information_bit,L):
    '''
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
        **llr_channel** : array_like
            channel LLR values `numpy.float32`
        if_information_bit: array_like
            stores if the bits correspoding to the indices are information bits or not, type  `bool`
        L: list size of the SCL `integer`

    '''
    # Define Polar code length and depth
    N     = len(llr_channel)
    n     = int(np.log2(N))

    #malloc and initialize Decoder LLRS
    llrs  = -np.inf*np.ones((n+1,1<<n),dtype=np.float32)

    llrs[-1,:] = llr_channel
    llrs = [llrs]*L

    #HARD Decisions of SCL decoder
    s = -1*np.ones((n+1,int(2**n)),dtype=np.int8)
    s = [s]*L

    DM    = np.zeros((L))                        # path metrics
    PM    = np.inf*np.ones((L),dtype=np.float32) # cumulative path metrics
    PM[0] = 0.0
    PM_DM = np.zeros((2*L))


    #--- START OF SCL Decoder ---
    for ii in range(N):
        #---START OF "IF INFORMATION BIT"---
        if if_information_bit[ii] == 0:
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = 0
                PM[dd]        += -llrs[dd][0,ii]*(llrs[dd][0,ii]<0)
        #---END OF "IF INFORMATION BIT"---


        elif if_information_bit[ii] == 2:
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = 0
        #---END OF "IF INFORMATION BIT"---

        #---START OF "IF FROZEN BIT"---
        else: #information bit
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = llrs[dd][0,ii]<0
                DM[dd]         = np.abs(llrs[dd][0,ii])
        #---END OF "IF FROZEN BIT"---

        #SELECTING THE BEST "L" PATHS
        if if_information_bit[ii] and L>1:
            PM_DM[:L] = PM        #Path metrics
            PM_DM[L:] = PM + DM   #Path metric + Decision metrics
            idx_sort   = np.argsort(PM_DM) #sort path metrics

            #find decoders in the list need to be updated
            idx_min_low  = idx_sort[:L][idx_sort[:L]>=L]-L
            idx_min_up   = idx_sort[L:][idx_sort[L:]<L]

            # If decoders in the list need to be updated
            len_list_change = len(idx_min_low) # or len(idx_min_up)
            if  len_list_change != 0:

                #---START OF DECODER COPYING---
                for bb in range(len_list_change):
                    llrs[idx_min_up[bb]] = np.copy(llrs[idx_min_low[bb]])  # LLR degerlerini tasi
                    s[idx_min_up[bb]]    = np.copy(s[idx_min_low[bb]])     # bitleri tasi
                    s[idx_min_up[bb]][0,ii] = 1-s[idx_min_low[bb]][0,ii]   # cozulen son biti flip et
                #---END OF DECODER COPYING---

                #Path metric guncelle
                PM[idx_min_up] = PM_DM[idx_min_low + L]
        #---END OF LIST UPDATE---
    #---END OF SCL DECODING---

    #Select the possible codeword having the minimum path metric
    dd_best = np.argmin(PM)
    return s[dd_best][0,:]


#%% #* SCL Decoding

@numba.jit(nopython = True)
def CRC_calculator(x,crc_polynomial):
    crc_polynomial_len = len(crc_polynomial)
    x_new = np.zeros(len(x) + crc_polynomial_len-1,dtype=np.int32)
    x_new[:len(x)] = x
    for ii in range(len(x_new)-crc_polynomial_len+1):
        if x_new[ii] == 1:
            x_new[ii:(ii+crc_polynomial_len)] ^= crc_polynomial
    return x_new[(-crc_polynomial_len+1):]

@numba.jit(nopython = True)
def SCLCRCDecoder(llr_channel,if_information_bit,information_indices,k_crc,L,crc_polynomial):
    '''
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
        **llr_channel** : array_like
            channel LLR values `numpy.float32`
        if_information_bit: array_like
            stores if the bits correspoding to the indices are information bits or not, type  `bool`
        L: list size of the SCL `integer`
    '''
    # Define Polar code length and depth
    N     = len(llr_channel)
    n     = int(np.log2(N))

    #malloc and initialize Decoder LLRS
    llrs  = -np.inf*np.ones((n+1,1<<n),dtype=np.float32)

    llrs[-1,:] = llr_channel
    llrs       = [llrs]*L

    #HARD Decisions of SCL decoder
    s = -1*np.ones((n+1,int(2**n)),dtype=np.int8)
    s = [s]*L

    DM    = np.zeros((L))                        # o anki yol metrikleri
    PM    = np.inf*np.ones((L),dtype=np.float32) # kumulatif yol metrikleri
    PM[0] = 0.0
    PM_DM = np.zeros((2*L))


    #--- START OF SCL Decoder ---
    for ii in range(N):
        #---START OF "IF INFORMATION BIT"---
        if if_information_bit[ii] == 0:
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = 0
                PM[dd]        += -llrs[dd][0,ii]*(llrs[dd][0,ii]<0)
        #---END OF "IF INFORMATION BIT"---

        #---START OF "IF SHORTENED BIT"---
        elif if_information_bit[ii] == 2:
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = 0
        #---END OF "IF SHORTENED BIT"---

        #---START OF "IF FROZEN BIT"---
        else: #information bit
            for dd in range(L): #her liste elemani icin
                llrs[dd][0,ii] = Li(0,ii,llrs[dd],s[dd])
                s[dd][0,ii]    = llrs[dd][0,ii]<0
                DM[dd]         = np.abs(llrs[dd][0,ii])
        #---END OF "IF FROZEN BIT"---

        #SELECTING THE BEST "L" PATHS
        if if_information_bit[ii] and L>1:
            PM_DM[:L] = PM        #Path metrics
            PM_DM[L:] = PM + DM   #Path metric + Decision metrics
            idx_sort   = np.argsort(PM_DM) #sort path metrics

            #find decoders in the list need to be updated
            idx_min_low  = idx_sort[:L][idx_sort[:L]>=L]-L
            idx_min_up   = idx_sort[L:][idx_sort[L:]<L]

            # If decoders in the list need to be updated
            len_list_change = len(idx_min_low) # or len(idx_min_up)
            if  len_list_change != 0:

                #---START OF DECODER COPYING---
                for bb in range(len_list_change):
                    llrs[idx_min_up[bb]] = np.copy(llrs[idx_min_low[bb]])  # LLR degerlerini tasi
                    s[idx_min_up[bb]]    = np.copy(s[idx_min_low[bb]])     # bitleri tasi
                    s[idx_min_up[bb]][0,ii] = 1-s[idx_min_low[bb]][0,ii]   # cozulen son biti flip et
                #---END OF DECODER COPYING---

                #Path metric guncelle
                PM[idx_min_up] = PM_DM[idx_min_low + L]
        #---END OF LIST UPDATE---
    #---END OF SCL DECODING---


    # Check L codewords whether they satisfy the CRC.
    crc_satisfied = np.zeros(L)
    len_crc       = len(crc_polynomial)
    for ii in range(L):
        crc_iter = CRC_calculator(s[ii][0,:][information_indices[:k_crc]], crc_polynomial)
        if np.all(crc_iter == s[ii][0,:][information_indices[k_crc:]]):
            crc_satisfied[ii] = 1

    #check if multiple decoded codewords satisfy the CRC.
    n_crc_staisfied = np.sum(crc_satisfied)


    if  n_crc_staisfied == 1:  #If there is only one codeword satisfies the CRC, return it.
        dd_best = np.nonzero(crc_satisfied==1)[0][0]
        return s[dd_best][0,:]

    elif n_crc_staisfied >= 1: #If multiple satisfy, return whose path metric is minimum.
        dd_best_list = np.nonzero(crc_satisfied==1)[0]
        ddd          = np.argmin(PM[dd_best_list])
        return s[dd_best_list[ddd]][0,:]
    else:
        # If none of the codewords satisfies the CRC, return whose path metric is minimum.
        dd_best = np.argmin(PM)
        return s[dd_best][0,:]
