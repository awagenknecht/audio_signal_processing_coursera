# Analysis and synthesis of sound using Sinusoidal Model with multiresolution DFT
# Week 10 assignment on multiresolution windowing

import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.signal.windows import blackmanharris, triang
from scipy.signal import get_window
from scipy.fft import ifft, fftshift
import math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import dftModel as DFT
import utilFunctions as UF

def sineModelMultiRes(x, fs, w, M, N, B, t):
    """
	Analysis/synthesis of a sound using a multiresolution sinusoidal model, without sine tracking.
    At each frame, three DFTs of different window sizes are computed for three frequency bands given as input.
	x: input array sound,
    fs: sample rate,
    window: tuple of three analysis windows of different lengths,
    M: tuple of three analysis window sizes,
    N: tuple of three FFT sizes for analysis,
    B: tuple of two frequency values (in Hz) separating the range [0, fs/2] into three bands,
    t: threshold in negative dB
	returns y: output array sound
	"""
    M1, M2, M3 = M[0], M[1], M[2] # gather three analysis window sizes from input
    N1, N2, N3 = N[0], N[1], N[2] # gather three FFT analysis sizes from input
    w1, w2, w3 = w[0], w[1], w[2] # gather three analyis windows from input
    B1, B2 = min(B), max(B) # gather frequency band edges from input
    if B1 == B2 or B1 <= 0 or B2 >= fs/2: # check validity of frequency band edges
        raise ValueError("Frequency edge inputs are not valid to separate [0, fs/2] into 3 bands.")
    hM1_1 = int(math.floor((w1.size + 1) / 2))  # half analysis window size by rounding
    hM2_1 = int(math.floor(w1.size / 2))  # half analysis window size by floor
    hM1_2 = int(math.floor((w2.size + 1) / 2))  # half analysis window size by rounding
    hM2_2 = int(math.floor(w2.size / 2))  # half analysis window size by floor
    hM1_3 = int(math.floor((w3.size + 1) / 2))  # half analysis window size by rounding
    hM2_3 = int(math.floor(w3.size / 2))  # half analysis window size by floor
    Ns = 512  # FFT size for synthesis (even)
    H = Ns // 4  # Hop size used for analysis and synthesis
    hNs = Ns // 2  # half of synthesis FFT size
    pin1 = max(hNs, hM1_1)  # init sound pointer in middle of anal window
    pend1 = x.size - max(hNs, hM1_1)  # last sample to start a frame
    pin2 = max(hNs, hM1_2)  # init sound pointer in middle of anal window
    pend2 = x.size - max(hNs, hM1_2)  # last sample to start a frame
    pin3 = max(hNs, hM1_3)  # init sound pointer in middle of anal window
    pend3 = x.size - max(hNs, hM1_3)  # last sample to start a frame
    pin = max(pin1, pin2, pin3)
    pend = min(pend1, pend2, pend3)
    yw = np.zeros(Ns)  # initialize output sound frame
    y = np.zeros(x.size)  # initialize output array
    w1 = w1 / sum(w1)  # normalize analysis window
    w2 = w2 / sum(w2)
    w3 = w3 / sum(w3)
    sw = np.zeros(Ns)  # initialize synthesis window
    ow = triang(2 * H)  # triangular window
    sw[hNs - H:hNs + H] = ow  # add triangular window
    bh = blackmanharris(Ns)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window
    while pin < pend:  # while input sound pointer is within sound

        # -----analysis-----
        x1 = x[pin1 - hM1_1:pin1 + hM2_1] # select frame
        x2 = x[pin2 - hM1_2:pin2 + hM2_2]
        x3 = x[pin3 - hM1_3:pin3 + hM2_3]
        mX1, pX1 = DFT.dftAnal(x1, w1, N1) # compute DFT
        mX2, pX2 = DFT.dftAnal(x2, w2, N2)
        mX3, pX3 = DFT.dftAnal(x3, w3, N3)
        ploc1 = UF.peakDetection(mX1, t)  # detect locations of peaks
        ploc2 = UF.peakDetection(mX2, t)
        ploc3 = UF.peakDetection(mX3, t)
        iploc1, ipmag1, ipphase1 = UF.peakInterp(mX1, pX1, ploc1)  # refine peak values by interpolation
        iploc2, ipmag2, ipphase2 = UF.peakInterp(mX2, pX2, ploc2)
        iploc3, ipmag3, ipphase3 = UF.peakInterp(mX3, pX3, ploc3)
        ipfreq1 = fs * iploc1 / float(N1)  # convert peak locations to Hertz
        ipfreq2 = fs * iploc2 / float(N2)
        ipfreq3 = fs * iploc3 / float(N3)

        # -----synthesis-----
        band1 = np.where(ipfreq1 < B1) # use peaks from mX1 in freq range [0, B1)
        band2 = np.where((ipfreq2 >= B1) & (ipfreq2 < B2)) # use peaks from mX2 in freq range [B1, B2)
        band3 = np.where(ipfreq3 >= B2) # use peaks from mX3 in freq range [B2, fs/2]
        # concatenate peaks from the three bands
        ipfreq = np.concatenate((ipfreq1[band1], ipfreq2[band2], ipfreq3[band3]))
        ipmag = np.concatenate((ipmag1[band1], ipmag2[band2], ipmag3[band3]))
        ipphase = np.concatenate((ipphase1[band1], ipphase2[band2], ipphase3[band3]))
        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs) # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))  # compute inverse FFT
        yw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
        yw[hNs - 1:] = fftbuffer[:hNs + 1]
        y[pin - hNs:pin + hNs] += sw * yw  # overlap-add and apply a synthesis window
        pin += H  # advance sound pointer   
        pin1 += H
        pin2 += H
        pin3 += H

    # output sound file name
    outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sineModelMultiRes.wav'

	# write the synthesized sound obtained from the sinusoidal synthesis
    UF.wavwrite(y, fs, outputFile)   

    return y


# def sineModel(x, fs, w, N, t):
#     """
# 	Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
# 	x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
# 	returns y: output array sound
# 	"""

#     hM1 = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
#     hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
#     Ns = 512  # FFT size for synthesis (even)
#     H = Ns // 4  # Hop size used for analysis and synthesis
#     hNs = Ns // 2  # half of synthesis FFT size
#     pin = max(hNs, hM1)  # init sound pointer in middle of anal window
#     pend = x.size - max(hNs, hM1)  # last sample to start a frame
#     yw = np.zeros(Ns)  # initialize output sound frame
#     y = np.zeros(x.size)  # initialize output array
#     w = w / sum(w)  # normalize analysis window
#     sw = np.zeros(Ns)  # initialize synthesis window
#     ow = triang(2 * H)  # triangular window
#     sw[hNs - H:hNs + H] = ow  # add triangular window
#     bh = blackmanharris(Ns)  # blackmanharris window
#     bh = bh / sum(bh)  # normalized blackmanharris window
#     sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window
#     while pin < pend:  # while input sound pointer is within sound
#         # -----analysis-----
#         x1 = x[pin - hM1:pin + hM2]  # select frame
#         mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
#         ploc = UF.peakDetection(mX, t)  # detect locations of peaks
#         iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values by interpolation
#         ipfreq = fs * iploc / float(N)  # convert peak locations to Hertz
#         # -----synthesis-----
#         Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)  # generate sines in the spectrum
#         fftbuffer = np.real(ifft(Y))  # compute inverse FFT
#         yw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
#         yw[hNs - 1:] = fftbuffer[:hNs + 1]
#         y[pin - hNs:pin + hNs] += sw * yw  # overlap-add and apply a synthesis window
#         pin += H  # advance sound pointer
#     # output sound file name
#     outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sineModel.wav'

# 	# write the synthesized sound obtained from the sinusoidal synthesis
#     UF.wavwrite(y, fs, outputFile) 
#     return y


# inputFile = 'orchestra.wav'
# window = 'blackman'
# M = (4501, 1201, 501)
# N = (8192, 2048, 1024)
# B = (100, 1000)
# t = -100

# M0 = 1201
# w0 = get_window(window, M0, fftbins=(M0%2==0))
# N0 = 2048

# fs, x = UF.wavread(inputFile) # read input sound
# # set up analysis windows
# # set fftbins parameter to True if window size is even, else False
# w1 = get_window(window, M[0], fftbins=(M[0]%2==0))
# w2 = get_window(window, M[1], fftbins=(M[1]%2==0))
# w3 = get_window(window, M[2], fftbins=(M[2]%2==0))
# w = (w1, w2, w3)

# y = sineModel(x, fs, w0, N0, t)
# ymr = sineModelMultiRes(x, fs, w, M, N, B, t)

# inputFile = 'melodic_metal.wav'
# window = 'hamming'
# M = (2001, 1301, 501)
# N = (2048, 2048, 1024)
# B = (135, 2000)
# t = -80

# M0 = 1301
# w0 = get_window(window, M0, fftbins=(M0%2==0))
# N0 = 2048

# fs, x = UF.wavread(inputFile) # read input sound
# # set up analysis windows
# # set fftbins parameter to True if window size is even, else False
# w1 = get_window(window, M[0], fftbins=(M[0]%2==0))
# w2 = get_window(window, M[1], fftbins=(M[1]%2==0))
# w3 = get_window(window, M[2], fftbins=(M[2]%2==0))
# w = (w1, w2, w3)

# y = sineModel(x, fs, w0, N0, t)
# ymr = sineModelMultiRes(x, fs, w, M, N, B, t)