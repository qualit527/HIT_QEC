from timeit import timeit
from bpdecoupling import Decoder
import numpy as np

Hx = np.array([[0, 1, 0, 0, 1],
               [1, 0, 1, 0, 0],
               [0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1]], dtype=np.uint8)
Hz = np.array([[0, 0, 1, 1, 0],
               [0, 0, 0, 1, 1],
               [1, 0, 0, 0, 1],
               [1, 1, 0, 0, 0]], dtype=np.uint8)
Hy = (Hx + Hz) % 2
H_bar = np.hstack([Hx, Hz, Hy])

px = 0
pz = 0
py = 0.1
channel_error_rate1 = [pz for i in range(5)]
channel_error_rate2 = [px for i in range(5)]
channel_error_rate3 = [py for i in range(5)]
channel_error_rate = channel_error_rate1 + \
    channel_error_rate2 + channel_error_rate3
channel_error_rate = np.array(channel_error_rate)

max_iter = H_bar.shape[1]

decoder = Decoder(Decoder.Method.MIN_SUM, max_iter, H_bar, channel_error_rate)
error = np.array([0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0])
syndrome = H_bar @ error.T % 2
print(syndrome)

decoding, run_iter, converge = decoder.bpDecode(syndrome)
print(decoding, converge)
decoding, run_iter, converge = decoder.bpOsdDecode(syndrome)
print(decoding, converge)
