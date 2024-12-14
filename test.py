import numpy as np
from LLRBP4_decoder import LLRBp4Decoder, ScheduleType, InitType, MethodType, OSDType
from bpdecoupling import Decoder as FDBPDecoder, Mod2SparseMatrix

Hz = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
code_length = Hz.shape[1]
Hx = np.zeros_like(Hz)

px = 0.01
pz = 0.01
py = 0.01

max_iter = 50

error = np.array([1, 1, 0, 0, 0, 0, 0, 0])
syndrome = (Hz @ error[0:4].T + Hx @ error[4:8].T) % 2

# 解耦BP调用
Hy = (Hx + Hz) % 2
H_bar = np.hstack([Hx, Hz, Hy])
channel_error_rate1 = [pz for i in range(code_length)]
channel_error_rate2 = [px for i in range(code_length)]
channel_error_rate3 = [py for i in range(code_length)]
channel_error_rate = np.array(channel_error_rate1 + channel_error_rate2 + channel_error_rate3)
FDBPdecoder = FDBPDecoder(FDBPDecoder.Method.PRODUCT_SUM, max_iter, Mod2SparseMatrix(H_bar), channel_error_rate)

correction, iter, flag = FDBPdecoder.bpDecode(syndrome)

correction = np.array(correction)
print("Decoupled BP: ")
print(correction)
print(flag, iter)
print((Hz @ correction[0:4].T + Hx @ correction[4:8].T) % 2)

# LLRBP4调用
LLRdecoder = LLRBp4Decoder(Hx, Hz, px, py, pz, max_iter, dimension=1)

correction, flag, iter = LLRdecoder.standard_decoder(syndrome, init=InitType.MOMENTUM, schedule=ScheduleType.LAYER, alpha=0.5)

correction = np.array(correction)
print("EWA-BP: ")
print(correction)
print(flag, iter)
print((Hz @ correction[0:4].T + Hx @ correction[4:8].T) % 2)
