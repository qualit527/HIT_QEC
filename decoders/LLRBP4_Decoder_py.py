import numpy as np
import galois
import time
import matplotlib.pyplot as plt

class LLRBp4Decoder:
    def __init__(self, Hx, Hz, px, py, pz, max_iter, Hs=None, ps=None, dimension=1):
        """
        Hx: X 错误的校验矩阵
        Hz: Z 错误的校验矩阵
        px: X 错误的概率
        py: Y 错误的概率
        pz: Z 错误的概率
        max_iter: 最大迭代次数
        Hs: （可选）综合症校验矩阵（适用于带测量噪声）
        ps: （可选）测量噪声的概率
        dimension: 维度（默认1）
        """
        # 这里的Hx和Hz的维度是相同的
        self.Hx = Hx
        self.Hz = Hz
        self.Hs = Hs
        self.H = Hx + Hz
        self.m = self.H.shape[0]
        self.n = self.H.shape[1]
        self.s = self.Hs.shape[1] if Hs is not None else 0
        self.px = px
        self.py = py
        self.pz = pz
        self.ps = ps
        self.pi = 1 - px - py - pz
        self.k = dimension
        self.eta = 0

        self.max_iter = max_iter
        self.flag = True

        self.decoupled_H = np.zeros([self.m, 3 * self.n], dtype=np.uint8)  # Z-Y-X
        for j in range(self.n):
            self.decoupled_H[:, j] = (Hx[:, j] == 0) & (Hz[:, j] != 0)  # Hz
            self.decoupled_H[:, j + 2*self.n] = (Hx[:, j] != 0) & (Hz[:, j] == 0)  # Hx
            self.decoupled_H[:, j + self.n] = (self.decoupled_H[:, j] + self.decoupled_H[:, j + 2*self.n]) % 2  # Hy
        
        self.binary_H = np.zeros([self.m, 2 * self.n + self.s], dtype=np.uint8)  # Z-X
        for j in range(self.n):
            self.binary_H[:, j] = (Hz[:, j] != 0)
            self.binary_H[:, j + self.n] = (Hx[:, j] != 0)
        if Hs is not None:
            for j in range(self.s):
                self.binary_H[:, j + 2*self.n] = (Hs[:, j] != 0)

        self.Q_matrix_X = np.zeros(self.H.shape)
        self.Q_matrix_Y = np.zeros(self.H.shape)
        self.Q_matrix_Z = np.zeros(self.H.shape)

        # 错误概率初始化
        self.Q_matrix_X[self.H != 0] = np.log(self.pi / self.px)
        self.Q_matrix_Y[self.H != 0] = np.log(self.pi / self.py)
        self.Q_matrix_Z[self.H != 0] = np.log(self.pi / self.pz)

        if Hs is not None:
            self.Q_matrix_S = np.zeros(self.Hs.shape)
            self.Q_matrix_S[self.Hs != 0] = np.log(self.pi / self.ps)

        # 标量消息初始化
        self.d_message = np.zeros(self.H.shape)
        self.d_message_ds = np.zeros((self.m, self.n + self.s))

        # 使用逻辑数组索引
        mask_X = (self.Hx == 1) & (self.Hz != 1)
        mask_Z = (self.Hx != 1) & (self.Hz == 1)
        mask_Y = (self.Hx == 1) & (self.Hz == 1)

        self.d_message[mask_X] = self.lambda_func("X", self.Q_matrix_X[mask_X], self.Q_matrix_Y[mask_X], self.Q_matrix_Z[mask_X])
        self.d_message[mask_Z] = self.lambda_func("Z", self.Q_matrix_X[mask_Z], self.Q_matrix_Y[mask_Z], self.Q_matrix_Z[mask_Z])
        self.d_message[mask_Y] = self.lambda_func("Y", self.Q_matrix_X[mask_Y], self.Q_matrix_Y[mask_Y], self.Q_matrix_Z[mask_Y])

        self.d_message_ds[:, :self.n] = self.d_message
        if Hs is not None:
            self.d_message_ds[:, self.n:] = self.Q_matrix_S

        self.delta_message = np.zeros(self.H.shape)
        self.delta_message_ds = np.zeros((self.m, self.n + self.s))

        self.energy_history = []

    
    def lambda_func(self, W, px, py, pz):
        if W == "X":
            return np.log((1 + np.exp(-px)) / (np.exp(-py) + np.exp(-pz)))
        elif W == "Y":
            return np.log((1 + np.exp(-py)) / (np.exp(-px) + np.exp(-pz)))
        elif W == "Z":
            return np.log((1 + np.exp(-pz)) / (np.exp(-px) + np.exp(-py)))
        else:
            raise ValueError(f"Invalid W value: {W}. Expected 'X', 'Y', or 'Z'.")
        

    def compute_energy(self, qX, qY, qZ, qX_0, qY_0, qZ_0, syndrome):
        J_D = 0.5 * np.linalg.norm(np.array([qX, qY, qZ]) - np.array([qX_0, qY_0, qZ_0]))**2
        
        J_S = 0
        for m in range(self.m):
            W = None
            index = np.where(self.H[m, :] != 0)[0]
            z_m = syndrome[m]
            tanh_values = []
            
            for n in index:
                if self.Hx[m, n] == 1 and self.Hz[m, n] == 0:
                    W = "X"
                elif self.Hx[m, n] == 1 and self.Hz[m, n] == 1:
                    W = "Y"
                elif self.Hx[m, n] == 0 and self.Hz[m, n] == 1:
                    W = "Z"

            tanh_value = np.tanh(self.lambda_func(W, qX[n], qY[n], qZ[n]) / 2)
            tanh_values.append(tanh_value)
            product_term = np.prod(tanh_values)
            J_S -= 2 * np.arctanh((-1)**z_m * product_term)
        
        energy = self.eta * J_D + J_S
        return energy

    
    def standard_decoder(self, syndrome, schedule="flooding", init="None", method="None", OSD="None", OTS=False, T=5, C=10, alpha=1, beta=0, test=0):
        d_message = np.copy(self.d_message)
        delta_message = np.copy(self.delta_message)

        last_Error = np.zeros(2 * self.n).astype(np.uint8)
        reliability = np.ones(self.n)  # 每个比特的历史可靠性

        qX = np.zeros(self.n)  # 每个比特发生X错误的概率
        qY = np.zeros(self.n)  # 每个比特发生Y错误的概率
        qZ = np.zeros(self.n)  # 每个比特发生Z错误的概率

        horizontal_time = 0
        vertical_time = 0

        # 第0次软判决向量
        qX_0 = np.full(self.n, np.log(self.pi / self.px))
        qY_0 = np.full(self.n, np.log(self.pi / self.py))
        qZ_0 = np.full(self.n, np.log(self.pi / self.pz))

        last_qX = np.zeros(self.n)
        last_qY = np.zeros(self.n)
        last_qZ = np.zeros(self.n)

        # if test == 1:
        #     last_qX = np.copy(qX_0)
        #     last_qY = np.copy(qY_0)
        #     last_qZ = np.copy(qZ_0)

        gt = np.zeros((self.n, 3))
        Vt = np.zeros((self.n, 3))

        last_qX_q = np.copy(qX_0)
        last_qY_q = np.copy(qY_0)
        last_qZ_q = np.copy(qZ_0)
        gt_q = np.zeros((self.n, 3))
        Vt_q = np.zeros((self.n, 3))

        oscillation = np.zeros(self.n)  # 振荡向量
        OTS_times = 0

        llr_history = {}

        for i in range(self.max_iter):
            Error = np.zeros(2 * self.n).astype(np.uint8)  # Error = [Ex,Ez]

            if schedule == "flooding":
                # 每一行进行更新 水平更新
                start_time = time.time()
                for j in range(self.m):
                    index = np.where(self.H[j, :] != 0)[0]
                    
                    tanh_values = np.tanh(d_message[j, index] / 2) + 1e-10
                    product_term = np.prod(tanh_values)

                    for k in range(len(index)):
                        product_excluding_current = product_term / tanh_values[k]
                        product_excluding_current = np.clip(product_excluding_current, -1+1e-10, 1-1e-10)
                        delta_value = 2 * np.arctanh(product_excluding_current)
                        delta_message[j, index[k]] = (-1)**syndrome[j] * delta_value

                horizontal_time += time.time() - start_time

                # 每一列进行更新 垂直更新
                start_time = time.time()
                for j in range(self.n):
                    index1 = np.where(self.H[:, j] != 0)[0]
                    qX[j] = qX_0[j]
                    qY[j] = qY_0[j]
                    qZ[j] = qZ_0[j]

                    if init == "Momentum":
                        if i > 0:
                        # gt[j, 0] = last_qX[j] - qX[j]
                        # gt[j, 1] = last_qY[j] - qY[j]
                        # gt[j, 2] = last_qZ[j] - qZ[j]

                        # qX[j] = last_qX[j] - alpha * gt[j, 0]
                        # qY[j] = last_qY[j] - alpha * gt[j, 1]
                        # qZ[j] = last_qZ[j] - alpha * gt[j, 2]

                            qX[j] = alpha * qX[j] + (1 - alpha) * last_qX[j]
                            qY[j] = alpha * qY[j] + (1 - alpha) * last_qY[j]
                            qZ[j] = alpha * qZ[j] + (1 - alpha) * last_qZ[j]

                        
                        # if j == 0:
                            # print(f"iter: {i+1}, qX0: {qX[j]}, qY0: {qY[j]}, qZ0: {qZ[j]}, last_qX0: {last_qX[j]}, last_qY0: {last_qY[j]}, last_qZ0: {last_qZ[j]}")

                        last_qX, last_qY, last_qZ = qX.copy(), qY.copy(), qZ.copy()

                    elif init == "Ada":
                        # AdaGrad
                        if test == 5:
                            gt[j, 0] = last_qX[j] - last_qX_q[j]
                            gt[j, 1] = last_qY[j] - last_qY_q[j]
                            gt[j, 2] = last_qZ[j] - last_qZ_q[j]

                        else:
                            gt[j, 0] = last_qX[j] - qX[j]
                            gt[j, 1] = last_qY[j] - qY[j]
                            gt[j, 2] = last_qZ[j] - qZ[j]

                        Vt[j] += np.square(gt[j])

                        if i > 0:
                            if test == 5:
                                qX[j] = last_qX_q[j] + (alpha * gt[j, 0]) / (np.sqrt(Vt[j, 0]) + 1e-10)
                                qY[j] = last_qY_q[j] + (alpha * gt[j, 1]) / (np.sqrt(Vt[j, 1]) + 1e-10)
                                qZ[j] = last_qZ_q[j] + (alpha * gt[j, 2]) / (np.sqrt(Vt[j, 2]) + 1e-10)
                            else:
                                qX[j] = qX[j] + (alpha * gt[j, 0]) / (np.sqrt(Vt[j, 0]) + 1e-10)
                                qY[j] = qY[j] + (alpha * gt[j, 1]) / (np.sqrt(Vt[j, 1]) + 1e-10)
                                qZ[j] = qZ[j] + (alpha * gt[j, 2]) / (np.sqrt(Vt[j, 2]) + 1e-10)

                        last_qX_q[j] = qX[j]
                        last_qY_q[j] = qY[j]
                        last_qZ_q[j] = qZ[j]
                        last_qX, last_qY, last_qZ = qX.copy(), qY.copy(), qZ.copy()
                    
                    elif init != "None":
                        raise ValueError(f"Invalid init method: {init}. Expected 'Momentum' or 'Ada'.")

                    # 软判决向量
                    # 使用向量化操作计算 qX, qY 和 qZ 的总和
                    qX_summands = np.sum(((self.Hz[index1, j] == 1)) * delta_message[index1, j])
                    qY_summands = np.sum(((self.Hx[index1, j] != self.Hz[index1, j])) * delta_message[index1, j])
                    qZ_summands = np.sum(((self.Hx[index1, j] == 1)) * delta_message[index1, j])

                    qX[j] += qX_summands
                    qY[j] += qY_summands
                    qZ[j] += qZ_summands

                    if method == "Momentum":
                        # 更新动量项
                        gt_q[j, 0] = beta * gt_q[j, 0] + (1-beta) * (last_qX_q[j] - qX[j])
                        gt_q[j, 1] = beta * gt_q[j, 1] + (1-beta) * (last_qY_q[j] - qY[j])
                        gt_q[j, 2] = beta * gt_q[j, 2] + (1-beta) * (last_qZ_q[j] - qZ[j])

                        qX[j] = last_qX_q[j] - alpha * gt_q[j, 0]
                        qY[j] = last_qY_q[j] - alpha * gt_q[j, 1]
                        qZ[j] = last_qZ_q[j] - alpha * gt_q[j, 2]

                        last_qX_q, last_qY_q, last_qZ_q = qX.copy(), qY.copy(), qZ.copy()

                    elif method == "Ada":
                        # AdaGrad
                        gt_q[j, 0] = last_qX_q[j] - qX[j]
                        gt_q[j, 1] = last_qY_q[j] - qY[j]
                        gt_q[j, 2] = last_qZ_q[j] - qZ[j]

                        Vt[j] += np.square(gt_q[j])
                        
                        if i != 0:
                            qX[j] = last_qX_q[j] - (alpha * gt_q[j, 0]) / (np.sqrt(Vt[j, 0]) + 1e-10)
                            qY[j] = last_qY_q[j] - (alpha * gt_q[j, 1]) / (np.sqrt(Vt[j, 1]) + 1e-10)
                            qZ[j] = last_qZ_q[j] - (alpha * gt_q[j, 2]) / (np.sqrt(Vt[j, 2]) + 1e-10)

                        last_qX_q, last_qY_q, last_qZ_q = qX.copy(), qY.copy(), qZ.copy()

                    elif method == "MBP":
                        qX[j] = qX[j] - np.sum(qX_summands) + np.sum(qX_summands) / alpha
                        qY[j] = qY[j] - np.sum(qY_summands) + np.sum(qY_summands) / alpha
                        qZ[j] = qZ[j] - np.sum(qZ_summands) + np.sum(qZ_summands) / alpha

                    elif method != "None":
                        raise ValueError(f"Invalid method: {method}. Expected 'Momentum' or 'Ada'.")

                    # 垂直更新消息
                    # 使用向量化操作更新 d_message
                    mask_X = (self.Hx[index1, j] == 1) & (self.Hz[index1, j] == 0)
                    mask_Y = (self.Hx[index1, j] == 1) & (self.Hz[index1, j] == 1)
                    mask_Z = (self.Hx[index1, j] == 0) & (self.Hz[index1, j] == 1)

                    d_message[index1[mask_X], j] = self.lambda_func("X", qX[j], qY[j] - delta_message[index1[mask_X], j], qZ[j] - delta_message[index1[mask_X], j])
                    d_message[index1[mask_Y], j] = self.lambda_func("Y", qX[j] - delta_message[index1[mask_Y], j], qY[j], qZ[j] - delta_message[index1[mask_Y], j])
                    d_message[index1[mask_Z], j] = self.lambda_func("Z", qX[j] - delta_message[index1[mask_Z], j], qY[j] - delta_message[index1[mask_Z], j], qZ[j])
                    
                    # if j == 0:
                        # print(f"iter: {i+1}, qX: {qX[j]}, qY: {qY[j]}, qZ: {qZ[j]}")

                    # hard decision 计算每一个比特上的出错概率
                    list = [qX[j], qY[j], qZ[j]]
                    min_value = min(list)
                    indx = list.index(min_value)

                    if qX[j] > 0 and qY[j] > 0 and qZ[j] > 0:
                        Error[j] = 0
                        Error[j + self.n] = 0
                    elif indx == 0:    # 发生X错误
                        Error[j] = 1
                        Error[j + self.n] = 0
                    elif indx == 1:    # 发生Y错误
                        Error[j] = 1
                        Error[j + self.n] = 1
                    elif indx == 2:    # 发生Z错误
                        Error[j] = 0
                        Error[j + self.n] = 1

                    if last_Error[j] == Error[j] & last_Error[j + self.n] == Error[j + self.n]:
                        reliability[j] += 1
                    else:
                        reliability[j] == 1
                    
                vertical_time += time.time() - start_time

                last_Error = np.copy(Error)

            elif schedule == "layer":
                # 变量节点层级调度
                for j in range(self.n):
                    index1 = np.where(self.H[:, j] != 0)[0]

                    # 水平更新
                    start_time = time.time()
                    for m in index1:
                        index_m = np.where(self.H[m, :] != 0)[0]
                        index_m = index_m[index_m != j]
                        tanh_values = np.tanh(d_message[m, index_m] / 2) + 1e-10
                        product_term = np.clip(np.prod(tanh_values), -1+1e-10, 1-1e-10)
                        
                        delta_value = 2 * np.arctanh(product_term)
                        delta_message[m, j] = (-1)**syndrome[m] * delta_value
                    
                    horizontal_time += time.time() - start_time

                    start_time = time.time()
                    
                    qX[j] = qX_0[j]
                    qY[j] = qY_0[j]
                    qZ[j] = qZ_0[j]

                    if init == "Momentum":
                        # 更新动量项
                        gt[j, 0] = beta * gt[j, 0] + (1-beta) * (last_qX[j] - qX[j])
                        gt[j, 1] = beta * gt[j, 1] + (1-beta) * (last_qY[j] - qY[j])
                        gt[j, 2] = beta * gt[j, 2] + (1-beta) * (last_qZ[j] - qZ[j])

                        qX[j] = last_qX[j] - alpha * gt[j, 0]
                        qY[j] = last_qY[j] - alpha * gt[j, 1]
                        qZ[j] = last_qZ[j] - alpha * gt[j, 2]

                        last_qX, last_qY, last_qZ = qX.copy(), qY.copy(), qZ.copy()

                    elif init == "Ada":
                        # AdaGrad
                        gt[j, 0] = last_qX[j] - qX[j]
                        gt[j, 1] = last_qY[j] - qY[j]
                        gt[j, 2] = last_qZ[j] - qZ[j]

                        Vt[j] += np.square(gt[j])

                        if test == 1:
                            if i > 0:
                                qX[j] = last_qX[j] - (alpha * gt[j, 0]) / (np.sqrt(Vt[j, 0]) + 1e-10)
                                qY[j] = last_qY[j] - (alpha * gt[j, 1]) / (np.sqrt(Vt[j, 1]) + 1e-10)
                                qZ[j] = last_qZ[j] - (alpha * gt[j, 2]) / (np.sqrt(Vt[j, 2]) + 1e-10)

                        elif i > 0:
                            qX[j] = qX[j] + (alpha * gt[j, 0]) / (np.sqrt(Vt[j, 0]) + 1e-10)
                            qY[j] = qY[j] + (alpha * gt[j, 1]) / (np.sqrt(Vt[j, 1]) + 1e-10)
                            qZ[j] = qZ[j] + (alpha * gt[j, 2]) / (np.sqrt(Vt[j, 2]) + 1e-10)

                        last_qX, last_qY, last_qZ = qX.copy(), qY.copy(), qZ.copy()
                    
                    elif init != "None":
                        raise ValueError(f"Invalid init method: {init}. Expected 'Momentum' or 'Ada'.")

                    # 软判决向量
                    qX_summands = ((self.Hz[index1, j] == 1)) * delta_message[index1, j]
                    qY_summands = ((self.Hx[index1, j] != self.Hz[index1, j])) * delta_message[index1, j]
                    qZ_summands = ((self.Hx[index1, j] == 1)) * delta_message[index1, j]

                    qX[j] += np.sum(qX_summands)
                    qY[j] += np.sum(qY_summands)
                    qZ[j] += np.sum(qZ_summands)

                    if method == "Momentum":
                        # 更新动量项
                        gt_q[j, 0] = beta * gt_q[j, 0] + (1-beta) * (last_qX_q[j] - qX[j])
                        gt_q[j, 1] = beta * gt_q[j, 1] + (1-beta) * (last_qY_q[j] - qY[j])
                        gt_q[j, 2] = beta * gt_q[j, 2] + (1-beta) * (last_qZ_q[j] - qZ[j])

                        qX[j] = last_qX_q[j] - alpha * gt_q[j, 0]
                        qY[j] = last_qY_q[j] - alpha * gt_q[j, 1]
                        qZ[j] = last_qZ_q[j] - alpha * gt_q[j, 2]

                        last_qX_q, last_qY_q, last_qZ_q = qX.copy(), qY.copy(), qZ.copy()

                    elif method == "Ada":
                        # AdaGrad
                        gt_q[j, 0] = last_qX_q[j] - qX[j]
                        gt_q[j, 1] = last_qY_q[j] - qY[j]
                        gt_q[j, 2] = last_qZ_q[j] - qZ[j]

                        Vt_q[j] += np.square(gt_q[j])
                        
                        if i != 0:
                            qX[j] = last_qX_q[j] - (alpha * gt_q[j, 0]) / (np.sqrt(Vt_q[j, 0]) + 1e-10)
                            qY[j] = last_qY_q[j] - (alpha * gt_q[j, 1]) / (np.sqrt(Vt_q[j, 1]) + 1e-10)
                            qZ[j] = last_qZ_q[j] - (alpha * gt_q[j, 2]) / (np.sqrt(Vt_q[j, 2]) + 1e-10)
                        
                        last_qX_q, last_qY_q, last_qZ_q = qX.copy(), qY.copy(), qZ.copy()
                    
                    elif method == "MBP":
                        qX[j] = qX[j] - np.sum(qX_summands) + np.sum(qX_summands) / alpha
                        qY[j] = qY[j] - np.sum(qY_summands) + np.sum(qY_summands) / alpha
                        qZ[j] = qZ[j] - np.sum(qZ_summands) + np.sum(qZ_summands) / alpha

                    elif method != "None":
                        raise ValueError(f"Invalid method: {method}. Expected 'Momentum' or 'Ada'.")

                    # 垂直更新
                    for m in index1:
                        if (self.Hx[m, j] == 1) & (self.Hz[m, j] == 0):
                            d_message[m, j] = self.lambda_func("X", qX[j], qY[j] - delta_message[m, j], qZ[j] - delta_message[m, j])
                        elif (self.Hx[m, j] == 1) & (self.Hz[m, j] == 1):
                            d_message[m, j] = self.lambda_func("Y", qX[j] - delta_message[m, j], qY[j], qZ[j] - delta_message[m, j])
                        elif (self.Hx[m, j] == 0) & (self.Hz[m, j] == 1):
                            d_message[m, j] = self.lambda_func("Z", qX[j] - delta_message[m, j], qY[j] - delta_message[m, j], qZ[j])

                    vertical_time += time.time() - start_time

                start_time = time.time()
                # hard decision 计算每一个比特上的出错概率
                for j in range(self.n):
                    list = [qX[j], qY[j], qZ[j]]
                    min_value = min(list)
                    indx = list.index(min_value)

                    if qX[j] > 0 and qY[j] > 0 and qZ[j] > 0:
                        Error[j] = 0
                        Error[j + self.n] = 0
                    elif indx == 0:    # 发生X错误
                        Error[j] = 1
                        Error[j + self.n] = 0
                    elif indx == 1:    # 发生Y错误
                        Error[j] = 1
                        Error[j + self.n] = 1
                    elif indx == 2:    # 发生Z错误
                        Error[j] = 0
                        Error[j + self.n] = 1

                    if last_Error[j] == Error[j] and last_Error[j + self.n] == Error[j + self.n]:
                        reliability[j] += 1
                    else:
                        oscillation[j] += 1
                        reliability[j] == 1
    
                last_Error = np.copy(Error)
                vertical_time += time.time() - start_time
            
            else:
                raise ValueError(f"Invalid schedule: {schedule}. Expected 'flooding' or 'layer'.")
            
            llr_history[i] = {"qX": qX.copy(), "qY": qY.copy(), "qZ": qZ.copy()}

            if ((self.Hz @ Error[0:self.n].T + self.Hx @ Error[self.n:2*self.n].T) % 2 == syndrome).all():
                self.flag = True
                # print(f"OTS_times: {OTS_times}") if OTS_times > 0 else None
                # print(f"iter: {i+1}, error: {Error}")
                return Error, self.flag, i+1
            
            # print(f"iter: {i+1}, error: {Error}")
            
            # OTS改变先验概率
            if OTS == True and (i+1) % T == 0:
                qX_0 = np.full(self.n, np.log(self.pi / self.px))
                qY_0 = np.full(self.n, np.log(self.pi / self.py))
                qZ_0 = np.full(self.n, np.log(self.pi / self.pz))

                OTS_times += 1
                # print(f"Running OTS at iter {i+1}")

                min_value = np.inf
                min_index = None

                if max(oscillation > 0):
                    max_indices = np.where(oscillation == np.max(oscillation))[0]
                    for j in max_indices:
                        values = [qX[j], qY[j], qZ[j]]
                        min_value_j = min(values)
                        if min_value_j < min_value:
                            min_value = min_value_j
                            min_index = j

                    oscillation[min_index] = 0
                    if min_value_j == qX[j]:
                        qX_0[j] = -C
                        # print(f"Biasing variable node {min_index}, qX_0 to -{C}")
                    elif min_value_j == qY[j]:
                        qY_0[j] = -C
                        # print(f"Biasing variable node {min_index}, qY_0 to -{C}")
                    elif min_value_j == qZ[j]:
                        qZ_0[j] = -C
                        # print(f"Biasing variable node {min_index}, qZ_0 to -{C}")

                min_value = np.inf
                min_index = None    
                for j in range(self.n):
                    values = [qX[j], qY[j], qZ[j]]
                    min_value_j = min(values)
                    if min_value_j < min_value:
                        min_value = min_value_j
                        min_index = j

                if min_value_j == qX[j]:
                    qX_0[j] = -C
                    # print(f"Biasing variable node {min_index}, qX_0 to -{C}")
                elif min_value_j == qY[j]:
                    qY_0[j] = -C
                    # print(f"Biasing variable node {min_index}, qY_0 to -{C}")
                elif min_value_j == qZ[j]:
                    qZ_0[j] = -C
                    # print(f"Biasing variable node {min_index}, qZ_0 to -{C}")
                if test == 3:
                    C = C + 10
            

        if OSD == "decoupled":
            probability = np.hstack((qX, qY, qZ))
            Error = osd0_post_processing(syndrome, self.decoupled_H, reliability, probability, self.k)

        elif OSD == "binary":
            px = 1 / (np.exp(qX) + 1)            
            py = 1 / (np.exp(qY) + 1)            
            pz = 1 / (np.exp(qZ) + 1)

            # binary_X = np.log((1 - px - py) / (px + py))
            # binary_Z = np.log((1 - pz - py) / (pz + py))
            binary_X = px + py
            binary_Z = pz + py

            probability = np.hstack((-binary_X, -binary_Z))

            Error = binary_osd(syndrome, self.binary_H, reliability, probability, self.k)
        
        elif OSD == "b_r":
            px = 1 / (np.exp(qX) + 1)            
            py = 1 / (np.exp(qY) + 1)            
            pz = 1 / (np.exp(qZ) + 1)

            binary_X = px + py
            binary_Z = pz + py
            probability = np.hstack((binary_X, binary_Z))

            Error = b_r_osd(syndrome, self.binary_H, reliability, probability, self.k)

        # print(f"OTS_times: {OTS_times}") if OTS_times > 0 else None

        return Error, False, self.max_iter


    def phenomenological_decoder(self, syndrome, schedule="flooding", init="None", method="None", OSD="None", alpha=1, beta=0):
        d_message_ds = np.copy(self.d_message_ds)
        delta_message_ds = np.copy(self.delta_message_ds)

        last_Error = np.zeros(2 * self.n).astype(np.uint8)
        reliability = np.ones(self.n)  # 每个比特的历史可靠性

        qX = np.zeros(self.n)  # 每个比特发生X错误的概率
        qY = np.zeros(self.n)  # 每个比特发生Y错误的概率
        qZ = np.zeros(self.n)  # 每个比特发生Z错误的概率
        qS = np.zeros(self.s)  # 稳定子测量发生错误的概率

        horizontal_time = 0
        vertical_time = 0

        # 第0次软判决向量
        qX_0 = np.full(self.n, np.log(self.pi / self.px))
        qY_0 = np.full(self.n, np.log(self.pi / self.py))
        qZ_0 = np.full(self.n, np.log(self.pi / self.pz))
        qS_0 = np.full(self.s, np.log(self.pi / self.ps))

        last_qX = np.zeros(self.n)
        last_qY = np.zeros(self.n)
        last_qZ = np.zeros(self.n)
        last_qS = np.zeros(self.s)

        gt = np.zeros((self.n, 3))
        Vt = np.zeros((self.n, 3))
        gt_S = np.zeros(self.s)
        Vt_S = np.zeros(self.s)

        last_qX_q = np.copy(qX_0)
        last_qY_q = np.copy(qY_0)
        last_qZ_q = np.copy(qZ_0)
        last_qS_q = np.copy(qS_0)

        gt_q = np.zeros((self.n, 3))
        Vt_q = np.zeros((self.n, 3))
        gt_S_q = np.zeros(self.s)
        Vt_S_q = np.zeros(self.s)

        oscillation = np.zeros(self.n)  # 振荡向量

        llr_history = {}

        for i in range(self.max_iter):
            Error = np.zeros(2 * self.n + self.s).astype(np.uint8)  # Error = [Ex,Ez]

            if schedule == "flooding":
                # 每一行进行更新 水平更新
                start_time = time.time()
                for j in range(self.m):
                    index = np.where(self.H[j, :] != 0)[0]
                    index = np.concatenate((index, np.where(self.Hs[j, :] != 0)[0] + self.n))
                    tanh_values = np.tanh(d_message_ds[j, index] / 2) + 1e-10
                    product_term = np.prod(tanh_values)

                    for k in range(len(index)):
                        product_excluding_current = product_term / tanh_values[k]
                        product_excluding_current = np.clip(product_excluding_current, -1+1e-10, 1-1e-10)
                        delta_value = 2 * np.arctanh(product_excluding_current)
                        delta_message_ds[j, index[k]] = (-1)**syndrome[j] * delta_value

                horizontal_time += time.time() - start_time

                # 每一列进行更新 垂直更新
                start_time = time.time()
                for j in range(self.n + self.s):
                    if j < self.n:
                        index1 = np.where(self.H[:, j] != 0)[0]
                    else:
                        index1 = np.where(self.Hs[:, j - self.n] != 0)[0]

                    if j < self.n:
                        qX[j] = qX_0[j]
                        qY[j] = qY_0[j]
                        qZ[j] = qZ_0[j]                   
                    else:
                        qS[j - self.n] = qS_0[j - self.n]

                    if init == "Momentum":
                        if j < self.n:
                            gt[j, 0] = last_qX[j] - qX[j]
                            gt[j, 1] = last_qY[j] - qY[j]
                            gt[j, 2] = last_qZ[j] - qZ[j]

                            qX[j] = last_qX[j] - alpha * gt[j, 0]
                            qY[j] = last_qY[j] - alpha * gt[j, 1]
                            qZ[j] = last_qZ[j] - alpha * gt[j, 2]
                        else:
                            gt_S[j - self.n] = last_qS[j - self.n] - qS[j - self.n]
                            qS[j - self.n] = last_qS[j - self.n] - alpha * gt_S[j - self.n]

                        last_qX, last_qY, last_qZ, last_qS = qX.copy(), qY.copy(), qZ.copy(), qS.copy()
                    
                    elif init != "None":
                        raise ValueError(f"Invalid init method: {init}. Expected 'Momentum'.")

                    # 软判决向量
                    # 使用向量化操作计算 qX, qY 和 qZ 的总和
                    if j < self.n:
                        qX_summands = ((self.Hz[index1, j] == 1)) * delta_message_ds[index1, j]
                        qY_summands = ((self.Hx[index1, j] != self.Hz[index1, j])) * delta_message_ds[index1, j]
                        qZ_summands = ((self.Hx[index1, j] == 1)) * delta_message_ds[index1, j]
                        qX[j] += np.sum(qX_summands)
                        qY[j] += np.sum(qY_summands)
                        qZ[j] += np.sum(qZ_summands)
                    else:
                        qS_summands = ((self.Hs[index1, j - self.n] == 1)) * delta_message_ds[index1, j]
                        qS[j - self.n] += np.sum(qS_summands)

                    if method == "Momentum":
                        # 更新动量项
                        if j < self.n:
                            gt_q[j, 0] = beta * gt_q[j, 0] + (1-beta) * (last_qX_q[j] - qX[j])
                            gt_q[j, 1] = beta * gt_q[j, 1] + (1-beta) * (last_qY_q[j] - qY[j])
                            gt_q[j, 2] = beta * gt_q[j, 2] + (1-beta) * (last_qZ_q[j] - qZ[j])

                            qX[j] = last_qX_q[j] - alpha * gt_q[j, 0]
                            qY[j] = last_qY_q[j] - alpha * gt_q[j, 1]
                            qZ[j] = last_qZ_q[j] - alpha * gt_q[j, 2]
                        else:
                            gt_S_q[j - self.n] = beta * gt_S_q[j - self.n] + (1-beta) * (last_qS_q[j - self.n] - qS[j - self.n])
                            qS[j - self.n] = last_qS_q[j - self.n] - alpha * gt_S_q[j - self.n]

                        last_qX_q, last_qY_q, last_qZ_q, last_qS_q = qX.copy(), qY.copy(), qZ.copy(), qS.copy()

                    elif method == "Ada":
                        # AdaGrad
                        if j < self.n:
                            gt_q[j, 0] = last_qX_q[j] - qX[j]
                            gt_q[j, 1] = last_qY_q[j] - qY[j]
                            gt_q[j, 2] = last_qZ_q[j] - qZ[j]

                            Vt[j] += np.square(gt_q[j])

                            if i != 0:
                                qX[j] = last_qX_q[j] - (alpha * gt_q[j, 0]) / (np.sqrt(Vt[j, 0]) + 1e-10)
                                qY[j] = last_qY_q[j] - (alpha * gt_q[j, 1]) / (np.sqrt(Vt[j, 1]) + 1e-10)
                                qZ[j] = last_qZ_q[j] - (alpha * gt_q[j, 2]) / (np.sqrt(Vt[j, 2]) + 1e-10)
                        
                        else:
                            gt_S_q[j - self.n] = last_qS_q[j - self.n] - qS[j - self.n]
                            Vt_S_q[j - self.n] += np.square(gt_S_q[j - self.n])

                            if i != 0:
                                qS[j - self.n] = last_qS_q[j - self.n] - (alpha * gt_S_q[j - self.n]) / (np.sqrt(Vt_S_q[j - self.n]) + 1e-10)

                        last_qX_q, last_qY_q, last_qZ_q, last_qS_q = qX.copy(), qY.copy(), qZ.copy(), qS.copy()

                    elif method == "MBP":
                        if j < self.n:
                            qX[j] = qX[j] - np.sum(qX_summands) + np.sum(qX_summands) / alpha
                            qY[j] = qY[j] - np.sum(qY_summands) + np.sum(qY_summands) / alpha
                            qZ[j] = qZ[j] - np.sum(qZ_summands) + np.sum(qZ_summands) / alpha
                        else:
                            qS[j - self.n] = qS[j - self.n] - np.sum(qS_summands) + np.sum(qS_summands) / alpha

                    elif method != "None":
                        raise ValueError(f"Invalid method: {method}. Expected 'Momentum' or 'Ada'.")


                    # 垂直更新消息
                    if j < self.n:
                        # 使用向量化操作更新 d_message_ds
                        mask_X = (self.Hx[index1, j] == 1) & (self.Hz[index1, j] == 0)
                        mask_Y = (self.Hx[index1, j] == 1) & (self.Hz[index1, j] == 1)
                        mask_Z = (self.Hx[index1, j] == 0) & (self.Hz[index1, j] == 1)

                        d_message_ds[index1[mask_X], j] = self.lambda_func("X", qX[j], qY[j] - delta_message_ds[index1[mask_X], j], qZ[j] - delta_message_ds[index1[mask_X], j])
                        d_message_ds[index1[mask_Y], j] = self.lambda_func("Y", qX[j] - delta_message_ds[index1[mask_Y], j], qY[j], qZ[j] - delta_message_ds[index1[mask_Y], j])
                        d_message_ds[index1[mask_Z], j] = self.lambda_func("Z", qX[j] - delta_message_ds[index1[mask_Z], j], qY[j] - delta_message_ds[index1[mask_Z], j], qZ[j])
                    else:
                        d_message_ds[index1, j] = qS[j - self.n] - delta_message_ds[index1, j]
                    

                    # hard decision 计算每一个比特上的出错概率
                    if j < self.n:
                        list = [qX[j], qY[j], qZ[j]]
                        min_value = min(list)
                        indx = list.index(min_value)

                        if qX[j] > 0 and qY[j] > 0 and qZ[j] > 0:
                            Error[j] = 0
                            Error[j + self.n] = 0
                        elif indx == 0:    # 发生X错误
                            Error[j] = 1
                            Error[j + self.n] = 0
                        elif indx == 1:    # 发生Y错误
                            Error[j] = 1
                            Error[j + self.n] = 1
                        elif indx == 2:    # 发生Z错误
                            Error[j] = 0
                            Error[j + self.n] = 1
                    else:
                        if qS[j - self.n] <= 0:
                            Error[j + self.n] = 1

                    # if last_Error[j] == Error[j] & last_Error[j + self.n] == Error[j + self.n]:
                    #     reliability[j] += 1
                    # else:
                    #     reliability[j] == 1
                    
                vertical_time += time.time() - start_time

                # last_Error = np.copy(Error)

            elif schedule == "layer":
                # 变量节点层级调度
                for j in range(self.n + self.s):
                    if j < self.n:
                        index1 = np.where(self.H[:, j] != 0)[0]
                    else:
                        index1 = np.where(self.Hs[:, j - self.n] != 0)[0]

                    # 水平更新
                    start_time = time.time()
                    for m in index1:
                        index_m = np.where(self.H[m, :] != 0)[0]
                        index_m = np.concatenate((index_m, np.where(self.Hs[m, :] != 0)[0] + self.n))
                        index_m = index_m[index_m != j]

                        tanh_values = np.tanh(d_message_ds[m, index_m] / 2) + 1e-10
                        product_term = np.clip(np.prod(tanh_values), -1+1e-10, 1-1e-10)
                        
                        delta_value = 2 * np.arctanh(product_term)
                        delta_message_ds[m, j] = (-1)**syndrome[m] * delta_value
                    
                    horizontal_time += time.time() - start_time

                    start_time = time.time()
                    
                    if j < self.n:
                        qX[j] = qX_0[j]
                        qY[j] = qY_0[j]
                        qZ[j] = qZ_0[j]
                    else:
                        qS[j - self.n] = qS_0[j - self.n]

                    if init == "Momentum":
                        # 更新动量项
                        if j < self.n:
                            gt[j, 0] = beta * gt[j, 0] + (1-beta) * (last_qX[j] - qX[j])
                            gt[j, 1] = beta * gt[j, 1] + (1-beta) * (last_qY[j] - qY[j])
                            gt[j, 2] = beta * gt[j, 2] + (1-beta) * (last_qZ[j] - qZ[j])

                            qX[j] = last_qX[j] - alpha * gt[j, 0]
                            qY[j] = last_qY[j] - alpha * gt[j, 1]
                            qZ[j] = last_qZ[j] - alpha * gt[j, 2]
                        else:
                            gt_S[j - self.n] = beta * gt_S[j - self.n] + (1-beta) * (last_qS[j - self.n] - qS[j - self.n])
                            qS[j - self.n] = last_qS[j - self.n] - alpha * gt_S[j - self.n]

                        last_qX, last_qY, last_qZ, last_qS = qX.copy(), qY.copy(), qZ.copy(), qS.copy()
                    
                    elif init != "None":
                        raise ValueError(f"Invalid init method: {init}. Expected 'Momentum'.")

                    # 软判决向量
                    if j < self.n:
                        qX_summands = ((self.Hz[index1, j] == 1)) * delta_message_ds[index1, j]
                        qY_summands = ((self.Hx[index1, j] != self.Hz[index1, j])) * delta_message_ds[index1, j]
                        qZ_summands = ((self.Hx[index1, j] == 1)) * delta_message_ds[index1, j]
                        qX[j] += np.sum(qX_summands)
                        qY[j] += np.sum(qY_summands)
                        qZ[j] += np.sum(qZ_summands)
                    else:
                        qS_summands = ((self.Hs[index1, j - self.n] == 1)) * delta_message_ds[index1, j]
                        qS[j - self.n] += np.sum(qS_summands)

                    if method == "Momentum":
                        # 更新动量项
                        if j < self.n:
                            gt_q[j, 0] = beta * gt_q[j, 0] + (1-beta) * (last_qX_q[j] - qX[j])
                            gt_q[j, 1] = beta * gt_q[j, 1] + (1-beta) * (last_qY_q[j] - qY[j])
                            gt_q[j, 2] = beta * gt_q[j, 2] + (1-beta) * (last_qZ_q[j] - qZ[j])

                            qX[j] = last_qX_q[j] - alpha * gt_q[j, 0]
                            qY[j] = last_qY_q[j] - alpha * gt_q[j, 1]
                            qZ[j] = last_qZ_q[j] - alpha * gt_q[j, 2]
                        else:
                            gt_S_q[j - self.n] = beta * gt_S_q[j - self.n] + (1-beta) * (last_qS_q[j - self.n] - qS[j - self.n])
                            qS[j - self.n] = last_qS_q[j - self.n] - alpha * gt_S_q[j - self.n]

                        last_qX_q, last_qY_q, last_qZ_q, last_qS_q = qX.copy(), qY.copy(), qZ.copy(), qS.copy()

                    elif method == "Ada":
                        # AdaGrad
                        if j < self.n:
                            gt_q[j, 0] = last_qX_q[j] - qX[j]
                            gt_q[j, 1] = last_qY_q[j] - qY[j]
                            gt_q[j, 2] = last_qZ_q[j] - qZ[j]

                            Vt[j] += np.square(gt_q[j])

                            if i != 0:
                                qX[j] = last_qX_q[j] - (alpha * gt_q[j, 0]) / (np.sqrt(Vt[j, 0]) + 1e-10)
                                qY[j] = last_qY_q[j] - (alpha * gt_q[j, 1]) / (np.sqrt(Vt[j, 1]) + 1e-10)
                                qZ[j] = last_qZ_q[j] - (alpha * gt_q[j, 2]) / (np.sqrt(Vt[j, 2]) + 1e-10)
                        
                        else:
                            gt_S_q[j - self.n] = last_qS_q[j - self.n] - qS[j - self.n]
                            Vt_S_q[j - self.n] += np.square(gt_S_q[j - self.n])

                            if i != 0:
                                qS[j - self.n] = last_qS_q[j - self.n] - (alpha * gt_S_q[j - self.n]) / (np.sqrt(Vt_S_q[j - self.n]) + 1e-10)

                        last_qX_q, last_qY_q, last_qZ_q, last_qS_q = qX.copy(), qY.copy(), qZ.copy(), qS.copy()
                    
                    elif method == "MBP":
                        if j < self.n:
                            qX[j] = qX[j] - np.sum(qX_summands) + np.sum(qX_summands) / alpha
                            qY[j] = qY[j] - np.sum(qY_summands) + np.sum(qY_summands) / alpha
                            qZ[j] = qZ[j] - np.sum(qZ_summands) + np.sum(qZ_summands) / alpha
                        else:
                            qS[j - self.n] = qS[j - self.n] - np.sum(qS_summands) + np.sum(qS_summands) / alpha

                    elif method != "None":
                        raise ValueError(f"Invalid method: {method}. Expected 'Momentum' or 'Ada'.")

                    # 垂直更新
                    if j < self.n:
                        for m in index1:
                            if (self.Hx[m, j] == 1) & (self.Hz[m, j] == 0):
                                d_message_ds[m, j] = self.lambda_func("X", qX[j], qY[j] - delta_message_ds[m, j], qZ[j] - delta_message_ds[m, j])
                            elif (self.Hx[m, j] == 1) & (self.Hz[m, j] == 1):
                                d_message_ds[m, j] = self.lambda_func("Y", qX[j] - delta_message_ds[m, j], qY[j], qZ[j] - delta_message_ds[m, j])
                            elif (self.Hx[m, j] == 0) & (self.Hz[m, j] == 1):
                                d_message_ds[m, j] = self.lambda_func("Z", qX[j] - delta_message_ds[m, j], qY[j] - delta_message_ds[m, j], qZ[j])
                    else:
                        d_message_ds[index1, j] = qS[j - self.n] - delta_message_ds[index1, j]

                    vertical_time += time.time() - start_time

                start_time = time.time()
                # hard decision 计算每一个比特上的出错概率
                for j in range(self.n):
                    list = [qX[j], qY[j], qZ[j]]
                    min_value = min(list)
                    indx = list.index(min_value)

                    if qX[j] > 0 and qY[j] > 0 and qZ[j] > 0:
                        Error[j] = 0
                        Error[j + self.n] = 0
                    elif indx == 0:    # 发生X错误
                        Error[j] = 1
                        Error[j + self.n] = 0
                    elif indx == 1:    # 发生Y错误
                        Error[j] = 1
                        Error[j + self.n] = 1
                    elif indx == 2:    # 发生Z错误
                        Error[j] = 0
                        Error[j + self.n] = 1

                for j in range(self.n, self.s + self.n):
                    if qS[j - self.n] <= 0:
                        Error[j + self.n] = 1

                    # if last_Error[j] == Error[j] and last_Error[j + self.n] == Error[j + self.n]:
                    #     reliability[j] += 1
                    # else:
                    #     oscillation[j] += 1
                    #     reliability[j] == 1
    
                # last_Error = np.copy(Error)
                vertical_time += time.time() - start_time
            
            else:
                raise ValueError(f"Invalid schedule: {schedule}. Expected 'flooding' or 'layer'.")
            
            # llr_history[i] = {"qX": qX.copy(), "qY": qY.copy(), "qZ": qZ.copy()}

            if ((self.Hz @ Error[0:self.n].T + self.Hx @ Error[self.n:2*self.n].T + self.Hs @ Error[2*self.n:2*self.n + self.s]) % 2 == syndrome).all():
                self.flag = True
                return Error, self.flag, i+1
            
        if OSD == "binary":
            px = 1 / (np.exp(qX) + 1)            
            py = 1 / (np.exp(qY) + 1)            
            pz = 1 / (np.exp(qZ) + 1)
            ps = 1 / (np.exp(qS) + 1)

            # binary_X = np.log((1 - px - py) / (px + py))
            # binary_Z = np.log((1 - pz - py) / (pz + py))
            binary_X = px + py
            binary_Z = pz + py

            probability = np.hstack((-binary_X, -binary_Z, -ps))

            Error = binary_osd_pheno(syndrome, self.binary_H, probability, self.k, self.s)
        
        elif OSD != "None":
            raise ValueError(f"Invalid OSD: {OSD}. Expected 'binary'.")

        # print(f"OTS_times: {OTS_times}") if OTS_times > 0 else None

        return Error, False, self.max_iter


    def plot_energy(self):
        import matplotlib.pyplot as plt  # 导入绘图库

        plt.figure(figsize=(8, 6))  # 设置图像大小
        plt.plot(self.energy_history, marker='o')  # 绘制能量值，带有标记
        plt.title('Energy per Iteration')  # 设置标题
        plt.xlabel('Iteration')  # 设置X轴标签
        plt.ylabel('Energy')  # 设置Y轴标签
        plt.grid(True)  # 显示网格
        plt.show()  # 显示图像
    

# Non-CSS OSD-0
def osd0_post_processing(syndrome, decoupled_H, reliability, probability, k):
    n = decoupled_H.shape[1] // 3
    GF = galois.GF(2)
    rank = n - k
    # print(rank)
    sorted_col = np.argsort(probability)
    index = [sorted_col[0]]
    # print(f"index: {index}")

    for i in range(1, 3*n):
        temp = index
        temp = np.append(temp, sorted_col[i])
        Hjprime = GF(decoupled_H[:, temp])
        Hj = GF(decoupled_H[:, index])
        rank1 = np.linalg.matrix_rank(Hjprime)
        rank2 = np.linalg.matrix_rank(Hj)

        if rank2 == rank:
            break
        if rank1 > rank2:
            index = temp

    decoupled_error = np.zeros(3*n).astype(np.uint8)
    syndromes = GF(syndrome)
    Hj = GF(decoupled_H[:, index])
    X = np.linalg.solve(Hj, syndromes)
    decoupled_error[index] = X

    Error = np.zeros(2*n).astype(np.uint8)
    
    for i in range(n):
        if decoupled_error[i] == 1:  # X error
            Error[i] += 1
        if decoupled_error[i + n] == 1:  # Y error
            Error[i] += 1
            Error[i + n] += 1
        if decoupled_error[i + 2*n] == 1:  # Z error
            Error[i + n] += 1

    return Error

def binary_osd(syndrome, binary_H, reliability, probability, k):
    n = binary_H.shape[1] // 2
    GF = galois.GF(2)
    rank = n - k

    while rank < binary_H.shape[0]:
        if binary_H.shape[0] % 2 == 0:
            binary_H = np.delete(binary_H, 0, axis=0)  # 删除最上面一行
            syndrome = np.delete(syndrome, 0, axis=0)
        else:
            binary_H = np.delete(binary_H, -1, axis=0)  # 删除最下面一行
            syndrome = np.delete(syndrome, -1, axis=0)

    sorted_col = np.argsort(probability)
    index = [sorted_col[0]]
    # print(f"index: {index}")

    for i in range(1, 2*n):
        temp = index
        temp = np.append(temp, sorted_col[i])
        Hjprime = GF(binary_H[:, temp])
        Hj = GF(binary_H[:, index])
        rank1 = np.linalg.matrix_rank(Hjprime)
        rank2 = np.linalg.matrix_rank(Hj)

        if rank2 == rank:
            break
        if rank1 > rank2:
            index = temp

    decoupled_error = np.zeros(2*n).astype(np.uint8)
    syndromes = GF(syndrome)
    Hj = GF(binary_H[:, index])
    X = np.linalg.solve(Hj, syndromes)
    decoupled_error[index] = X

    Error = np.zeros(2*n).astype(np.uint8)
    
    for i in range(n):
        if decoupled_error[i] == 1:  # X error
            Error[i] += 1
        if decoupled_error[i + n] == 1:  # Z error
            Error[i + n] += 1

    return Error


def binary_osd_pheno(syndrome, binary_H, probability, k, s):
    total_col = binary_H.shape[1]
    n = (binary_H.shape[1] - s) // 2
    rank = n - k

    while rank < binary_H.shape[0]:
        if binary_H.shape[0] % 2 == 0:
            binary_H = np.delete(binary_H, 0, axis=0)  # 删除最上面一行
            syndrome = np.delete(syndrome, 0, axis=0)
        else:
            binary_H = np.delete(binary_H, -1, axis=0)  # 删除最下面一行
            syndrome = np.delete(syndrome, -1, axis=0)
            
    GF = galois.GF(2)

    sorted_col = np.argsort(probability)
    index = [sorted_col[0]]
    # print(f"index: {index}")

    for i in range(1, total_col):
        temp = index.copy()
        temp.append(sorted_col[i])
        Hjprime = GF(binary_H[:, temp])
        Hj = GF(binary_H[:, index])
        rank1 = np.linalg.matrix_rank(Hjprime)
        rank2 = np.linalg.matrix_rank(Hj)

        if rank2 == rank:
            break
        if rank1 > rank2:
            index = temp

    decoupled_error = np.zeros(total_col).astype(np.uint8)
    syndromes = GF(syndrome)
    Hj = GF(binary_H[:, index])
    X = np.linalg.solve(Hj, syndromes)
    decoupled_error[index] = X

    Error = np.zeros(total_col).astype(np.uint8)
    
    for i in range(n):
        if decoupled_error[i] == 1:  # X error
            Error[i] += 1
        if decoupled_error[i + n] == 1:  # Z error
            Error[i + n] += 1

    for i in range(2*n, total_col):
        if decoupled_error[i] == 1:  # 测量错误
            Error[i] += 1

    return Error


def b_r_osd(syndrome, decoupled_H, reliability, probability, k):
    n = decoupled_H.shape[1] // 2
    GF = galois.GF(2)
    rank = n - k

    sorted_bit_indices = np.argsort(reliability)
    sorted_col = np.zeros(2*n, dtype=int)
    
    # 对每个比特，根据其错误概率对其两种错误进行排序
    for i, bit_index in enumerate(sorted_bit_indices):
        error_indices = np.array([bit_index, bit_index + n])
        error_probs = probability[error_indices]
        sorted_error_indices = error_indices[np.argsort(error_probs)]

        # 将排序后的错误索引填入最终的排序列表
        sorted_col[2 * i: 2 * i + 2] = sorted_error_indices

    index = [sorted_col[0]]

    for i in range(1, 2*n):
        temp = index
        temp = np.append(temp, sorted_col[i])
        Hjprime = GF(decoupled_H[:, temp])
        Hj = GF(decoupled_H[:, index])
        rank1 = np.linalg.matrix_rank(Hjprime)
        rank2 = np.linalg.matrix_rank(Hj)

        if rank2 == rank:
            break
        if rank1 > rank2:
            index = temp

    decoupled_error = np.zeros(2*n).astype(np.uint8)
    syndromes = GF(syndrome)
    Hj = GF(decoupled_H[:, index])
    X = np.linalg.solve(Hj, syndromes)
    decoupled_error[index] = X

    Error = np.zeros(2*n).astype(np.uint8)
    
    for i in range(n):
        if decoupled_error[i] == 1:  # X error
            Error[i] += 1
        if decoupled_error[i + n] == 1:  # Z error
            Error[i + n] += 1

    return Error


def osd1_post_processing(syndrome, parity_check_matrix, error, probability_distribution):
    X, index, index_c = osd0_post_processing(syndrome, parity_check_matrix, error, probability_distribution)
    n = parity_check_matrix.shape[1]
    GF = galois.GF(2)
    Error = np.zeros(n).astype(np.uint8)
    weight_X = 0
    for i in range(np.size(X)):
        if X[i] == 1:
            weight_X = weight_X + 1

    for i in range(1, 2):
        Error[index_c] = 1
        syndromes = (syndrome + parity_check_matrix @ Error.T) % 2
        Hj = GF(parity_check_matrix[:, index])
        syndromes = GF(syndromes)
        XX = np.linalg.solve(Hj, syndromes)
    weight_XX = 0
    for i in range(np.size(XX)):
        if XX[i] == 1:
            weight_XX = weight_XX + 1
    if ((weight_XX+1) < weight_X):
        Error[index] = XX
    else:
        Error[index] = X
        Error[index_c] = 0
    return Error



if __name__ == '__main__':

    Hz = np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    Hx = np.zeros_like(Hz)

    px = 0.01
    pz = 0.01
    py = 0.01

    max_iter = 50
    decoder = LLRBp4Decoder(Hx, Hz, px, py, pz, max_iter, dimension=1)
    error = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    syndrome = (Hz @ error[0:4].T + Hx @ error[4:8].T) % 2
    print(syndrome)
    correction, flag, iter, llr = decoder.standard_decoder(syndrome, schedule="layer", method="Ada", alpha=5)
    # print(correction)
    print(flag, iter)
    # print(reliability)
    # print((Hz @ correction[0:4].T + Hx @ correction[4:8].T) % 2)

    print(llr)

    bit_llr = {i: {"qX": [], "qY": [], "qZ": []} for i in range(Hz.shape[1])}

    actual_iters = len(llr)

    for i in range(actual_iters):
        if i in llr:
            for bit in range(Hz.shape[1]):
                bit_llr[bit]["qX"].append(llr[i]["qX"][bit])
                bit_llr[bit]["qY"].append(llr[i]["qY"][bit])
                bit_llr[bit]["qZ"].append(llr[i]["qZ"][bit])

    # print(bit_llr)

    all_llr_values = []
    for bit in range(Hz.shape[1]):
        all_llr_values.extend(bit_llr[bit]["qX"])
        all_llr_values.extend(bit_llr[bit]["qY"])
        all_llr_values.extend(bit_llr[bit]["qZ"])

    min_llr = min(all_llr_values)
    max_llr = max(all_llr_values)

    # 绘制每个比特的对数似然比变化
    fig, axes = plt.subplots(1, Hz.shape[1], figsize=(4 * Hz.shape[1], 3))

    for bit in range(Hz.shape[1]):
        axes[bit].plot(range(1, actual_iters + 1), bit_llr[bit]["qX"], label='qX', marker='o', linewidth=4)
        axes[bit].plot(range(1, actual_iters + 1), bit_llr[bit]["qY"], label='qY', marker='x')
        axes[bit].plot(range(1, actual_iters + 1), bit_llr[bit]["qZ"], label='qZ', marker='s')
        axes[bit].set_title(f'Qubit {bit+1}')
        axes[bit].set_xlabel('Iteration')
        axes[bit].set_ylabel('Log Likelihood Ratio')
        axes[bit].legend()
        axes[bit].grid(True)
        axes[bit].set_xlim(1, actual_iters + 1)
        axes[bit].set_ylim(min_llr - 2, max_llr + 2)

    # plt.autoscale()
    plt.tight_layout()
    plt.show()


    # Hx = np.array([[1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [0, 1, 1, 1, 1]])
    # Hz = np.array([[0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]])
    # px = 0.001
    # pz = 0.001
    # py = 0.001
    # max_iter = Hx.shape[1]
    # decoder = LLRBp4Decoder(Hx, Hz, px, py, pz, max_iter, 1)
    # error = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    # syndrome = (Hz @ error[0:5].T + Hx @ error[5:10].T) % 2
    # print(syndrome)
    # correction, flag, iter, _, _ = decoder.standard_decoder(syndrome, method="AdaGrad", alpha=5)
    # print(correction)
    # print(flag, iter)
    # print((Hz @ correction[0:5].T + Hx @ correction[5:10].T) % 2)
