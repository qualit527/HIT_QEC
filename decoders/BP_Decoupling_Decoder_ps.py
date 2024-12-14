import numpy as np
from numpy import log, tanh, mod
import galois
import warnings
warnings.filterwarnings("ignore")
# 解耦译码的时候所适用的校验矩阵化为H=[Hx|Hz|Hx+Hz]的形式
# 将 X Y Z 三种错误解耦
# 目前所实现的消息更新的方法属于ps 积和算法


class BpDecoder:

    def __init__(self, parity_check_matrix, error_rate, max_iter, logical_qubits_num):
        self.parity_check_matrix = parity_check_matrix
        self.m = parity_check_matrix.shape[0]
        self.n = parity_check_matrix.shape[1]
        self.error_rate = error_rate
        self.max_iter = max_iter
        self.logical_qubits_num = logical_qubits_num
        self.flag = True

        # 这一段是论文《Decoding Across the Quantum LDPC Code Landscape》
        # 中的算法需要初始化的部分
        self.log_prob_ratio_initial = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            self.log_prob_ratio_initial[i] = log((1 - self.error_rate[i]) / self.error_rate[i])
            if np.isnan(self.log_prob_ratio_initial[i]):
                self.log_prob_ratio_initial[i] = np.inf
                pass

        self.message_Parity2Data = np.zeros(self.parity_check_matrix.shape, dtype=np.float64)
        self.message_Data2Parity = np.zeros(self.parity_check_matrix.shape, dtype=np.float64)
        self.message_Parity2Data[self.parity_check_matrix == 1] = 1.0

        # 每一个变量节点的对数似然比初始化
        # 记录每一列非零元所在的位置
        # 记录每一行非零元所在的位置
        self.nonzero_index_of_each_row = []
        self.nonzero_num_of_each_row = []
        self.same_position_of_each_row = []
        self.nonzero_index_of_each_column = []
        self.nonzero_num_of_each_column = []
        for i in range(self.n):
            self.message_Data2Parity[self.parity_check_matrix[:, i] == 1, i] = self.log_prob_ratio_initial[i]
            index = np.where(self.parity_check_matrix[:, i] == 1)[0]
            self.nonzero_index_of_each_column.append(index)
            self.nonzero_num_of_each_column.append(np.size(index))
        for i in range(self.m):
            index = np.where(self.parity_check_matrix[i, :] == 1)[0]
            self.nonzero_index_of_each_row.append(index)
            self.nonzero_num_of_each_row.append(np.size(index))
            same_position = np.zeros(np.size(index), dtype=np.uint8)
            for j in range(np.size(index)):
                sp1 = mod(index[j] + self.n // 3, self.n)
                sp2 = mod(index[j] + 2 * self.n // 3, self.n)
                if self.parity_check_matrix[i, sp1] == 1:
                    same_position[j] = sp1
                elif self.parity_check_matrix[i, sp2] == 1:
                    same_position[j] = sp2
            self.same_position_of_each_row.append(same_position)
        self.log_prob_ratios = np.zeros(self.n, dtype=np.float64)
        self.last_round_log_prob_ratios = 0

    def standard_decoder(self, syndrome):
        log_prob_ratio_initial = np.copy(self.log_prob_ratio_initial)
        message_Parity2Data = np.copy(self.message_Parity2Data)
        message_Data2Parity = np.copy(self.message_Data2Parity)
        log_prob_ratios = np.copy(self.log_prob_ratios)

        for i in range(self.max_iter):
            alpha = 1
            Error = np.zeros(self.n).astype(np.uint8)
            # 对每一行进行更新
            # 更新校验节点给数据节点的消息
            # 所以需要操作的是原来 数据节点->校验节点 的信息，即message_Data2Parity
            for j in range(self.m):
                index = self.nonzero_index_of_each_row[j]
                num = self.nonzero_num_of_each_row[j]

                for k in range(num):
                    tmp1 = 1.0
                    tmp2 = 1.0
                    for kk in range(num):
                        if kk != k:
                            tmp1 = tmp1*tanh(message_Data2Parity[j, index[kk]]/2)
                        if kk != k and index[kk] != self.same_position_of_each_row[j][k]:
                            tmp2 = tmp2*tanh(message_Data2Parity[j, index[kk]]/2)
                    if syndrome[j] == 0:
                        message_Parity2Data[j, index[k]] = alpha*log((1+tmp1)/(1-tmp2))
                    else:
                        message_Parity2Data[j, index[k]] = alpha*log((1-tmp1)/(1+tmp2))
            # 对每一列进行更新
            # 更新变量节点给校验节点的消息
            # 所以需要操作的是原来 校验节点->变量节点 的信息，即message_Parity2Data
            for j in range(self.n):
                index = self.nonzero_index_of_each_column[j]
                num = self.nonzero_num_of_each_column[j]

                for k in range(num):
                    pr = log_prob_ratio_initial[j]
                    # pr = message_Data2Parity[index[k], j]
                    tmp3 = pr
                    for k1 in range(num):
                        kk = self.same_position_of_each_row[index[k1]]
                        kkk = self.nonzero_index_of_each_row[index[k1]]
                        position = np.where(kkk == j)[0]
                        pr = pr + message_Parity2Data[index[k1], j] - log(1 - self.error_rate[kk[position]])
                        if k1 != k:
                            tmp3 = tmp3 + message_Parity2Data[index[k1], j] - log(1 - self.error_rate[kk[position]])
                    message_Data2Parity[index[k], j] = tmp3
                log_prob_ratios[j] = pr

                # pr = log_prob_ratio_initial[j]
                # for k in range(num):
                #     kk = self.same_position_of_each_row[index[k]]
                #     kkk = self.nonzero_index_of_each_row[index[k]]
                #     position = np.where(kkk == j)[0]
                #     pr = pr + message_Parity2Data[index[k], j] - log(1-self.error_rate[kk[position]])
                # for k in range(num):
                #     kk = self.same_position_of_each_row[index[k]]
                #     kkk = self.nonzero_index_of_each_row[index[k]]
                #     position = np.where(kkk == j)[0]
                #     message_Data2Parity[index[k], j] = pr - message_Parity2Data[index[k], j]
                # log_prob_ratios[j] = pr

            # hard decision
            for j in range(self.n//3):
                list_value = [log_prob_ratios[j], log_prob_ratios[j+self.n//3], log_prob_ratios[j+2*self.n//3]]
                min_num = min(list_value)
                if min_num < 0:
                    indx = list_value.index(min_num)
                    Error[j + indx*self.n//3] = 1
            print(Error)
            print(log_prob_ratios)

            if (self.parity_check_matrix @ Error.T % 2 == syndrome).all():
                self.flag = True
                self.last_round_log_prob_ratios = log_prob_ratios
                return Error, self.flag, log_prob_ratios
        self.flag = False
        self.last_round_log_prob_ratios = log_prob_ratios
        return Error, self.flag, log_prob_ratios

    def bp_osd_decoder(self, syndrome):
        error, flag, reliability = self.standard_decoder(syndrome)
        self.last_round_log_prob_ratios = reliability
        if flag:
            return error
        else:
            Error = osd0_post_processing(syndrome, self.parity_check_matrix, error, reliability, self.logical_qubits_num)
            return Error


def osd0_post_processing(syndrome, parity_check_matrix, error, probability_distribution, logicol_qubits_num):
    n = parity_check_matrix.shape[1]
    GF = galois.GF(2)
    col = []
    sorted_col = np.argsort(probability_distribution)
    col = np.append(col, sorted_col[0] % (n//3)).astype(np.uint8)
    index = [sorted_col[0]]

    for i in range(1, n):
        temp = index
        temp = np.append(temp, sorted_col[i])
        Hjprime = GF(parity_check_matrix[:, temp])
        Hj = GF(parity_check_matrix[:, index])
        rank1 = np.linalg.matrix_rank(Hjprime)
        rank2 = np.linalg.matrix_rank(Hj)
        # Hjs = GF(np.c_[Hj, syndrome])
        # rank3 = np.linalg.matrix_rank(Hjs)
        if rank2 == n//3 - logicol_qubits_num:
            break
        if rank1 > rank2:
            index = temp
    Error = np.zeros(n).astype(np.uint8)
    syndromes = GF(syndrome)
    Hj = GF(parity_check_matrix[:, index])
    X = np.linalg.solve(Hj, syndromes)
    Error[index] = X
    index_c = list(set(temp)-set(index))
    syndromess = syndrome
    weight = 0
    for i in range(n//3):
        if Error[i] == 1 and Error[i + n//3] == 1 and Error[i + 2*n//3] == 0:
            Error[i] = 0
            Error[i + n//3] = 0
            Error[i + 2 * n//3] = 1
            syndromess = (syndromess + parity_check_matrix[:, i + 2 * n//3]) % 2
            weight = weight + 1
        elif Error[i] == 1 and Error[i + n//3] == 0 and Error[i + 2*n//3] == 1:
            Error[i] = 0
            Error[i + n//3] = 1
            Error[i + 2 * n//3] = 0
            syndromess = (syndromess + parity_check_matrix[:, i + n//3]) % 2
            weight = weight + 1
        elif Error[i] == 0 and Error[i + n//3] == 1 and Error[i + 2 * n//3] == 1:
            Error[i] = 1
            Error[i + n//3] = 0
            Error[i + 2 * n//3] = 0
            syndromess = (syndromess + parity_check_matrix[:, i]) % 2
            weight = weight + 1
    # X = Error[index]
    # return X, index, index_c, syndromess, weight
    return Error


def osd1_post_processing(syndrome, parity_check_matrix, error, probability_distribution, logicol_qubits_num):
    X, index, index_c, syndromess, weight = osd0_post_processing(syndrome, parity_check_matrix, error,
                                                                 probability_distribution, logicol_qubits_num)
    n = parity_check_matrix.shape[1]
    GF = galois.GF(2)
    Error = np.zeros(n).astype(np.uint8)
    weight_X = 0
    for i in range(np.size(X)):
        if X[i] == 1:
            weight_X = weight_X + 1

    for i in range(1, len(index_c)+1):
        Error[index_c] = 1
        syndromes = (syndromess + parity_check_matrix @ Error.T) % 2
        Hj = GF(parity_check_matrix[:, index])
        syndromes = GF(syndromes)
        XX = np.linalg.solve(Hj, syndromes)
    weight_XX = 0
    for i in range(np.size(XX)):
        if XX[i] == 1:
            weight_XX = weight_XX + 1
    if (weight_XX+1) < weight_X:
        Error[index] = XX
    else:
        Error[index] = X
        Error[index_c] = 0

    for i in range(n//3):
        if Error[i] == 1 and Error[i + n//3] == 1 and Error[i + 2*n//3] == 0:
            Error[i] = 0
            Error[i + n//3] = 0
            Error[i + 2 * n//3] = 1
        elif Error[i] == 1 and Error[i + n//3] == 0 and Error[i + 2*n//3] == 1:
            Error[i] = 0
            Error[i + n//3] = 1
            Error[i + 2 * n//3] = 0
        elif Error[i] == 0 and Error[i + n//3] == 1 and Error[i + 2 * n//3] == 1:
            Error[i] = 1
            Error[i + n//3] = 0
            Error[i + 2 * n//3] = 0
    return Error


if __name__ == '__main__':

    Hx = np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]], dtype=np.uint8)
    Hz = np.array([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1], [1, 1, 0, 0, 0]], dtype=np.uint8)
    Hy = (Hx + Hz) % 2
    px = 0.02
    pz = 0.02
    py = 0.02
    H_bar = np.hstack([Hx, Hz, Hy])
    channel_error_rate1 = [pz for i in range(5)]
    channel_error_rate2 = [px for i in range(5)]
    channel_error_rate3 = [py for i in range(5)]
    channel_error_rate = channel_error_rate1 + channel_error_rate2 + channel_error_rate3
    channel_error_rate = np.array(channel_error_rate)
    max_iter = H_bar.shape[1]
    # col = [0, 1, 5, 6, 10, 11]
    # H_bar1 = H_bar[:, col]
    # channel_error_rate = channel_error_rate[col]
    # print(H_bar)
    # print(H_bar1)
    # print(channel_error_rate)
    decoder = BpDecoder(H_bar, channel_error_rate, 5, 1)
    error = np.array([0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0])
    syndrome = H_bar @ error.T % 2
    print(syndrome)
    correction, flag, reliability = decoder.standard_decoder(syndrome)
    # correction1 = np.zeros(H_bar.shape[1]).astype(np.uint8)
    # correction1[col] = correction
    # correction = decoder.bp_osd_decoder(syndrome)
    # print(flag)
    print(reliability)
    print(correction)
    # print(correction1)

    # bpd = ldpc.bp_decoder(H_bar, channel_probs=channel_error_rate, max_iter=max_iter, bp_method='ps',
    #                          ms_scaling_factor=0)
    # correction = bpd.decode(syndrome)
    # reliability = bpd.log_prob_ratios
    # print(reliability)
    # print(correction)
