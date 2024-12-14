import numpy as np
from tqdm import tqdm
from scipy.sparse import hstack, vstack, csr_matrix, block_diag
from collections import defaultdict
from .code_builder import build_code
from .decoder_builder import build_decoder, run_decoder
import sys
np.set_printoptions(threshold=sys.maxsize)

class QecExpRunner:
    def __init__(self, config):
        self.config = config
        self.code_config = config.get('code')
        self.decoder_config = config.get('decoders')
        self.max_iter = config.get('max_iter', 100)
        self.n_test = config.get('n_test')
        self.rx = config.get('rx')
        self.ry = config.get('ry')
        self.rz = config.get('rz')
        self.rs = config.get('rs')
        self.p_range = config.get('p_range')
        self.noise_model = config.get('noise_model')
        self.readout = config.get('readout')

        self.block_error_rate = defaultdict(lambda: defaultdict(dict))
        self.slq_error_rate = defaultdict(lambda: defaultdict(dict))
        self.results = defaultdict(lambda: defaultdict(dict))


    def _initialize_metrics(self, num_decoders, dim):
        return {
            "iterations": {i: 0 for i in range(num_decoders)},
            "num_errors": {i: 0 for i in range(num_decoders)},
            "num_not_converge": {i: 0 for i in range(num_decoders)},
            "num_converge_but_logical": {i: 0 for i in range(num_decoders)},
            "num_postprocessed_logical": {i: 0 for i in range(num_decoders)},
            "time_cost": {i: 0 for i in range(num_decoders)},
            "slq_error": [np.zeros(dim, dtype=np.uint16) for _ in range(num_decoders)]
        }
    

    def _update_metrics(self, metrics, isCSS, flag, dim, decoder_num, code_length, syndrome_matching, Lx, Lz, errorX, errorZ):
        # 是否形成逻辑算符
        if isCSS:
            logical_X_or_Y_errors = (errorX @ Lz.T % 2) == 1
            logical_Z_or_Y_errors = (errorZ @ Lx.T % 2) == 1

        else:
            logical_X_or_Y_errors = ((errorX @ Lz[:, code_length:2 * code_length].T + errorZ @ Lz[:, 0:code_length].T) % 2) == 1
            logical_Z_or_Y_errors = ((errorX @ Lx[:, code_length:2 * code_length].T + errorZ @ Lx[:, 0:code_length].T) % 2) == 1

        if np.any(syndrome_matching):
            metrics["num_errors"][decoder_num] += 1

        elif np.any(logical_Z_or_Y_errors) or np.any(logical_X_or_Y_errors):
            metrics["num_errors"][decoder_num] += 1
            if flag:
                metrics["num_converge_but_logical"][decoder_num] += 1
            else:
                metrics["num_postprocessed_logical"][decoder_num] += 1

        if not flag:
            metrics["num_not_converge"][decoder_num] += 1

        if dim < 2:
            if logical_X_or_Y_errors or logical_Z_or_Y_errors:
                metrics["slq_error"][decoder_num][0] += 1
        else:
            for k in range(dim):
                if logical_X_or_Y_errors[k] or logical_Z_or_Y_errors[k]:
                    metrics["slq_error"][decoder_num][k] += 1

    
    def _update_results(self, L, decoder, p, metrics, dim):
        self.block_error_rate[L][decoder][p] = metrics["num_errors"][decoder] / self.n_test
        self.slq_error_rate[L][decoder][p] = np.sum(metrics["slq_error"][decoder]) / (self.n_test * dim)

        self.results[L][decoder][p] = {
            "block_error_rate": self.block_error_rate[L][decoder][p],
            "slq_error_rate": self.slq_error_rate[L][decoder][p],
            "not_converge_rate": metrics["num_not_converge"][decoder] / self.n_test,
            "converge_but_logical_rate": metrics["num_converge_but_logical"][decoder] / self.n_test,
            "postprocessed_logical_rate": metrics["num_postprocessed_logical"][decoder] / self.n_test,
            "avg_iter": metrics["iterations"][decoder] / self.n_test,
            "avg_time": metrics["time_cost"][decoder] / self.n_test
        }


    def _print_results(self, L, p, metrics, M, N, dim):
        result_str = f"p={p}, H=[{M}, {N}]"

        for i, decoder_info in enumerate(self.decoder_config):
            print(f"{decoder_info.get('name')} "
                  f"收敛失败次数={metrics['num_not_converge'][i]}, "
                  f"收敛逻辑错误次数={metrics['num_converge_but_logical'][i]}, "
                  f"后处理逻辑错误次数={metrics['num_postprocessed_logical'][i]}, "
                  f"平均迭代次数={metrics['iterations'][i] / self.n_test}, "
                  f"平均译码时间={metrics['time_cost'][i] / self.n_test:.4f}s")

            if dim > 2:
                result_str += (f", {decoder_info.get('name')} "
                            f"block_error_rate={self.block_error_rate[L][i][p]:.3f}, "
                            f"slq_error_rate={self.slq_error_rate[L][i][p]:.3f}")
            else:
                result_str += (f", {decoder_info.get('name')} num_errors={metrics['num_errors'][i]}")

        print(result_str)


    def run_code_capacity(self):
        for L in self.code_config.get("L_range"):
            print(f"\nSimulating L={L}...")
            Hx, Hz, Lx, Lz, dim, isCSS = build_code(self.code_config.get("name"), L)
            if isCSS:
                Hx0 = np.zeros(Hx.shape).astype(np.uint8)
                Hz0 = np.zeros(Hz.shape).astype(np.uint8)
                Hx = np.vstack([Hx, Hx0])
                Hz = np.vstack([Hz0, Hz])

            code_length = Hx.shape[1]
            assert code_length == Hz.shape[1], f"Error: Hx shape {Hx.shape[1]} does not match Hz shape {Hz.shape[1]}."

            max_iter = self.max_iter
            if max_iter == "N":
                max_iter = code_length

            for p in self.p_range:       
                px = p * self.rx
                py = p * self.ry
                pz = p * self.rz

                decoders = build_decoder("capacity", self.decoder_config, Hx, Hz, dim, px, py, pz, p, max_iter=max_iter)
                
                metrics = self._initialize_metrics(len(self.decoder_config), dim)

                for _ in tqdm(range(self.n_test)):
                    # 生成随机错误，计算差错症状
                    rand = np.random.rand(code_length)
                    z_error = rand < pz
                    x_error = (pz <= rand) & (rand < pz + px)
                    y_error = (pz + px <= rand) & (rand < pz + px + py)
                    z_error = (z_error + y_error) % 2
                    x_error = (x_error + y_error) % 2

                    syndrome_z = Hx @ z_error.T % 2
                    syndrome_x = Hz @ x_error.T % 2
                    syndrome = (syndrome_z + syndrome_x) % 2
                    syndrome = syndrome.astype(int)

                    for i, decoder_info in enumerate(self.decoder_config):
                        name = decoder_info.get("name")
                        params = decoder_info.get("params", {})

                        # 调用抽象decoder
                        correction, time_spent, iter, flag = run_decoder(
                            name=name, 
                            decoder=decoders[i], 
                            syndrome=syndrome, 
                            code_length=code_length, 
                            params=params,
                            noise_model="capacity"
                        )
                        metrics["time_cost"][i] += time_spent
                        metrics["iterations"][i] += iter

                        correction_x, correction_z = correction
                        
                        errorZ = (z_error + correction_z) % 2
                        errorX = (x_error + correction_x) % 2

                        syndrome_matching = (Hz @ errorX.T + Hx @ errorZ.T) % 2

                        self._update_metrics(metrics, isCSS, flag, dim, i, code_length, syndrome_matching, Lx, Lz, errorX, errorZ)

                if not isinstance(L, (int, float)):
                    L = str(L)

                for i, _ in enumerate(self.decoder_config):
                    self._update_results(L, i, p, metrics, dim)

                self._print_results(L, p, metrics, Hx.shape[0], code_length, dim)

        return self.results, dim


    def run_phenomenological(self):
        L_range = self.code_config.get("L_range")
        m_range = self.code_config.get("m_range")

        if len(L_range) == 1:
            L_range = [L_range[0]] * len(m_range)
            iter_range = m_range
        else:
            iter_range = L_range
        
        if len(m_range) == 1:
            m_range = [m_range[0]] * len(L_range)

        for idx, iter_var in enumerate(iter_range):
            L = L_range[idx]
            m_times = m_range[idx]
            print(f"\nSimulating L={L}, rep_times={m_times}...")
            m_times += 1 if self.readout else 0
            Hx, Hz, Lx, Lz, dim, isCSS = build_code(self.code_config.get("name"), L)
            if isCSS:
                Hx0 = np.zeros(Hx.shape).astype(np.uint8)
                Hz0 = np.zeros(Hz.shape).astype(np.uint8)
                Hx = np.vstack([Hx, Hx0])
                Hz = np.vstack([Hz0, Hz])

            code_length = Hx.shape[1]
            assert code_length == Hz.shape[1], f"Error: Hx shape {Hx.shape[1]} does not match Hz shape {Hz.shape[1]}."

            # Construct H with m_times H_bar as diagonal elements
            Hx_concat = Hx.copy()
            for i in range(m_times-1):
                Hx_concat = block_diag([Hx_concat, Hx]).toarray()

            Hz_concat = Hz.copy()
            for i in range(m_times-1):
                Hz_concat = block_diag([Hz_concat, Hz]).toarray()

            # Construct Hs
            m = Hx.shape[0]
            Im = csr_matrix(np.eye(m, dtype=np.uint8))
            rows = []

            s_block_times = m_times - 2 if self.readout else m_times - 1

            first_row = [Im] + [csr_matrix((m, m), dtype=np.uint8)] * s_block_times
            rows.append(hstack(first_row))

            for i in range(s_block_times):
                row_blocks = [csr_matrix((m, m), dtype=np.uint8)] * i + [Im, Im] + [csr_matrix((m, m), dtype=np.uint8)] * (s_block_times - i - 1)
                rows.append(hstack(row_blocks))
            
            if self.readout:
                last_row = [csr_matrix((m, m), dtype=np.uint8)] * s_block_times + [Im]
                rows.append(hstack(last_row))
            
            Hs = vstack(rows).toarray()

            assert Hx_concat.shape[0] == Hz_concat.shape[0] == Hs.shape[0], "Hx_concat, Hz_concat, and Hs must have the same number of rows"

            print(f"Hx shape: {Hx_concat.shape}")
            print(f"Hz shape: {Hz_concat.shape}")
            print(f"Hs shape: {Hs.shape}")

            for p in self.p_range:       
                px = p * self.rx
                py = p * self.ry
                pz = p * self.rz
                ps = p * self.rs

                decoders = build_decoder("phenomenological", self.decoder_config, Hx_concat, Hz_concat, dim, px, py, pz, p, Hs, ps, self.max_iter)

                metrics = self._initialize_metrics(len(self.decoder_config), dim)

                for _ in tqdm(range(self.n_test)):
                    x_error_total = np.zeros(code_length, dtype=np.uint8)
                    z_error_total = np.zeros(code_length, dtype=np.uint8)

                    last_syndrome = np.zeros(Hx.shape[0], dtype=np.uint8)

                    x_error_concat = np.zeros(code_length * m_times, dtype=np.uint8)
                    z_error_concat = np.zeros(code_length * m_times, dtype=np.uint8)
                    syndrome_error_concat = np.zeros(Hs.shape[1], dtype=np.uint8)
                    syndrome_concat = np.zeros(Hx.shape[0] * m_times, dtype=np.uint8)

                    # 重复测量，累积错误、拼接差错症状
                    for t in range(m_times):
                        rand = np.random.rand(code_length)
                        z_error = rand < pz
                        x_error = (pz <= rand) & (rand < pz + px)
                        y_error = (pz + px <= rand) & (rand < pz + px + py)
                        z_error = (z_error + y_error) % 2
                        x_error = (x_error + y_error) % 2

                        x_error_total = (x_error + x_error_total) % 2
                        z_error_total = (z_error + z_error_total) % 2

                        syndrome_noiseless = (Hx @ z_error_total.T % 2 + Hz @ x_error_total.T % 2) % 2

                        if t == m_times - 1 and self.readout:
                            syndrome_error = np.zeros(Hx.shape[0], dtype=np.uint8)
                        else:
                            syndrome_error = (np.random.rand(Hx.shape[0]) < ps).astype(np.uint8)
                            syndrome_error_concat[t * Hx.shape[0]: (t + 1) * Hx.shape[0]] = syndrome_error

                        syndrome_noisy = ((syndrome_noiseless + syndrome_error) % 2).astype(np.uint8)

                        syndrome_concat[t * Hx.shape[0]: (t + 1) * Hx.shape[0]] = (syndrome_noisy + last_syndrome) % 2

                        last_syndrome = syndrome_noisy

                        x_error_concat[t * code_length: (t + 1) * code_length] = x_error
                        z_error_concat[t * code_length: (t + 1) * code_length] = z_error

                    for i, decoder_info in enumerate(self.decoder_config):
                        name = decoder_info.get("name")
                        params = decoder_info.get("params", {})

                        # 调用decoder接口
                        correction, time_spent, iter, flag = run_decoder(
                            name=name, 
                            decoder=decoders[i], 
                            syndrome=syndrome_concat, 
                            code_length=code_length * m_times, 
                            params=params,
                            noise_model="phenomenological"
                        )
                        metrics["time_cost"][i] += time_spent
                        metrics["iterations"][i] += iter

                        correction_x, correction_z, correction_syndrome = correction
                        
                        errorZ = (z_error_concat + correction_z) % 2
                        errorX = (x_error_concat + correction_x) % 2
                        errorS = (syndrome_error_concat + correction_syndrome) % 2

                        syndrome_matching = (Hz_concat @ errorX.T + Hx_concat @ errorZ.T + Hs @ errorS.T) % 2

                        errorX_blocks = np.split(errorX, m_times)
                        errorX_sum = np.sum(errorX_blocks, axis=0) % 2
                        errorZ_blocks = np.split(errorZ, m_times)
                        errorZ_sum = np.sum(errorZ_blocks, axis=0) % 2

                        self._update_metrics(metrics, isCSS, flag, dim, i, code_length, syndrome_matching, Lx, Lz, errorX_sum, errorZ_sum)

                for i, _ in enumerate(self.decoder_config):
                    self._update_results(iter_var, i, p, metrics, dim)

                self._print_results(iter_var, p, metrics, Hx_concat.shape[0], Hx_concat.shape[1] + Hs.shape[1], dim)

        return self.results, dim
                    
