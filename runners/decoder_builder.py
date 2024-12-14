import sys
sys.path.append("..")
from decoders import *
import numpy as np
import time
import ldpc

def build_decoder(noise_model, decoder_config, Hx, Hz, dim, px, py, pz, p, Hs=None, ps=0, max_iter=100):
    """
    实例化对应译码器

    Args:
        decoder_config (list): 包含多个译码器的配置信息的列表
        Hx (ndarray): X 稳定子矩阵
        Hz (ndarray): Z 稳定子矩阵
        dim (int): 码的维数
        px (float): X 错误率
        py (float): Y 错误率
        pz (float): Z 错误率
        max_iter (int): 最大迭代次数
    
    Returns:
        list: 包含实例化译码器对象的列表
    """

    decoders = []

    # 遍历 decoder_config 列表，实例化每个译码器
    for decoder_num, decoder_info in enumerate(decoder_config):
        name = decoder_info.get("name")
        params = decoder_info.get("params", {})
        bias = decoder_info.get("bias", {})
        code_length = Hx.shape[1]

        if bias:
            p = bias.get("rp", 1) * p
            if bias.get("p") is not None:
                p = bias.get("p")
            px = p * bias.get("rx")
            py = p * bias.get("ry")
            pz = p * bias.get("rz")
        
        if name == "LLRBP_py":
            if noise_model == "capacity":
                decoders.append(LLRBp4Decoder_py(Hx, Hz, px, py, pz, max_iter, dimension=dim))
            elif noise_model == "phenomenological":
                decoders.append(LLRBp4Decoder_py(Hx, Hz, px, py, pz, max_iter, Hs, ps, dimension=dim))
        
        elif name in ["LLRBP", "EWA-BP", "MBP", "AMBP", "AEWA-BP"]:
            if noise_model == "capacity":
                decoders.append(LLRBp4Decoder(Hx, Hz, px, py, pz, max_iter, dimension=dim))

            elif noise_model == "phenomenological":
                decoders.append(LLRBp4Decoder(Hx, Hz, px, py, pz, max_iter, Hs, ps, dimension=dim))

            else:
                raise ValueError(f"Invalid noise model for {name}: {noise_model}")

        elif name in ["BP2", "BP-OSD", "BPOSD"]:
            isOSD = params.get("OSD", False)
            if name in ["BP-OSD", "BPOSD"]:
                isOSD = True
            
            channel_error_rate4 = [pz + py for i in range(code_length)]
            channel_error_rate5 = [px + py for i in range(code_length)]

            H = np.hstack([Hx, Hz])
            channel_probs = np.array(channel_error_rate4 + channel_error_rate5)

            if isOSD:
                decoders.append(ldpc.bposd_decoder(H, channel_probs=channel_probs, max_iter=max_iter, bp_method='ps', osd_method="osd_cs", osd_order=0))
            else:
                decoders.append(ldpc.bp_decoder(H, channel_probs=channel_probs, max_iter=max_iter, bp_method='ps'))

        elif name in ["PDBP", "PDBP-OSD", "FDBP", "FDBP-OSD"]:
            Hy = (Hx + Hz) % 2

            channel_error_rate1 = [pz for i in range(code_length)]
            channel_error_rate2 = [px for i in range(code_length)]
            channel_error_rate3 = [py for i in range(code_length)]

            if noise_model == "capacity":
                H_bar = np.hstack([Hx, Hz, Hy])
                channel_error_rate = np.array(channel_error_rate1 + channel_error_rate2 + channel_error_rate3)

            elif noise_model == "phenomenological":
                H_bar = np.hstack([Hx, Hz, Hy, Hs])
                channel_error_rate4 = [ps for i in range(code_length)]
                channel_error_rate = np.array(channel_error_rate1 + channel_error_rate2 + channel_error_rate3 + channel_error_rate4)
                raise ValueError(f"FDBP-noisy not implemented yet")
            
            else:
                raise ValueError(f"Invalid noise model for {name}: {noise_model}")
            
            if name == "PDBP":
                decoders.append(ldpc.bp_decoder(H_bar, channel_probs=channel_error_rate, max_iter=max_iter, bp_method='ps'))
            elif params.get("OSD") is not None or name == "PDBP-OSD":
                decoders.append(ldpc.bposd_decoder(H_bar, channel_probs=channel_error_rate, max_iter=max_iter, bp_method='ps', osd_method="osd_cs", osd_order=0))
            elif isinstance(max_iter, list):
                decoders.append(FDBPDecoder(FDBPDecoder.Method.PRODUCT_SUM, max_iter[decoder_num], Mod2SparseMatrix(H_bar), channel_error_rate))
            else:
                decoders.append(FDBPDecoder(FDBPDecoder.Method.PRODUCT_SUM, max_iter, Mod2SparseMatrix(H_bar), channel_error_rate))
                
        elif name in ["Matching", "MWPM"]:
            weights_x = np.full(Hx.shape[1], np.log((1 - (px + py)) / (px + py)))
            weights_z = np.full(Hz.shape[1], np.log((1 - (pz + py)) / (pz + py)))

            if noise_model == "capacity":
                decoders.append(Matching.from_check_matrix(np.hstack([Hx, Hz]), np.hstack([weights_x, weights_z])))

            elif noise_model == "phenomenological":
                weights_s = np.full(Hs.shape[1], np.log((1 - ps) / ps))
                decoders.append(Matching.from_check_matrix(np.hstack([Hx, Hz, Hs]), np.hstack([weights_x, weights_z, weights_s])))

            else:
                raise ValueError(f"Invalid noise model for {name}: {noise_model}")

        else:
            raise ValueError(f"Unknown decoder name: {name}")
        
    return decoders


# 定义字符串到枚举的映射
schedule_map = {
    "flooding": ScheduleType.FLOODING,
    "layer": ScheduleType.LAYER
}

init_map = {
    "Momentum": InitType.MOMENTUM,
    "None": InitType.NONE
}

method_map = {
    "Momentum": MethodType.MOMENTUM,
    "Ada": MethodType.ADA,
    "MBP": MethodType.MBP,
    "None": MethodType.NONE
}

OSD_map = {
    "binary": OSDType.BINARY,
    "None": OSDType.NONE
}

def run_decoder(name, decoder, syndrome, code_length, params, noise_model):
    """
    封装译码器的运行

    Args:
        name (str): 译码器名称，例如 "LLRBP", "EWA-BP", "MBP", "BP2", "FDBP"
        decoder (object): 译码器实例
        syndrome (ndarray): 码的差错症状
        code_length (int): 码的长度
        params (dict): 译码器参数，例如 schedule, OSD 等

    Returns:
        correction_x (ndarray): X 错误估计
        correction_z (ndarray): Z 错误估计
        time_cost (float): 译码运行时间
        iter (int): 译码迭代次数
        flag (bool): 译码是否成功
    """

    start_time = time.time()

    # 创建一个参数的副本以避免修改原始字典
    enum_params = params.copy()

    if name == "LLRBP_py":
        if noise_model == "capacity":
            correction, flag, iter = decoder.standard_decoder(syndrome, **params)
            time_cost = time.time() - start_time
            correction_x = correction[0:code_length]
            correction_z = correction[code_length:2 * code_length]
            return [correction_x, correction_z], time_cost, iter, flag

        elif noise_model == "phenomenological":
            correction, flag, iter = decoder.phenomenological_decoder(syndrome, **params)
            time_cost = time.time() - start_time
            correction_x = correction[0:code_length]
            correction_z = correction[code_length:2 * code_length]
            correction_s = correction[2 * code_length:]
            return [correction_x, correction_z, correction_s], time_cost, iter, flag
        

    # LLRBP类型的译码器
    elif name in ["LLRBP", "EWA-BP", "MBP", "AMBP", "AEWA-BP"]:

        if "schedule" in enum_params:
            enum_value = schedule_map.get(enum_params["schedule"])
            enum_params["schedule"] = enum_value

        if "init" in enum_params:
            enum_value = init_map.get(enum_params["init"])
            enum_params["init"] = enum_value

        if "method" in enum_params:
            enum_value = method_map.get(enum_params["method"])
            enum_params["method"] = enum_value

        if "OSD" in enum_params:
            enum_value = OSD_map.get(enum_params["OSD"])
            enum_params["OSD"] = enum_value
        

        correction = np.zeros(2 * code_length)
        flag = False
        iter = 0

        if name == "MBP":
            correction, flag, iter = decoder.standard_decoder(syndrome, method=MethodType.MBP, **enum_params)

        elif name == "AMBP":
            alphas = params.get("alphas", [1, 0.5, 11])
            alpha_range = np.linspace(alphas[0], alphas[1], alphas[2])
            for alpha in alpha_range:
                enum_params["alpha"] = alpha
                params_copy = enum_params.copy()
                params_copy.pop("alphas", None)
                correction, flag, iter = decoder.standard_decoder(syndrome, method=MethodType.MBP, **params_copy)
                if flag:
                    break

        else:
            if name in ["EWA-BP", "AEWA-BP"] and "init" not in params:
                enum_params["init"] = InitType.MOMENTUM
                
            if name == "AEWA-BP":
                alphas = params.get("alphas", [1, 0, 11])
                alpha_range = np.linspace(alphas[0], alphas[1], alphas[2])
                for alpha in alpha_range:
                    enum_params["alpha"] = alpha
                    params_copy = enum_params.copy()
                    params_copy.pop("alphas", None)
                    correction, flag, iter = decoder.standard_decoder(syndrome, **params_copy)
                    if flag:
                        # if alpha < 0.1:
                            # print(f"params: {params_copy}")
                        break
            
            else:
                correction, flag, iter = decoder.standard_decoder(syndrome, **enum_params)
                
        time_cost = time.time() - start_time

        if noise_model == "capacity":
            correction_x = correction[0:code_length]
            correction_z = correction[code_length:2 * code_length]
            return [correction_x, correction_z], time_cost, iter, flag

        elif noise_model == "phenomenological":
            correction_x = correction[0:code_length]
            correction_z = correction[code_length:2 * code_length]
            correction_s = correction[2 * code_length:]
            return [correction_x, correction_z, correction_s], time_cost, iter, flag
        
        else:
            raise ValueError(f"Invalid noise model for {name}: {noise_model}")
            

    # FDBP类型的译码器
    elif name in ["PDBP", "PDBP-OSD", "FDBP", "FDBP-OSD"]:
        if name in ["PDBP", "PDBP-OSD"]:
            correction = decoder.decode(syndrome)
            iter = 0
            flag = decoder.converge

        elif params.get("OSD") is not None or name == "FDBP-OSD":
            correction, iter, flag = decoder.bpOsdDecode(syndrome)
        else:
            correction, iter, flag = decoder.bpDecode(syndrome)

        correction_y = correction[2 * code_length:3 * code_length]
        correction_z = (correction[0:code_length] + correction_y) % 2
        correction_x = (correction[code_length:2 * code_length] + correction_y) % 2

        time_cost = time.time() - start_time

        if noise_model == "capacity":
            return [correction_x, correction_z], time_cost, iter, flag

        else:
            raise ValueError(f"Invalid noise model for {name}: {noise_model}")
        
    # BP in GF(2)
    elif name in ["BP2", "BP-OSD", "BPOSD"]:
        correction = decoder.decode(syndrome)
        correction_z = correction[0:code_length]
        correction_x = correction[code_length:2 * code_length]

        time_cost = time.time() - start_time

        if noise_model == "capacity":
            return [correction_x, correction_z], time_cost, 0, decoder.converge
        else:
            raise ValueError(f"Invalid noise model for {name}: {noise_model}")
    

    # 最小重完美匹配
    elif name in ["Matching", "MWPM"]:
        correction = decoder.decode(syndrome)
        correction_z = correction[0:code_length]
        correction_x = correction[code_length:2 * code_length]
        time_cost = time.time() - start_time

        if noise_model == "capacity":
            return [correction_x, correction_z], time_cost, 0, True
        elif noise_model == "phenomenological":
            correction_s = correction[2 * code_length:]
            return [correction_x, correction_z, correction_s], time_cost, 0, True

    else:
        raise ValueError(f"Invalid decoder: {name}")

