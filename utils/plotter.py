import pandas as pd
import re
import os
import matplotlib.pyplot as plt

def save_results(config, dim, results, path):
    """
    Args:
        p_range (list): 物理错误率数组
        decoder_config (list): 译码器配置信息，包含每个译码器的名字
        results (dict): 译码器的实验结果，格式为 [L][decoder][p]
        path (str): 输出 Excel 文件路径
    """

    p_range = config.get('p_range')
    decoder_config = config.get('decoders')
    decoder_names = [decoder.get('name') for decoder in decoder_config]
    code_config = config.get('code')
    code_name = code_config.get('name')
    L_range = code_config.get('L_range')
    noise_model = config.get('noise_model')
    n_test = config.get('n_test')

    # 检查 code_name 文件夹是否存在
    path = os.path.join(path, code_name)
    if not os.path.exists(path):
        os.makedirs(path)

    decoder_names_str = "_".join(decoder_names)
    path += f"/{L_range}_{decoder_names_str}_{noise_model}_T={n_test}.xlsx"

    # 检查文件是否存在，若存在则修改文件名
    while os.path.exists(path):
        match = re.search(r"_(\d+)\.xlsx$", path)  # 查找最后的下划线后的内容是否为数字
        if match:
            number = int(match.group(1)) + 1
            path = re.sub(r"_(\d+)\.xlsx$", f"_{number}.xlsx", path)
        else:
            path = path.replace(f"T={n_test}.xlsx", f"T={n_test}_1.xlsx")

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        workbook = writer.book  # 获取 workbook 对象
        center_format = workbook.add_format({'align': 'center'})
                                            
        for L, result_data in results.items():
            # Sheet 名称为 "L=x"
            sheet_name = f"L={L}"
            data = []
            
            # 构建表格的标题行，第一列为 'Decoder'，后续列为物理错误率
            header = ['p'] + [f"{p:.3f}" for p in p_range]
            data.append(header)

            for i, decoder_info in enumerate(decoder_config):
                # 第一列为 decoder 名字
                decoder_name = decoder_info.get('name')

                block_row = [decoder_name + '_block']
                slq_row = [decoder_name + '_slq']
                not_converge_row = [decoder_name + '_not_converge']
                converge_but_logical_row = [decoder_name + '_converge_but_logical']
                postprocessed_logical_row = [decoder_name + '_postprocessed_logical']
                avg_iter_row = [decoder_name + '_iter']
                avg_time_row = [decoder_name + '_time']

                for p in p_range:
                    decoder_result = result_data.get(i, {}).get(p, {})

                    block_row.append(decoder_result.get('block_error_rate', 'N/A'))
                    if dim > 2:
                        slq_row.append(decoder_result.get('slq_error_rate', 'N/A'))
                    not_converge_row.append(decoder_result.get('not_converge_rate', 'N/A'))
                    converge_but_logical_row.append(decoder_result.get('converge_but_logical_rate', 'N/A'))
                    postprocessed_logical_row.append(decoder_result.get('postprocessed_logical_rate', 'N/A'))
                    avg_iter_row.append(decoder_result.get('avg_iter', 'N/A'))
                    avg_time_row.append(decoder_result.get('avg_time', 'N/A'))

                data.append(block_row)
                if dim > 2:
                    data.append(slq_row)
                data.append(not_converge_row)
                data.append(converge_but_logical_row)
                data.append(postprocessed_logical_row)
                data.append(avg_iter_row)
                data.append(avg_time_row)

                # 添加空行分隔译码器
                data.append([""])

            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
            # 设置所有列的居中对齐
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, len(p_range), None, center_format)


def plot_results(config, dim, results, path):
    """
    根据实验结果绘制block_error_rate （如果 dim > 2 则绘制 slq_error_rate ）。
    Args:
        p_range (list): 物理错误率数组
        config (dict): 配置信息，包含译码器、码类型等
        dim (int): 码的维度
        results (dict): 实验结果，包含不同码大小、译码器、物理错误率下的错误率
        path (str): 保存图像的路径
    """

    p_range = config.get('p_range')
    code_config = config.get('code')
    code_name = code_config.get('name')
    L_range = code_config.get('L_range')
    decoder_config = config.get('decoders')
    decoder_names = [decoder.get('name') for decoder in decoder_config]
    noise_model = config.get('noise_model')
    n_test = config.get('n_test')

    # 检查 code_name 文件夹是否存在
    path = os.path.join(path, code_name)
    if not os.path.exists(path):
        os.makedirs(path)

    decoder_names_str = "_".join(decoder_names)
    path += f"/{L_range}_{decoder_names_str}_{noise_model}_{n_test}times.pdf"


    # 定义不同的颜色和线型
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # 生成颜色数组
    linestyles = ['--', '-.', ':']
    markers = ['s', '^', 'D'] 

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rc('font', size=12)

    num_decoders = len(decoder_config)        

    for i, decoder_info in enumerate(decoder_config):
        if num_decoders == 1 or i == num_decoders - 1:
            linestyle = '-'  # 实线
            marker = 'o'     # 圆点
        else:
            linestyle = linestyles[i % len(linestyles)]  # 根据译码器索引选择线型
            marker = markers[i % len(markers)]          # 根据译码器索引选择标记

        decoder_name = decoder_info.get('name')

        for idx, L in enumerate(L_range):
            color = colors[idx]  # 为每个 L 分配一个颜色

            error_rates = []

            for p in p_range:
                if dim <= 2:
                    error_rate = results[L][i].get(p, {}).get('block_error_rate', 'N/A')
                else:
                    error_rate = results[L][i].get(p, {}).get('slq_error_rate', 'N/A')

                error_rates.append(error_rate)

            plt.plot(p_range, error_rates, label=f"{decoder_name} L={L}", color=color, linestyle=linestyle, marker=marker, linewidth=1.5)

    ax.set_xlabel('Physical Error Rate')
    if dim <= 2:
        ax.set_ylabel('Block Logical Error Rate')
        ax.set_title('Block LER of ' + code_name + ' code under ' + noise_model + ' noise')
    else:
        ax.set_ylabel('Slq Error Rate')
        ax.set_title('Slq Error Rate of ' + code_name + ' code under ' + noise_model + ' noise')

    domain = config.get('p_range')[0]
    if domain == 'linear':
        ax.set_xscale('linear')
        ax.set_yscale('linear')
    elif domain == 'log':
        ax.set_xscale('log')  # X轴使用对数刻度
        ax.set_yscale('log')  # Y轴使用对数刻度
    ax.legend()

    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='grey')  # 主网格
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='grey')  # 次网格

    plt.tight_layout()

    plt.savefig(path)
    plt.show()
