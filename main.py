import argparse
import json
import numpy as np
from runners.runner import QecExpRunner
from utils.plotter import save_results, plot_results


def load_config(config_file, config_name):
    with open(config_file, 'r') as f:
        all_configs = json.load(f)
    config = all_configs.get(config_name, {})

    rx = config.get('rx', 1/3)
    ry = config.get('ry', 1/3)
    rz = config.get('rz', 1/3)
    rs = config.get('rs', 1)
    config['rx'] = rx
    config['ry'] = ry
    config['rz'] = rz
    config['rs'] = rs

    model = config.get('noise_model', 'capacity')
    config['noise_model'] = model

    readout = config.get('readout', 'True')
    config['readout'] = readout

    if model == 'phenomenological':
        code = config.get('code', {})
        L_range = code.get('L_range', [3, 5, 7])
        m_range = code.get('m_range', L_range)
        code['L_range'] = L_range
        code['m_range'] = m_range
        config['code'] = code

    n_test = config.get('n_test', 1000)
    config['n_test'] = n_test

    max_iter = config.get('max_iter', 100)
    config['max_iter'] = max_iter
    
    # 处理 p_range
    p_range = config.get('p_range')
    if not p_range or len(p_range) < 2:
        raise ValueError("Invalid p_range format in config")

    method = p_range[0]
    ranges = p_range[1:]

    final_p_range = []

    if method == 'list':
        for value in ranges[0]:
            final_p_range.append(float(value))

    else:
        for range_triplet in ranges:
            if len(range_triplet) != 3:
                raise ValueError("Each range_triplet must contain exactly three elements: start, stop, and num")
            start, stop, num = range_triplet
            if method == 'linear':
                final_p_range.extend(np.linspace(start, stop, int(num)))
            elif method == 'log':
                if final_p_range == []:
                    part_range = np.logspace(np.log10(start), np.log10(stop), int(num))
                else:
                    part_range = np.logspace(np.log10(start), np.log10(stop), int(num))[1:]
                final_p_range.extend(part_range)
            else:
                raise ValueError("Invalid method for p_range. Supported methods are 'list', 'linear' and 'log'")

    config['p_range'] = np.array(final_p_range)
    return config


def main():
    parser = argparse.ArgumentParser(description="Quantum Error Correction Code Simulation")
    parser.add_argument('--config_path', type=str, default='./config.json', help='Path to config file')
    parser.add_argument('--config', type=str, default='Surface', help='Name of the configuration in the config file')
    parser.add_argument('--save_path', type=str, default='./results/', help='Path to save the results')
    args = parser.parse_args()

    config = load_config(args.config_path, args.config)

    runner = QecExpRunner(config)
    
    noise_model = config.get('noise_model')

    if noise_model == 'capacity':
        results, dim = runner.run_code_capacity()

    elif noise_model == 'phenomenological':
        results, dim = runner.run_phenomenological()
    
    else:
        raise ValueError("Invalid noise model.")
    
    save_results(config, dim, results, args.save_path)
    plot_results(config, dim, results, args.save_path)


if __name__ == '__main__':
    main()
