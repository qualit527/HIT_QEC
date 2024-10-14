# Quantum Error Correction Experimental Framework

## 介绍

该框架用于量子纠错（QEC）译码实验，支持通过定制配置文件快速搭建实验，比较不同的纠错码和译码算法在各种噪声模型下的表现。

## 项目结构

- `main.py`：程序的入口，负责解析 `config.json` 并调用相应的 runner。
- `config.json`：实验配置文件。
- `runners`
  - `./runner.py`: 实验框架，支持在 code capacity noise 和 phenomenological noise下进行实验，并负责记录实验数据。
  - `./code_builder.py`: 管理不同纠错码的实例化。
  - `./decoder_builder`: 实例化自定义译码器的接口，以及封装译码器在实验中的行为。
- `codes`：实现了多种量子纠错码。
- `decoders`：实现（导入）了多种译码器。
- `utils`：辅助函数，如计算逻辑算符、绘图等。

## 1. 配置 (`config.json`)

实验的核心设置存储在 `config.json` 文件中（支持新建json文件，需要在运行时指定路径）。

### 配置文件结构与默认参数

 `config.json` 中包含了多个代码段，每个段落对应一种纠错码的默认配置，包括以下关键内容：

- **`code`：** 定义使用的纠错码。

  - `name`：纠错码的名称，目前支持：`Surface`、`Toric`、`RotatedSurface`、`RotatedToric`、`HGP`（使用随机 LDPC 码构建）、`XZTGRE`、`ZTGRE-HP`、`XYZ3D`、`XYZ4D`。
  - `L_range`：纠错码的迭代参数，使用列表形式，例如：[3, 5, 7]。各纠错码中 `L` 的解释如下：
    - `Surface`、`Toric`、`RotatedSurface`、`RotatedToric`：晶格大小。
    - `HGP`：随机 LDPC 码的码长（目前仅支持 4 的倍数）。
    - `XZTGRE`、`ZTGRE-HP`：Tanner 图递归扩展次数。
    - `XYZ3D`：乘积所用的三个重复码的码长，默认码长相同。
    - `XYZ4D`：形式为 `[[2, 2], [3, 3], [4, 4]]`，用于乘积的重复码码长。
  - `m_range`：**（仅限现象学噪声模型）** 使用列表形式，例如：[3, 5, 7]，含义为各码长对应的稳定子重复测量次数。

- **`decoders`：** 译码器列表。

  - `name`：译码器名称，目前支持：

    - `BP2`、`BP-OSD`：二元域经典置信传播算法，参见 [quantumgizmos/bp_osd](https://github.com/quantumgizmos/bp_osd)；
    - `FDBP`、`FDBP-OSD`：二元域完全解耦置信传播算法 [arxiv:2305.17505](https://arxiv.org/abs/2305.17505)；**（高精度，依赖OSD，$O(N^3)$ ）**
    - `LLRBP`：四元对数域置信传播算法 [TQE 2021](https://ieeexplore.ieee.org/abstract/document/9542859/)；**（高精度，依赖OSD，$O(N^3)$ ）**
    - `MBP`：具有额外记忆效应的 LLRBP [Npj Quantum Inf. 2022](https://www.nature.com/articles/s41534-022-00623-2)；
    - **`AMBP`：** 遍历超参数的 MBP；**（高精度，近似 $O(N)$ ）**
    - `EWA-BP`：自适应先验概率的 LLRBP [arxiv:2407.11523](https://arxiv.org/abs/2407.11523)；
    - **`AEWA-BP`：** 遍历超参数的 EWA-BP；**（高精度，近似 $O(N)$ ）**
    - **`MWPM`：** 最小重完美匹配，参见 [oscarhiggott/PyMatching](https://github.com/oscarhiggott/PyMatching)。**（高精度，仅限拓扑码，**$O(N^3)$）**

  - `params`：**（可选）** 用于设置各译码器的具体参数：

    - `BP2`、`FDBP`：

      ```python
      { "OSD": "True" | "False" }                // 是否使用OSD后处理，默认为False
      ```

    - `LLRBP`、`EWA-BP`：

      ```python
      {
        "schedule": "flooding" | "layer",        // 消息调度方式，默认为flooding
        "init": "Momentum",                      // 等效于EWA-BP
        "method": "Momentum" | "Ada" | "MBP",    // 改变消息更新公式，不支持与init混用
        "alpha": 0.7                             // init或method的超参数
      }
      ```

    - `MBP`：

      ```python
      { "alpha": 0.7  }                          // 控制更新步长的超参数
      ```

    - `AMBP`、`AEWA-BP`：

      ```python
      { "alphas": [1, 0, 11]  }                  // 超参数的遍历范围
      ```

    - `MWPM`：无。

- **`max_iter`：** 置信传播算法的最大迭代次数，默认为100。

- **`noise_model`：** 指定噪声模型（`capacity`、`phenomenological`），默认为 capacity。

- **`p_range`：** 物理错误率区间，可定义为 `["linear", [start, end, steps], [...]]` 或 `["log", [start, end, steps], [...]]`，支持多段区间的拼接。

- **`n_test`：** 仿真次数，默认为1000。

### 示例配置

```json
"RToric": {
    "code": {
        "name": "RotatedToric",
        "L_range": [4, 6, 8],
        "m_range": [2, 4, 6]
    },
    "decoders": [
        {
            "name": "AEWA-BP"
        }
    ],
    "noise_model": "phenomenological",
    "p_range": ["linear", [0.001, 0.04, 14]]
}
```

## 2. 添加新的纠错码类

要添加新的纠错码类，请按照以下步骤进行：

1. **创建纠错码类**：在 `codes/` 目录下创建纠错码类，并在`\__init__.py`中调用，需要能够访问 Hx、Hz、Logical_X、Logical_Z，以及是否为CSS码。

   - 确保新类的构造函数接受必要的参数，如 `L`（晶格大小）。

2. **更新 Code Builder**：

   - 打开 `runners/code_builder.py` ；

   - 在 `build_code()` 函数中添加对新纠错码的支持。示例代码：

     ```python
     def build_code(code_name, L):
         if code_name == "Toric":
             toric = ToricCode(L)
             return toric.hx, toric.hz, toric.lx, toric.lz, toric.k, True
         # 现有的其他纠错码类型...
     ```

## 3. 添加新的译码算法

要添加新的译码算法，请按照以下步骤进行：

1. **创建译码器类**：在 `decoders/` 目录下创建一个新的译码类（或安装现有的译码器），并在`\__init__.py`中调用。该类应实现以下方法：

   - `decode(syndrome)`: 接收错误综合作为输入，并输出错误估计。

2. **更新 Decoder Builder**：

   - 打开 `runners/decoder_builder.py` ；

   - 在 `build_decoder()` 函数中添加译码器的实例化方法。示例代码：

     ```python
     def build_decoder(noise_model, decoder_config, Hx, Hz, dim, px, py, pz, Hs=None, ps=0, max_iter=100):
         if name in ["Matching", "MWPM"]:
             weights_x = np.full(Hx.shape[1], np.log((1 - (px + py)) / (px + py)))
             weights_z = np.full(Hz.shape[1], np.log((1 - (pz + py)) / (pz + py)))
     
             decoders.append(Matching.from_check_matrix(np.hstack([Hx, Hz]), np.hstack([weights_x, weights_z])))
             
         # 现有的其他译码器类型...
     ```

   - 在 `run_decoder()` 函数中封装译码器行为。示例代码：

     ```python
     def run_decoder(name, decoder, syndrome, code_length, params, noise_model):
         if name in ["Matching", "MWPM"]:
                 correction = decoder.decode(syndrome)
                 correction_z = correction[0:code_length]
                 correction_x = correction[code_length:2 * code_length]
                 time_cost = time.time() - start_time
     
                 return [correction_x, correction_z], time_cost, 0, True
         # ...
     ```

## 运行实验

1. 定制 `config.json` 中的设置。

2. 运行 `main.py` 启动实验：

   ```bash
   python main.py --config=Toric
   ```

   --config默认为Surface，此外可选： (--config_path CONFIG_PATH) (--save_path SAVE_PATH)。

3. 在 `./results/CODE_NAME` 下找到仿真结果图像和数据。

## 结果示例

- Surface code，code capacity noise，MWPM vs FDBP-OSD vs AEWA-BP

![image-20241014010501971](https://raw.githubusercontent.com/qualit527/FigureBed/main/image-20241014010501971.png)

- XZTGRE，code capacity noise，FDBP-OSD vs AEWA-BP

![image-20241014010426637](https://raw.githubusercontent.com/qualit527/FigureBed/main/image-20241014010426637.png)
