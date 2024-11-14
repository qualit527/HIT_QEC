// src/LLRBp4Decoder.cpp

#include "LLRBp4Decoder.h"
#include <algorithm>
#include <iostream>
#include <fstream>

LLRBp4Decoder::LLRBp4Decoder(const Eigen::MatrixXi& Hx_,
                             const Eigen::MatrixXi& Hz_,
                             double px_,
                             double py_,
                             double pz_,
                             int max_iter_,
                             const Eigen::MatrixXi& Hs_,
                             double ps_,
                             int dimension)
    : Hx(Hx_), Hz(Hz_), Hs(Hs_), px(px_), py(py_), pz(pz_), ps(ps_),
      pi(1.0 - px_ - py_ - pz_), k(dimension), max_iter(max_iter_), flag(true)
{
    H = Hx + Hz;
    m = static_cast<int>(H.rows());     // 显式转换 Eigen::Index 到 int
    n = static_cast<int>(H.cols());     // 显式转换 Eigen::Index 到 int
    s = Hs_.size() > 0 ? static_cast<int>(Hs_.cols()) : 0; // 显式转换 Eigen::Index 到 int

    // Initialize binary_H
    binary_H = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(m, 2 * n + s);
    binary_H.block(0, 0, m, n) = (Hz.array() != 0).cast<uint8_t>();
    binary_H.block(0, n, m, n) = (Hx.array() != 0).cast<uint8_t>();
    if (s > 0) {
        binary_H.block(0, 2 * n, m, s) = (Hs.array() != 0).cast<uint8_t>();
    }

    // Initialize Q matrices
    const double log_pi_over_px = std::log(pi / px);
    const double log_pi_over_py = std::log(pi / py);
    const double log_pi_over_pz = std::log(pi / pz);
    const double log_pi_over_ps = (s > 0) ? std::log(pi / ps) : 0.0;

    Q_matrix_X = (H.array() != 0).cast<double>() * log_pi_over_px;
    Q_matrix_Y = (H.array() != 0).cast<double>() * log_pi_over_py;
    Q_matrix_Z = (H.array() != 0).cast<double>() * log_pi_over_pz;

    if (s > 0) {
        Q_matrix_S = (Hs.array() != 0).cast<double>() * log_pi_over_ps;
    }

    // Initialize d_message and d_message_ds
    d_message = Eigen::MatrixXd::Zero(m, n);
    d_message_ds = Eigen::MatrixXd::Zero(m, n + s);

    // Masks
    Eigen::MatrixXi mask_X = ((Hx.array() == 1) && (Hz.array() != 1)).cast<int>();
    Eigen::MatrixXi mask_Z = ((Hx.array() != 1) && (Hz.array() == 1)).cast<int>();
    Eigen::MatrixXi mask_Y = ((Hx.array() == 1) && (Hz.array() == 1)).cast<int>();

    // Apply lambda_func
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            if(mask_X(i,j)){
                d_message(i,j) = lambda_func(PauliType::X, Q_matrix_X(i,j), Q_matrix_Y(i,j), Q_matrix_Z(i,j));
            }
            else if(mask_Z(i,j)){
                d_message(i,j) = lambda_func(PauliType::Z, Q_matrix_X(i,j), Q_matrix_Y(i,j), Q_matrix_Z(i,j));
            }
            else if(mask_Y(i,j)){
                d_message(i,j) = lambda_func(PauliType::Y, Q_matrix_X(i,j), Q_matrix_Y(i,j), Q_matrix_Z(i,j));
            }
        }
    }

    // Initialize d_message_ds
    d_message_ds.block(0, 0, m, n) = d_message;
    if(s > 0){
        d_message_ds.block(0, n, m, s) = Q_matrix_S;
    }

    // Initialize delta_message and delta_message_ds
    delta_message = Eigen::MatrixXd::Zero(m, n);
    delta_message_ds = Eigen::MatrixXd::Zero(m, n + s);

    // 预计算 H 和 Hs 的非零索引
    H_rows_nonzero_cols.resize(m, std::vector<int>());
    H_cols_nonzero_rows.resize(n + s, std::vector<int>());

    for(int row = 0; row < m; ++row){
        for(int col = 0; col < n; ++col){
            if(H(row, col) != 0){
                H_rows_nonzero_cols[row].push_back(col);
                H_cols_nonzero_rows[col].push_back(row);
            }
        }
        if(s > 0){
            for(int col = 0; col < s; ++col){
                if(Hs(row, col) != 0){
                    H_rows_nonzero_cols[row].push_back(col + n);
                    H_cols_nonzero_rows[col + n].push_back(row);
                }
            }
        }
    }

}


// 定义一个通用的模 2 函数
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> mod2(const Eigen::MatrixBase<Derived>& mat) {
    return mat.unaryExpr([](typename Derived::Scalar x) -> typename Derived::Scalar {
        return x % 2;
    });
}

inline double LLRBp4Decoder::lambda_func(const PauliType W, double px_val, double py_val, double pz_val){
    double numerator, denominator;

    switch(W) {
        case PauliType::X:
            numerator = 1.0 + std::exp(-px_val);
            denominator = std::exp(-py_val) + std::exp(-pz_val);
            break;
        case PauliType::Y:
            numerator = 1.0 + std::exp(-py_val);
            denominator = std::exp(-px_val) + std::exp(-pz_val);
            break;
        case PauliType::Z:
            numerator = 1.0 + std::exp(-pz_val);
            denominator = std::exp(-px_val) + std::exp(-py_val);
            break;
        default:
            throw std::invalid_argument("Invalid PauliType. Expected X, Y, or Z.");
    }

    return std::log(numerator / denominator);
}


std::tuple<std::vector<int>, bool, int> LLRBp4Decoder::standard_decoder(const Eigen::VectorXi& syndrome,
                                                                        ScheduleType schedule,
                                                                        InitType init,
                                                                        MethodType method,
                                                                        OSDType OSD,
                                                                        double alpha,
                                                                        double beta,
                                                                        int test){
    // 复制成员变量
    Eigen::MatrixXd d_message_local = (s > 0) ? d_message_ds : d_message; // 根据是否存在 Hs 选择使用哪个消息矩阵
    Eigen::MatrixXd delta_message_local = (s > 0) ? delta_message_ds : delta_message;

    // 初始化 Error，大小为 2n+s
    std::vector<int> Error(2 * n + s, 0); 

    // 初始化可靠性
    Eigen::VectorXd reliability = Eigen::VectorXd::Ones(n + s); // 每个比特的历史可靠性

    // 初始化 qX, qY, qZ, qS（现象学噪声）
    Eigen::VectorXd qX = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd qY = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd qZ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd qS = Eigen::VectorXd::Zero((s > 0) ? s : 1); // 如果 s == 0，则至少有一个元素

    // 第0次软判决向量
    Eigen::VectorXd qX_0 = Eigen::VectorXd::Constant(n, std::log(pi / px));
    Eigen::VectorXd qY_0 = Eigen::VectorXd::Constant(n, std::log(pi / py));
    Eigen::VectorXd qZ_0 = Eigen::VectorXd::Constant(n, std::log(pi / pz));
    Eigen::VectorXd qS_0 = Eigen::VectorXd::Constant((s > 0) ? s : 1, std::log(pi / ps));

    // 初始化动量相关变量
    Eigen::VectorXd last_qX = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd last_qY = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd last_qZ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd last_qS = Eigen::VectorXd::Zero((s > 0) ? s : 1);

    Eigen::MatrixXd gt = Eigen::MatrixXd::Zero(n, 3);
    Eigen::MatrixXd Vt = Eigen::MatrixXd::Zero(n, 3);
    Eigen::VectorXd gt_S = Eigen::VectorXd::Zero((s > 0) ? s : 1);

    Eigen::VectorXd last_qX_q = qX_0;
    Eigen::VectorXd last_qY_q = qY_0;
    Eigen::VectorXd last_qZ_q = qZ_0;
    Eigen::VectorXd last_qS_q = qS_0;

    Eigen::MatrixXd gt_q = Eigen::MatrixXd::Zero(n, 3);
    Eigen::MatrixXd Vt_q = Eigen::MatrixXd::Zero(n, 3);
    Eigen::VectorXd gt_S_q = Eigen::VectorXd::Zero((s > 0) ? s : 1); // 如果 s == 0，则至少有一个元素
    Eigen::VectorXd Vt_S_q = Eigen::VectorXd::Zero((s > 0) ? s : 1);

    // 振荡向量
    Eigen::VectorXd oscillation = Eigen::VectorXd::Zero(n + s);

    // 初始化最后的错误
    std::vector<int> last_Error(2 * n + s, 0);

    std::vector<double> tanh_values_buffer; // 缓冲区用于 tanh 值
    tanh_values_buffer.reserve(n + s);

    // 开始迭代
    for(int iter = 0; iter < max_iter; ++iter){
        // 初始化 Error
        std::fill(Error.begin(), Error.end(), 0); // Error = [Ex, Ez]

        if(schedule == ScheduleType::FLOODING){
            // 水平更新
            for(int j = 0; j < m; ++j){
                // 计算 tanh_values
                size_t num_indices = H_rows_nonzero_cols[j].size();

                tanh_values_buffer.resize(num_indices);
                for(size_t k = 0; k < num_indices; ++k){
                    tanh_values_buffer[k] = std::tanh(d_message_local(j, H_rows_nonzero_cols[j][k]) / 2.0) + 1e-10;
                }

                // 计算乘积项
                double product_term = 1.0;
                for(const auto& val : tanh_values_buffer){
                    product_term *= val;
                }

                double syndrome_factor = (syndrome(j) == 1) ? -1.0 : 1.0;

                // 更新 delta_message
                for(size_t k = 0; k < num_indices; ++k){
                    double product_excluding_current = product_term / tanh_values_buffer[k];
                    // Clip the value to avoid numerical issues
                    product_excluding_current = std::clamp(product_excluding_current, -1.0 + 1e-10, 1.0 - 1e-10);
                    double delta_value = 2.0 * std::atanh(product_excluding_current);
                    delta_message_local(j, H_rows_nonzero_cols[j][k]) = syndrome_factor * delta_value;
                }
            }

            // 垂直更新
            for(int j = 0; j < n + s; ++j){
                if(j < n){
                    qX(j) = qX_0(j);
                    qY(j) = qY_0(j);
                    qZ(j) = qZ_0(j);
                }
                else{
                    // 初始化 qS[j - n]
                    qS(j - n) = qS_0(j - n);
                }

                // 处理 init 方法
                if(init == InitType::MOMENTUM){
                    if(test == 1){
                        if(iter > 0){
                            qX(j) = alpha * qX(j) + (1.0 - alpha) * last_qX(j);
                            qY(j) = alpha * qY(j) + (1.0 - alpha) * last_qY(j);
                            qZ(j) = alpha * qZ(j) + (1.0 - alpha) * last_qZ(j);
                        }
                    }

                    else{
                        if(j < n){
                            qX(j) = alpha * qX(j) + (1.0 - alpha) * last_qX(j);
                            qY(j) = alpha * qY(j) + (1.0 - alpha) * last_qY(j);
                            qZ(j) = alpha * qZ(j) + (1.0 - alpha) * last_qZ(j);

                            if(j == 0){
                                last_qX(n-1) = qX(n-1);
                                last_qY(n-1) = qY(n-1);
                                last_qZ(n-1) = qZ(n-1);
                            }
                            else{
                                last_qX(j-1) = qX(j-1);
                                last_qY(j-1) = qY(j-1);
                                last_qZ(j-1) = qZ(j-1);
                            }
                        }
                        else{
                            // 更新动量项 for qS
                            gt_S(j - n) = last_qS(j - n) - qS(j - n);
                            qS(j - n) = last_qS(j - n) - alpha * gt_S(j - n);

                            if(j == n){
                                last_qS(s-1) = qS(s-1);
                            }
                            else{
                                last_qS(j-1) = qS(j-1);
                            }
                        }
                    }
                    // // 更新 last_qS
                    // for(int k = 0; k < s; ++k){
                    //     last_qS(k) = qS(k);
                    // }
                }

                else if(init != InitType::NONE){
                    throw std::invalid_argument("Invalid init method. Expected 'Momentum'.");
                }


                // 软判决向量
                double sum_qX_summands = 0.0;
                double sum_qY_summands = 0.0;
                double sum_qZ_summands = 0.0;
                double sum_qS_summands = 0.0;

                if(j < n){
                    for(auto &row : H_cols_nonzero_rows[j]){
                        if(Hz(row, j) == 1){
                            sum_qX_summands += delta_message_local(row, j);
                        }
                        if(Hx(row, j) != Hz(row, j)){
                            sum_qY_summands += delta_message_local(row, j);
                        }
                        if(Hx(row, j) == 1){
                            sum_qZ_summands += delta_message_local(row, j);
                        }
                    }

                    qX(j) += sum_qX_summands;
                    qY(j) += sum_qY_summands;
                    qZ(j) += sum_qZ_summands;
                }
                else{
                    // 处理 qS 的软判决
                    for(auto &row : H_cols_nonzero_rows[j]){
                        if(Hs(row, j - n) == 1){
                            sum_qS_summands += delta_message_local(row, j);
                        }
                    }
                    qS(j - n) += sum_qS_summands;
                }

                // 处理 method 方法
                if(method == MethodType::MOMENTUM){
                    if(j < n){
                        // 更新动量项 for qX, qY, qZ
                        gt_q(j, 0) = beta * gt_q(j, 0) + (1.0 - beta) * (last_qX_q(j) - qX(j));
                        gt_q(j, 1) = beta * gt_q(j, 1) + (1.0 - beta) * (last_qY_q(j) - qY(j));
                        gt_q(j, 2) = beta * gt_q(j, 2) + (1.0 - beta) * (last_qZ_q(j) - qZ(j));

                        qX(j) = last_qX_q(j) - alpha * gt_q(j, 0);
                        qY(j) = last_qY_q(j) - alpha * gt_q(j, 1);
                        qZ(j) = last_qZ_q(j) - alpha * gt_q(j, 2);

                        if(j == 0){
                            last_qX_q(n-1) = qX(n-1);
                            last_qY_q(n-1) = qY(n-1);
                            last_qZ_q(n-1) = qZ(n-1);
                        }
                        else{
                            last_qX_q(j-1) = qX(j-1);
                            last_qY_q(j-1) = qY(j-1);
                            last_qZ_q(j-1) = qZ(j-1);
                        }
                    }
                    else{
                        // 更新动量项 for qS
                        gt_S_q(j - n) = beta * gt_S_q(j - n) + (1.0 - beta) * (last_qS_q(j - n) - qS(j - n));
                        qS(j - n) = last_qS_q(j - n) - alpha * gt_S_q(j - n);
                    }

                    // 更新 last_qS
                    for(int k = 0; k < s; ++k){
                        last_qS_q(k) = qS(k);
                    }
                }

                else if(method == MethodType::ADA){
                    if(j < n){
                        // AdaGrad 更新 for qX, qY, qZ
                        gt_q(j, 0) = last_qX_q(j) - qX(j);
                        gt_q(j, 1) = last_qY_q(j) - qY(j);
                        gt_q(j, 2) = last_qZ_q(j) - qZ(j);

                        Vt(j, 0) += std::pow(gt_q(j, 0), 2);
                        Vt(j, 1) += std::pow(gt_q(j, 1), 2);
                        Vt(j, 2) += std::pow(gt_q(j, 2), 2);

                        if(iter != 0){
                            qX(j) = last_qX_q(j) - (alpha * gt_q(j, 0)) / (std::sqrt(Vt(j, 0)) + 1e-10);
                            qY(j) = last_qY_q(j) - (alpha * gt_q(j, 1)) / (std::sqrt(Vt(j, 1)) + 1e-10);
                            qZ(j) = last_qZ_q(j) - (alpha * gt_q(j, 2)) / (std::sqrt(Vt(j, 2)) + 1e-10);
                        }

                        if(j == 0){
                            last_qX_q(n-1) = qX(n-1);
                            last_qY_q(n-1) = qY(n-1);
                            last_qZ_q(n-1) = qZ(n-1);
                        }
                        else{
                            last_qX_q(j-1) = qX(j-1);
                            last_qY_q(j-1) = qY(j-1);
                            last_qZ_q(j-1) = qZ(j-1);
                        }
                    }
                    else{
                        // AdaGrad 更新 for qS
                        gt_S_q(j - n) = last_qS_q(j - n) - qS(j - n);
                        Vt_S_q(j - n) += std::pow(gt_S_q(j - n), 2);

                        if(iter != 0){
                            qS(j - n) = last_qS_q(j - n) - (alpha * gt_S_q(j - n)) / (std::sqrt(Vt_S_q(j - n)) + 1e-10);
                        }
                    }

                    // 更新 last_qS
                    for(int k = 0; k < s; ++k){
                        last_qS_q(k) = qS(k);
                    }
                }

                else if(method == MethodType::MBP){
                    if(j < n){
                        qX(j) = qX(j) - sum_qX_summands + (sum_qX_summands / alpha);
                        qY(j) = qY(j) - sum_qY_summands + (sum_qY_summands / alpha);
                        qZ(j) = qZ(j) - sum_qZ_summands + (sum_qZ_summands / alpha);
                    }
                    else{
                        // MBP 更新 for qS
                        qS(j - n) = qS(j - n) - sum_qS_summands + (sum_qS_summands / alpha);
                    }
                }
                else if(method != MethodType::NONE){
                    throw std::invalid_argument("Invalid method. Expected 'Momentum', 'Ada', or 'MBP'.");
                }

                // 更新 d_message
                if(j < n){
                    for(auto &row : H_cols_nonzero_rows[j]){
                        if((Hx(row, j) == 1) && (Hz(row, j) == 0)){
                            d_message_local(row, j) = lambda_func(PauliType::X, qX(j), qY(j) - delta_message_local(row, j), qZ(j) - delta_message_local(row, j));
                        }
                        else if((Hx(row, j) == 1) && (Hz(row, j) == 1)){
                            d_message_local(row, j) = lambda_func(PauliType::Y, qX(j) - delta_message_local(row, j), qY(j), qZ(j) - delta_message_local(row, j));
                        }
                        else if((Hx(row, j) == 0) && (Hz(row, j) == 1)){
                            d_message_local(row, j) = lambda_func(PauliType::Z, qX(j) - delta_message_local(row, j), qY(j) - delta_message_local(row, j), qZ(j));
                        }
                    }
                }
                else{
                    // 更新 d_message_ds 中的 qS 部分
                    for(auto &row : H_cols_nonzero_rows[j]){
                        d_message_local(row, j) = qS(j - n) - delta_message_local(row, j);
                    }
                }

                // 硬判决
                if(j < n){
                    Eigen::Vector3d q_values(qX(j), qY(j), qZ(j));
                    int indx;
                    q_values.minCoeff(&indx);

                    if(qX(j) > 0 && qY(j) > 0 && qZ(j) > 0){
                        Error[j] = 0;
                        Error[j + n] = 0;
                    }
                    else{
                        switch(indx){
                            case 0:
                                Error[j] = 1;
                                Error[j + n] = 0;
                                break;
                            case 1:
                                Error[j] = 1;
                                Error[j + n] = 1;
                                break;
                            case 2:
                                Error[j] = 0;
                                Error[j + n] = 1;
                                break;
                            default:
                                // 处理异常情况
                                Error[j] = 0;
                                Error[j + n] = 0;
                                break;
                        }
                    }
                }
                else{
                    if(qS(j - n) <= 0){
                        Error[j + n] = 1;
                    }
                    else{
                        Error[j + n] = 0;
                    }
                }    

                // // 更新可靠性或振荡
                // if(j < n){
                //     if((last_Error[j] == Error[j]) && (last_Error[j + n] == Error[j + n])){
                //         reliability(j) += 1.0;
                //     }
                //     else{
                //         oscillation(j) += 1.0;
                //         reliability(j) = 1.0;
                //     }
                // }
                // else{
                //     if(last_Error[j + n] == Error[j + n]){
                //         reliability(j) += 1.0;
                //     }
                //     else{
                //         oscillation(j) += 1.0;
                //         reliability(j) = 1.0;
                //     }
                // }
            }
            last_Error = Error; // 复制当前错误到最后错误
        }

        else if(schedule == ScheduleType::LAYER){
            // 变量节点层级调度
            for(int j = 0; j < n + s; ++j){
                // 水平更新
                for(auto &m_row : H_cols_nonzero_rows[j]){
                    // 找出 H(m_row, :) != 0 且 != j
                    std::vector<int> index_m;
                    for(int col = 0; col < n; ++col){
                        if(H(m_row, col) != 0 && col != j){
                            index_m.push_back(col);
                        }
                    }

                    if(s > 0){
                        for(int col = 0; col < s; ++col){
                            if(Hs(m_row, col) != 0 && (col + n) != j){
                                index_m.push_back(col + n);
                            }
                        }
                    }

                    // 计算 tanh_values
                    tanh_values_buffer.resize(index_m.size());
                    for(size_t k = 0; k < index_m.size(); ++k){
                        tanh_values_buffer[k] = std::tanh(d_message_local(m_row, index_m[k]) / 2.0) + 1e-10;
                    }

                    // 计算乘积项
                    double product_term = 1.0;
                    for(const auto& val : tanh_values_buffer){
                        product_term *= val;
                    }
                    product_term = std::clamp(product_term, -1.0 + 1e-10, 1.0 - 1e-10);

                    double syndrome_factor = (syndrome(m_row) == 1) ? -1.0 : 1.0;

                    // 计算 delta_value
                    double delta_value = 2.0 * std::atanh(product_term);
                    delta_message_local(m_row, j) = syndrome_factor * delta_value;
                }

                // 垂直更新
                if(j < n){
                    // 初始化 qX[j], qY[j], qZ[j]
                    qX(j) = qX_0(j);
                    qY(j) = qY_0(j);
                    qZ(j) = qZ_0(j);
                }
                else{
                    // 初始化 qS[j - n]
                    qS(j - n) = qS_0(j - n);
                }

                // 处理 init 方法
                if(init == InitType::MOMENTUM){
                    // 更新动量项
                    if(test == 1){
                        if(iter > 0){
                            qX(j) = alpha * qX(j) + (1.0 - alpha) * last_qX(j);
                            qY(j) = alpha * qY(j) + (1.0 - alpha) * last_qY(j);
                            qZ(j) = alpha * qZ(j) + (1.0 - alpha) * last_qZ(j);

                            if(j == 0){
                                last_qX(n-1) = qX(n-1);
                                last_qY(n-1) = qY(n-1);
                                last_qZ(n-1) = qZ(n-1);
                            }
                            else{
                                last_qX(j-1) = qX(j-1);
                                last_qY(j-1) = qY(j-1);
                                last_qZ(j-1) = qZ(j-1);
                            }
                        }
                    }

                    else{
                        if(j < n){
                            // 更新动量项
                            gt(j, 0) = beta * gt(j, 0) + (1.0 - beta) * (last_qX(j) - qX(j));
                            gt(j, 1) = beta * gt(j, 1) + (1.0 - beta) * (last_qY(j) - qY(j));
                            gt(j, 2) = beta * gt(j, 2) + (1.0 - beta) * (last_qZ(j) - qZ(j));

                            qX(j) = last_qX(j) - alpha * gt(j, 0);
                            qY(j) = last_qY(j) - alpha * gt(j, 1);
                            qZ(j) = last_qZ(j) - alpha * gt(j, 2);

                            if(j == 0){
                                last_qX(n-1) = qX(n-1);
                                last_qY(n-1) = qY(n-1);
                                last_qZ(n-1) = qZ(n-1);
                            }
                            else{
                                last_qX(j-1) = qX(j-1);
                                last_qY(j-1) = qY(j-1);
                                last_qZ(j-1) = qZ(j-1);
                            }
                        }
                        else{
                            // 更新动量项 for qS
                            gt_S(j - n) = beta * gt_S(j - n) + (1.0 - beta) * (last_qS(j - n) - qS(j - n));
                            qS(j - n) = last_qS(j - n) - alpha * gt_S(j - n);
                        }
                    }

                    // 更新 last_qS
                    for(int k = 0; k < s; ++k){
                        last_qS(k) = qS(k);
                    }
                }
                else if(init != InitType::NONE){
                    throw std::invalid_argument("Invalid init method. Expected 'Momentum'.");
                }

                // 软判决向量
                double sum_qX_summands = 0.0;
                double sum_qY_summands = 0.0;
                double sum_qZ_summands = 0.0;
                double sum_qS_summands = 0.0;

                if(j < n){
                    for(auto &row : H_cols_nonzero_rows[j]){
                        if(Hz(row, j) == 1){
                            sum_qX_summands += delta_message_local(row, j);
                        }
                        if(Hx(row, j) != Hz(row, j)){
                            sum_qY_summands += delta_message_local(row, j);
                        }
                        if(Hx(row, j) == 1){
                            sum_qZ_summands += delta_message_local(row, j);
                        }
                    }

                    qX(j) += sum_qX_summands;
                    qY(j) += sum_qY_summands;
                    qZ(j) += sum_qZ_summands;
                }
                else{
                    // 处理 qS 的软判决
                    for(auto &row : H_cols_nonzero_rows[j]){
                        if(Hs(row, j - n) == 1){
                            sum_qS_summands += delta_message_local(row, j);
                        }
                    }
                    qS(j - n) += sum_qS_summands;
                }

                // 处理 method 方法
                if(method == MethodType::MOMENTUM){
                    if(j < n){
                        // 更新动量项 for qX, qY, qZ
                        gt_q(j, 0) = beta * gt_q(j, 0) + (1.0 - beta) * (last_qX_q(j) - qX(j));
                        gt_q(j, 1) = beta * gt_q(j, 1) + (1.0 - beta) * (last_qY_q(j) - qY(j));
                        gt_q(j, 2) = beta * gt_q(j, 2) + (1.0 - beta) * (last_qZ_q(j) - qZ(j));

                        qX(j) = last_qX_q(j) - alpha * gt_q(j, 0);
                        qY(j) = last_qY_q(j) - alpha * gt_q(j, 1);
                        qZ(j) = last_qZ_q(j) - alpha * gt_q(j, 2);

                        if(j == 0){
                            last_qX_q(n-1) = qX(n-1);
                            last_qY_q(n-1) = qY(n-1);
                            last_qZ_q(n-1) = qZ(n-1);
                        }
                        else{
                            last_qX_q(j-1) = qX(j-1);
                            last_qY_q(j-1) = qY(j-1);
                            last_qZ_q(j-1) = qZ(j-1);
                        }
                    }
                    else{
                        // 更新动量项 for qS
                        gt_S_q(j - n) = beta * gt_S_q(j - n) + (1.0 - beta) * (last_qS_q(j - n) - qS(j - n));
                        qS(j - n) = last_qS_q(j - n) - alpha * gt_S_q(j - n);
                    }

                    // 更新 last_qS
                    for(int k = 0; k < s; ++k){
                        last_qS_q(k) = qS(k);
                    }
                }

                else if(method == MethodType::ADA){
                    if(j < n){
                        // AdaGrad 更新 for qX, qY, qZ
                        gt_q(j, 0) = last_qX_q(j) - qX(j);
                        gt_q(j, 1) = last_qY_q(j) - qY(j);
                        gt_q(j, 2) = last_qZ_q(j) - qZ(j);

                        Vt(j, 0) += std::pow(gt_q(j, 0), 2);
                        Vt(j, 1) += std::pow(gt_q(j, 1), 2);
                        Vt(j, 2) += std::pow(gt_q(j, 2), 2);

                        if(iter != 0){
                            qX(j) = last_qX_q(j) - (alpha * gt_q(j, 0)) / (std::sqrt(Vt(j, 0)) + 1e-10);
                            qY(j) = last_qY_q(j) - (alpha * gt_q(j, 1)) / (std::sqrt(Vt(j, 1)) + 1e-10);
                            qZ(j) = last_qZ_q(j) - (alpha * gt_q(j, 2)) / (std::sqrt(Vt(j, 2)) + 1e-10);
                        }

                        if(j == 0){
                            last_qX_q(n-1) = qX(n-1);
                            last_qY_q(n-1) = qY(n-1);
                            last_qZ_q(n-1) = qZ(n-1);
                        }
                        else{
                            last_qX_q(j-1) = qX(j-1);
                            last_qY_q(j-1) = qY(j-1);
                            last_qZ_q(j-1) = qZ(j-1);
                        }
                    }
                    else{
                        // AdaGrad 更新 for qS
                        gt_S_q(j - n) = last_qS_q(j - n) - qS(j - n);
                        Vt_S_q(j - n) += std::pow(gt_S_q(j - n), 2);

                        if(iter != 0){
                            qS(j - n) = last_qS_q(j - n) - (alpha * gt_S_q(j - n)) / (std::sqrt(Vt_S_q(j - n)) + 1e-10);
                        }
                    }

                    // 更新 last_qS
                    for(int k = 0; k < s; ++k){
                        last_qS_q(k) = qS(k);
                    }
                }

                else if(method == MethodType::MBP){
                    if(j < n){
                        qX(j) = qX(j) - sum_qX_summands + (sum_qX_summands / alpha);
                        qY(j) = qY(j) - sum_qY_summands + (sum_qY_summands / alpha);
                        qZ(j) = qZ(j) - sum_qZ_summands + (sum_qZ_summands / alpha);
                    }
                    else{
                        // MBP 更新 for qS
                        qS(j - n) = qS(j - n) - sum_qS_summands + (sum_qS_summands / alpha);
                    }
                }

                else if(method != MethodType::NONE){
                    throw std::invalid_argument("Invalid method. Expected 'Momentum', 'Ada', or 'MBP'.");
                }

                // 更新 d_message
                if(j < n){
                    for(auto &row : H_cols_nonzero_rows[j]){
                        if((Hx(row, j) == 1) && (Hz(row, j) == 0)){
                            d_message_local(row, j) = lambda_func(PauliType::X, qX(j), qY(j) - delta_message_local(row, j), qZ(j) - delta_message_local(row, j));
                        }
                        else if((Hx(row, j) == 1) && (Hz(row, j) == 1)){
                            d_message_local(row, j) = lambda_func(PauliType::Y, qX(j) - delta_message_local(row, j), qY(j), qZ(j) - delta_message_local(row, j));
                        }
                        else if((Hx(row, j) == 0) && (Hz(row, j) == 1)){
                            d_message_local(row, j) = lambda_func(PauliType::Z, qX(j) - delta_message_local(row, j), qY(j) - delta_message_local(row, j), qZ(j));
                        }
                    }
                }
                else{
                    // 更新 d_message_ds 中的 qS 部分
                    for(auto &row : H_cols_nonzero_rows[j]){
                        d_message_local(row, j) = qS(j - n) - delta_message_local(row, j);
                    }
                }
            }

            // 硬判决
            for(int j = 0; j < n; ++j){
                std::vector<double> list = {qX(j), qY(j), qZ(j)};
                double min_value = *std::min_element(list.begin(), list.end());
                int indx = std::distance(list.begin(), std::min_element(list.begin(), list.end()));

                if(qX(j) > 0 && qY(j) > 0 && qZ(j) > 0){
                    Error[j] = 0;
                    Error[j + n] = 0;
                }
                else{
                    switch(indx){
                        case 0:
                            Error[j] = 1;
                            Error[j + n] = 0;
                            break;
                        case 1:
                            Error[j] = 1;
                            Error[j + n] = 1;
                            break;
                        case 2:
                            Error[j] = 0;
                            Error[j + n] = 1;
                            break;
                        default:
                            // 处理异常情况
                            Error[j] = 0;
                            Error[j + n] = 0;
                            break;
                    }
                }

                // // 更新可靠性或振荡
                // if((last_Error[j] == Error[j]) && (last_Error[j + n] == Error[j + n])){
                //     reliability(j) += 1.0;
                // }
                // else{
                //     oscillation(j) += 1.0;
                //     reliability(j) = 1.0;
                // }
            }

            // 处理 s 个稳定子的硬判决（仅适用于现象学译码）
            if(s > 0){
                for(int j = 0; j < s; ++j){
                    if(qS(j) <= 0){
                        Error[2 * n + j] = 1;
                    }
                }
            }

            //         if(last_Error[j + 2 * n] == Error[j + 2 * n]){
            //             reliability(j) += 1.0;
            //         }
            //         else{
            //             oscillation(j) += 1.0;
            //             reliability(j) = 1.0;
            //         }                    
            //     }
            // }

            last_Error = Error; // 复制当前错误到最后错误
        }

        else{
            throw std::invalid_argument("Invalid schedule. Expected 'flooding' or 'layer'.");
        }

        // 检查是否满足解码条件
        // (Hz * Error_X + Hx * Error_Z) % 2 == syndrome
        // 其中 Error_X = Error[0:n], Error_Z = Error[n:2n]
        Eigen::VectorXi Error_X = Eigen::VectorXi::Zero(n);
        Eigen::VectorXi Error_Z = Eigen::VectorXi::Zero(n);
        Eigen::VectorXi Error_S = Eigen::VectorXi::Zero((s > 0) ? s : 1);

        for(int j = 0; j < n; ++j){
            Error_X(j) = Error[j];
            Error_Z(j) = Error[j + n];
        }
        if(s > 0){
            for(int j = 0; j < s; ++j){
                Error_S(j) = Error[2 * n + j];
            }
        }

        // 计算综合症
        Eigen::VectorXi syndrome_computed = (Hz * Error_X + Hx * Error_Z);
        if(s > 0){
            syndrome_computed += (Hs * Error_S);
        }
        syndrome_computed = mod2(syndrome_computed);

        if((syndrome_computed.array() == syndrome.array()).all()){
            this->flag = true;
            return std::make_tuple(Error, this->flag, iter + 1);
        }
    }

    // 如果达到最大迭代次数后，进行 OSD 后处理
    bool decoding_success = false;
    int final_iter = max_iter;

    if(OSD == OSDType::BINARY){
        // 计算 px, py, pz
        Eigen::VectorXd px_prob = (1.0 / (qX.array().exp() + 1.0)).matrix();
        Eigen::VectorXd py_prob = (1.0 / (qY.array().exp() + 1.0)).matrix();
        Eigen::VectorXd pz_prob = (1.0 / (qZ.array().exp() + 1.0)).matrix();
        Eigen::VectorXd ps_prob = (s > 0) ? (1.0 / (qS.array().exp() + 1.0)).matrix() : Eigen::VectorXd();

        // 计算 binary_X 和 binary_Z
        Eigen::VectorXd binary_X = px_prob + py_prob;
        Eigen::VectorXd binary_Z = pz_prob + py_prob;

        // 组合概率
        Eigen::VectorXd probability;
        if(s > 0){
            Eigen::VectorXd binary_S = ps_prob;
            probability.resize(2 * n + s);
            probability << -binary_X, -binary_Z, -binary_S;
        }
        else{
            probability.resize(2 * n);
            probability << -binary_X, -binary_Z;
        }

        // 调用 binary_osd
        Error = binary_osd(syndrome, binary_H, probability, k, s);

        // 解码标志为失败
        decoding_success = false;
    }
    else if(OSD != OSDType::NONE){
        throw std::invalid_argument("Invalid OSD method. Expected 'binary'.");
    }

    return std::make_tuple(Error, decoding_success, final_iter);
}


// 辅助方法：计算 GF(2) 矩阵的秩
int LLRBp4Decoder::gf2_rank(Eigen::MatrixXi mat){
    int rank = 0;
    int rows = static_cast<int>(mat.rows());
    int cols = static_cast<int>(mat.cols());

    for(int col = 0; col < cols; ++col){
        // 寻找非零行
        int sel = -1;
        for(int row = rank; row < rows; ++row){
            if(mat(row, col) != 0){
                sel = row;
                break;
            }
        }

        if(sel == -1){
            continue;
        }

        // 交换行
        if(sel != rank){
            mat.row(sel).swap(mat.row(rank));
        }

        // 消去其他行
        for(int row = 0; row < rows; ++row){
            if(row != rank && mat(row, col) != 0){
                mat.row(row) = mod2(mat.row(row) + mat.row(rank));
            }
        }

        rank++;
        if(rank == rows){
            break;
        }
    }

    return rank;
}

// 辅助方法：在 GF(2) 中求解线性方程组
Eigen::VectorXi LLRBp4Decoder::gf2_solve(Eigen::MatrixXi mat, Eigen::VectorXi vec){
    int rows = static_cast<int>(mat.rows());
    int cols = static_cast<int>(mat.cols());
    Eigen::MatrixXi augmented(mat.rows(), mat.cols() + 1);
    augmented << mat, vec;

    int rank = 0;
    for(int col = 0; col < cols; ++col){
        // 寻找非零行
        int sel = -1;
        for(int row = rank; row < rows; ++row){
            if(augmented(row, col) != 0){
                sel = row;
                break;
            }
        }

        if(sel == -1){
            continue;
        }

        // 交换行
        if(sel != rank){
            augmented.row(sel).swap(augmented.row(rank));
        }

        // 消去其他行
        for(int row = 0; row < rows; ++row){
            if(row != rank && augmented(row, col) != 0){
                augmented.row(row) = mod2(augmented.row(row) + augmented.row(rank));
            }
        }

        rank++;
        if(rank == rows){
            break;
        }
    }

    // 检查是否有解
    for(int row = rank; row < rows; ++row){
        if(augmented(row, cols) != 0){
            throw std::runtime_error("No solution exists for the given system.");
        }
    }

    // 解
    Eigen::VectorXi solution = Eigen::VectorXi::Zero(cols);
    for(int row = 0; row < rank; ++row){
        int first_col = -1;
        for(int col = 0; col < cols; ++col){
            if(augmented(row, col) != 0){
                first_col = col;
                break;
            }
        }
        if(first_col != -1){
            solution(first_col) = augmented(row, cols);
        }
    }

    return solution;
}

// 合并后的 OSD 函数实现
std::vector<int> LLRBp4Decoder::binary_osd( const Eigen::VectorXi& syndrome,
                                            const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& binary_H,
                                            const Eigen::VectorXd& probability,
                                            int k,
                                            int s){
    int total_col = static_cast<int>(binary_H.cols());
    int n = static_cast<int>((binary_H.cols() - s) / 2);
    int rank = n - k;

    // 创建可修改的二进制矩阵和综合症向量
    Eigen::MatrixXi H_mod = binary_H.cast<int>(); // 转换为 int 以便于运算
    Eigen::VectorXi syndrome_mod = syndrome;

    // 调整行数以满足 rank
    while(rank < gf2_rank(H_mod)){
        if(H_mod.rows() % 2 == 0){
            // 删除最上面一行（第一行）
            H_mod = H_mod.bottomRows(H_mod.rows() - 1); // 保留除第一行之外的所有行
            syndrome_mod = syndrome_mod.tail(syndrome_mod.size() - 1); // 删除第一元素
        }
        else{
            // 删除最下面一行（最后一行）
            H_mod = H_mod.topRows(H_mod.rows() - 1); // 保留除最后一行之外的所有行
            syndrome_mod = syndrome_mod.head(syndrome_mod.size() - 1); // 删除最后一个元素
        }
    }

    // 获取排序后的列索引
    std::vector<int> sorted_col(total_col);
    for(int i = 0; i < total_col; ++i){
        sorted_col[i] = i;
    }
    std::sort(sorted_col.begin(), sorted_col.end(),
              [&probability](int a, int b) -> bool {
                  return probability(a) < probability(b);
              });

    // 选择列以达到所需的秩
    std::vector<int> index;
    index.push_back(sorted_col[0]);

    for(int i = 1; i < total_col && index.size() < rank + 1; ++i){
        std::vector<int> temp = index;
        temp.push_back(sorted_col[i]);

        Eigen::MatrixXi Hjprime = H_mod(Eigen::placeholders::all, Eigen::Map<const Eigen::VectorXi>(temp.data(), temp.size()));
        Eigen::MatrixXi Hj = H_mod(Eigen::placeholders::all, Eigen::Map<const Eigen::VectorXi>(index.data(), index.size()));

        int current_rank = gf2_rank(Hj);
        int new_rank = gf2_rank(Hjprime);

        if(new_rank > current_rank){
            index = temp;
        }

        if(current_rank == rank){
            break;
        }
    }

    // 初始化错误向量
    std::vector<int> binary_error(total_col, 0);

    try{
        // 提取选择的子矩阵
        Eigen::MatrixXi Hj = H_mod(Eigen::placeholders::all, Eigen::Map<const Eigen::VectorXi>(index.data(), index.size()));
        // 求解线性方程组 Hj * X = syndrome_mod
        Eigen::VectorXi X = gf2_solve(Hj, syndrome_mod.head(Hj.rows()));

        // 设置错误向量
        for(int i = 0; i < index.size(); ++i){
            binary_error[index[i]] = X(i);
        }
    }
    catch(const std::runtime_error& e){
        // 如果没有解，则返回全零错误
        return std::vector<int>(total_col, 0);
    }

    // 转换为最终的 Error 向量
    std::vector<int> Error(total_col, 0);
    for(int i = 0; i < n; ++i){
        if(binary_error[i] == 1){ // X error
            Error[i] += 1;
        }
        if(binary_error[i + n] == 1){ // Z error
            Error[i + n] += 1;
        }
    }
    if(s > 0){
        for(int i = 0; i < s; ++i){
            if(binary_error[2*n + i] == 1){ // S error
                Error[2*n + i] += 1;
            }
        }
    }

    return Error;
}
