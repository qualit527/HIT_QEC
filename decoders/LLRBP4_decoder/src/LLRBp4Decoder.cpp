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
    m = static_cast<int>(H.rows());     // ��ʽת�� Eigen::Index �� int
    n = static_cast<int>(H.cols());     // ��ʽת�� Eigen::Index �� int
    s = Hs_.size() > 0 ? static_cast<int>(Hs_.cols()) : 0; // ��ʽת�� Eigen::Index �� int

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

    // Ԥ���� H �� Hs �ķ�������
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


// ����һ��ͨ�õ�ģ 2 ����
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
    // ���Ƴ�Ա����
    Eigen::MatrixXd d_message_local = (s > 0) ? d_message_ds : d_message; // �����Ƿ���� Hs ѡ��ʹ���ĸ���Ϣ����
    Eigen::MatrixXd delta_message_local = (s > 0) ? delta_message_ds : delta_message;

    // ��ʼ�� Error����СΪ 2n+s
    std::vector<int> Error(2 * n + s, 0); 

    // ��ʼ���ɿ���
    Eigen::VectorXd reliability = Eigen::VectorXd::Ones(n + s); // ÿ�����ص���ʷ�ɿ���

    // ��ʼ�� qX, qY, qZ, qS������ѧ������
    Eigen::VectorXd qX = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd qY = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd qZ = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd qS = Eigen::VectorXd::Zero((s > 0) ? s : 1); // ��� s == 0����������һ��Ԫ��

    // ��0�����о�����
    Eigen::VectorXd qX_0 = Eigen::VectorXd::Constant(n, std::log(pi / px));
    Eigen::VectorXd qY_0 = Eigen::VectorXd::Constant(n, std::log(pi / py));
    Eigen::VectorXd qZ_0 = Eigen::VectorXd::Constant(n, std::log(pi / pz));
    Eigen::VectorXd qS_0 = Eigen::VectorXd::Constant((s > 0) ? s : 1, std::log(pi / ps));

    // ��ʼ��������ر���
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
    Eigen::VectorXd gt_S_q = Eigen::VectorXd::Zero((s > 0) ? s : 1); // ��� s == 0����������һ��Ԫ��
    Eigen::VectorXd Vt_S_q = Eigen::VectorXd::Zero((s > 0) ? s : 1);

    // ������
    Eigen::VectorXd oscillation = Eigen::VectorXd::Zero(n + s);

    // ��ʼ�����Ĵ���
    std::vector<int> last_Error(2 * n + s, 0);

    std::vector<double> tanh_values_buffer; // ���������� tanh ֵ
    tanh_values_buffer.reserve(n + s);

    // ��ʼ����
    for(int iter = 0; iter < max_iter; ++iter){
        // ��ʼ�� Error
        std::fill(Error.begin(), Error.end(), 0); // Error = [Ex, Ez]

        if(schedule == ScheduleType::FLOODING){
            // ˮƽ����
            for(int j = 0; j < m; ++j){
                // ���� tanh_values
                size_t num_indices = H_rows_nonzero_cols[j].size();

                tanh_values_buffer.resize(num_indices);
                for(size_t k = 0; k < num_indices; ++k){
                    tanh_values_buffer[k] = std::tanh(d_message_local(j, H_rows_nonzero_cols[j][k]) / 2.0) + 1e-10;
                }

                // ����˻���
                double product_term = 1.0;
                for(const auto& val : tanh_values_buffer){
                    product_term *= val;
                }

                double syndrome_factor = (syndrome(j) == 1) ? -1.0 : 1.0;

                // ���� delta_message
                for(size_t k = 0; k < num_indices; ++k){
                    double product_excluding_current = product_term / tanh_values_buffer[k];
                    // Clip the value to avoid numerical issues
                    product_excluding_current = std::clamp(product_excluding_current, -1.0 + 1e-10, 1.0 - 1e-10);
                    double delta_value = 2.0 * std::atanh(product_excluding_current);
                    delta_message_local(j, H_rows_nonzero_cols[j][k]) = syndrome_factor * delta_value;
                }
            }

            // ��ֱ����
            for(int j = 0; j < n + s; ++j){
                if(j < n){
                    qX(j) = qX_0(j);
                    qY(j) = qY_0(j);
                    qZ(j) = qZ_0(j);
                }
                else{
                    // ��ʼ�� qS[j - n]
                    qS(j - n) = qS_0(j - n);
                }

                // ���� init ����
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
                            // ���¶����� for qS
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
                    // // ���� last_qS
                    // for(int k = 0; k < s; ++k){
                    //     last_qS(k) = qS(k);
                    // }
                }

                else if(init != InitType::NONE){
                    throw std::invalid_argument("Invalid init method. Expected 'Momentum'.");
                }


                // ���о�����
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
                    // ���� qS �����о�
                    for(auto &row : H_cols_nonzero_rows[j]){
                        if(Hs(row, j - n) == 1){
                            sum_qS_summands += delta_message_local(row, j);
                        }
                    }
                    qS(j - n) += sum_qS_summands;
                }

                // ���� method ����
                if(method == MethodType::MOMENTUM){
                    if(j < n){
                        // ���¶����� for qX, qY, qZ
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
                        // ���¶����� for qS
                        gt_S_q(j - n) = beta * gt_S_q(j - n) + (1.0 - beta) * (last_qS_q(j - n) - qS(j - n));
                        qS(j - n) = last_qS_q(j - n) - alpha * gt_S_q(j - n);
                    }

                    // ���� last_qS
                    for(int k = 0; k < s; ++k){
                        last_qS_q(k) = qS(k);
                    }
                }

                else if(method == MethodType::ADA){
                    if(j < n){
                        // AdaGrad ���� for qX, qY, qZ
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
                        // AdaGrad ���� for qS
                        gt_S_q(j - n) = last_qS_q(j - n) - qS(j - n);
                        Vt_S_q(j - n) += std::pow(gt_S_q(j - n), 2);

                        if(iter != 0){
                            qS(j - n) = last_qS_q(j - n) - (alpha * gt_S_q(j - n)) / (std::sqrt(Vt_S_q(j - n)) + 1e-10);
                        }
                    }

                    // ���� last_qS
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
                        // MBP ���� for qS
                        qS(j - n) = qS(j - n) - sum_qS_summands + (sum_qS_summands / alpha);
                    }
                }
                else if(method != MethodType::NONE){
                    throw std::invalid_argument("Invalid method. Expected 'Momentum', 'Ada', or 'MBP'.");
                }

                // ���� d_message
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
                    // ���� d_message_ds �е� qS ����
                    for(auto &row : H_cols_nonzero_rows[j]){
                        d_message_local(row, j) = qS(j - n) - delta_message_local(row, j);
                    }
                }

                // Ӳ�о�
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
                                // �����쳣���
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

                // // ���¿ɿ��Ի���
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
            last_Error = Error; // ���Ƶ�ǰ����������
        }

        else if(schedule == ScheduleType::LAYER){
            // �����ڵ�㼶����
            for(int j = 0; j < n + s; ++j){
                // ˮƽ����
                for(auto &m_row : H_cols_nonzero_rows[j]){
                    // �ҳ� H(m_row, :) != 0 �� != j
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

                    // ���� tanh_values
                    tanh_values_buffer.resize(index_m.size());
                    for(size_t k = 0; k < index_m.size(); ++k){
                        tanh_values_buffer[k] = std::tanh(d_message_local(m_row, index_m[k]) / 2.0) + 1e-10;
                    }

                    // ����˻���
                    double product_term = 1.0;
                    for(const auto& val : tanh_values_buffer){
                        product_term *= val;
                    }
                    product_term = std::clamp(product_term, -1.0 + 1e-10, 1.0 - 1e-10);

                    double syndrome_factor = (syndrome(m_row) == 1) ? -1.0 : 1.0;

                    // ���� delta_value
                    double delta_value = 2.0 * std::atanh(product_term);
                    delta_message_local(m_row, j) = syndrome_factor * delta_value;
                }

                // ��ֱ����
                if(j < n){
                    // ��ʼ�� qX[j], qY[j], qZ[j]
                    qX(j) = qX_0(j);
                    qY(j) = qY_0(j);
                    qZ(j) = qZ_0(j);
                }
                else{
                    // ��ʼ�� qS[j - n]
                    qS(j - n) = qS_0(j - n);
                }

                // ���� init ����
                if(init == InitType::MOMENTUM){
                    // ���¶�����
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
                            // ���¶�����
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
                            // ���¶����� for qS
                            gt_S(j - n) = beta * gt_S(j - n) + (1.0 - beta) * (last_qS(j - n) - qS(j - n));
                            qS(j - n) = last_qS(j - n) - alpha * gt_S(j - n);
                        }
                    }

                    // ���� last_qS
                    for(int k = 0; k < s; ++k){
                        last_qS(k) = qS(k);
                    }
                }
                else if(init != InitType::NONE){
                    throw std::invalid_argument("Invalid init method. Expected 'Momentum'.");
                }

                // ���о�����
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
                    // ���� qS �����о�
                    for(auto &row : H_cols_nonzero_rows[j]){
                        if(Hs(row, j - n) == 1){
                            sum_qS_summands += delta_message_local(row, j);
                        }
                    }
                    qS(j - n) += sum_qS_summands;
                }

                // ���� method ����
                if(method == MethodType::MOMENTUM){
                    if(j < n){
                        // ���¶����� for qX, qY, qZ
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
                        // ���¶����� for qS
                        gt_S_q(j - n) = beta * gt_S_q(j - n) + (1.0 - beta) * (last_qS_q(j - n) - qS(j - n));
                        qS(j - n) = last_qS_q(j - n) - alpha * gt_S_q(j - n);
                    }

                    // ���� last_qS
                    for(int k = 0; k < s; ++k){
                        last_qS_q(k) = qS(k);
                    }
                }

                else if(method == MethodType::ADA){
                    if(j < n){
                        // AdaGrad ���� for qX, qY, qZ
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
                        // AdaGrad ���� for qS
                        gt_S_q(j - n) = last_qS_q(j - n) - qS(j - n);
                        Vt_S_q(j - n) += std::pow(gt_S_q(j - n), 2);

                        if(iter != 0){
                            qS(j - n) = last_qS_q(j - n) - (alpha * gt_S_q(j - n)) / (std::sqrt(Vt_S_q(j - n)) + 1e-10);
                        }
                    }

                    // ���� last_qS
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
                        // MBP ���� for qS
                        qS(j - n) = qS(j - n) - sum_qS_summands + (sum_qS_summands / alpha);
                    }
                }

                else if(method != MethodType::NONE){
                    throw std::invalid_argument("Invalid method. Expected 'Momentum', 'Ada', or 'MBP'.");
                }

                // ���� d_message
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
                    // ���� d_message_ds �е� qS ����
                    for(auto &row : H_cols_nonzero_rows[j]){
                        d_message_local(row, j) = qS(j - n) - delta_message_local(row, j);
                    }
                }
            }

            // Ӳ�о�
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
                            // �����쳣���
                            Error[j] = 0;
                            Error[j + n] = 0;
                            break;
                    }
                }

                // // ���¿ɿ��Ի���
                // if((last_Error[j] == Error[j]) && (last_Error[j + n] == Error[j + n])){
                //     reliability(j) += 1.0;
                // }
                // else{
                //     oscillation(j) += 1.0;
                //     reliability(j) = 1.0;
                // }
            }

            // ���� s ���ȶ��ӵ�Ӳ�о���������������ѧ���룩
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

            last_Error = Error; // ���Ƶ�ǰ����������
        }

        else{
            throw std::invalid_argument("Invalid schedule. Expected 'flooding' or 'layer'.");
        }

        // ����Ƿ������������
        // (Hz * Error_X + Hx * Error_Z) % 2 == syndrome
        // ���� Error_X = Error[0:n], Error_Z = Error[n:2n]
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

        // �����ۺ�֢
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

    // ����ﵽ�����������󣬽��� OSD ����
    bool decoding_success = false;
    int final_iter = max_iter;

    if(OSD == OSDType::BINARY){
        // ���� px, py, pz
        Eigen::VectorXd px_prob = (1.0 / (qX.array().exp() + 1.0)).matrix();
        Eigen::VectorXd py_prob = (1.0 / (qY.array().exp() + 1.0)).matrix();
        Eigen::VectorXd pz_prob = (1.0 / (qZ.array().exp() + 1.0)).matrix();
        Eigen::VectorXd ps_prob = (s > 0) ? (1.0 / (qS.array().exp() + 1.0)).matrix() : Eigen::VectorXd();

        // ���� binary_X �� binary_Z
        Eigen::VectorXd binary_X = px_prob + py_prob;
        Eigen::VectorXd binary_Z = pz_prob + py_prob;

        // ��ϸ���
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

        // ���� binary_osd
        Error = binary_osd(syndrome, binary_H, probability, k, s);

        // �����־Ϊʧ��
        decoding_success = false;
    }
    else if(OSD != OSDType::NONE){
        throw std::invalid_argument("Invalid OSD method. Expected 'binary'.");
    }

    return std::make_tuple(Error, decoding_success, final_iter);
}


// �������������� GF(2) �������
int LLRBp4Decoder::gf2_rank(Eigen::MatrixXi mat){
    int rank = 0;
    int rows = static_cast<int>(mat.rows());
    int cols = static_cast<int>(mat.cols());

    for(int col = 0; col < cols; ++col){
        // Ѱ�ҷ�����
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

        // ������
        if(sel != rank){
            mat.row(sel).swap(mat.row(rank));
        }

        // ��ȥ������
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

// ������������ GF(2) ��������Է�����
Eigen::VectorXi LLRBp4Decoder::gf2_solve(Eigen::MatrixXi mat, Eigen::VectorXi vec){
    int rows = static_cast<int>(mat.rows());
    int cols = static_cast<int>(mat.cols());
    Eigen::MatrixXi augmented(mat.rows(), mat.cols() + 1);
    augmented << mat, vec;

    int rank = 0;
    for(int col = 0; col < cols; ++col){
        // Ѱ�ҷ�����
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

        // ������
        if(sel != rank){
            augmented.row(sel).swap(augmented.row(rank));
        }

        // ��ȥ������
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

    // ����Ƿ��н�
    for(int row = rank; row < rows; ++row){
        if(augmented(row, cols) != 0){
            throw std::runtime_error("No solution exists for the given system.");
        }
    }

    // ��
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

// �ϲ���� OSD ����ʵ��
std::vector<int> LLRBp4Decoder::binary_osd( const Eigen::VectorXi& syndrome,
                                            const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& binary_H,
                                            const Eigen::VectorXd& probability,
                                            int k,
                                            int s){
    int total_col = static_cast<int>(binary_H.cols());
    int n = static_cast<int>((binary_H.cols() - s) / 2);
    int rank = n - k;

    // �������޸ĵĶ����ƾ�����ۺ�֢����
    Eigen::MatrixXi H_mod = binary_H.cast<int>(); // ת��Ϊ int �Ա�������
    Eigen::VectorXi syndrome_mod = syndrome;

    // �������������� rank
    while(rank < gf2_rank(H_mod)){
        if(H_mod.rows() % 2 == 0){
            // ɾ��������һ�У���һ�У�
            H_mod = H_mod.bottomRows(H_mod.rows() - 1); // ��������һ��֮���������
            syndrome_mod = syndrome_mod.tail(syndrome_mod.size() - 1); // ɾ����һԪ��
        }
        else{
            // ɾ��������һ�У����һ�У�
            H_mod = H_mod.topRows(H_mod.rows() - 1); // ���������һ��֮���������
            syndrome_mod = syndrome_mod.head(syndrome_mod.size() - 1); // ɾ�����һ��Ԫ��
        }
    }

    // ��ȡ������������
    std::vector<int> sorted_col(total_col);
    for(int i = 0; i < total_col; ++i){
        sorted_col[i] = i;
    }
    std::sort(sorted_col.begin(), sorted_col.end(),
              [&probability](int a, int b) -> bool {
                  return probability(a) < probability(b);
              });

    // ѡ�����Դﵽ�������
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

    // ��ʼ����������
    std::vector<int> binary_error(total_col, 0);

    try{
        // ��ȡѡ����Ӿ���
        Eigen::MatrixXi Hj = H_mod(Eigen::placeholders::all, Eigen::Map<const Eigen::VectorXi>(index.data(), index.size()));
        // ������Է����� Hj * X = syndrome_mod
        Eigen::VectorXi X = gf2_solve(Hj, syndrome_mod.head(Hj.rows()));

        // ���ô�������
        for(int i = 0; i < index.size(); ++i){
            binary_error[index[i]] = X(i);
        }
    }
    catch(const std::runtime_error& e){
        // ���û�н⣬�򷵻�ȫ�����
        return std::vector<int>(total_col, 0);
    }

    // ת��Ϊ���յ� Error ����
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
