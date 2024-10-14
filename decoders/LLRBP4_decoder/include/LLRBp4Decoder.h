// include/LLRBp4Decoder.h

#ifndef LLRBP4DECODER_H
#define LLRBP4DECODER_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <stdexcept>
#include <cmath>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>


enum class PauliType { X, Y, Z };
enum class ScheduleType { FLOODING, LAYER, NONE };
enum class InitType { MOMENTUM, NONE };
enum class MethodType { MOMENTUM, ADA, MBP, NONE };
enum class OSDType { BINARY, NONE };

class LLRBp4Decoder {
public:
    // 构造函数
    LLRBp4Decoder(const Eigen::MatrixXi& Hx,
                 const Eigen::MatrixXi& Hz,
                 double px,
                 double py,
                 double pz,
                 int max_iter,
                 const Eigen::MatrixXi& Hs = Eigen::MatrixXi(),
                 double ps = 1,
                 int dimension = 1);

    // 解码函数
    std::tuple<std::vector<int>, bool, int> standard_decoder(const Eigen::VectorXi& syndrome,
                                                             ScheduleType schedule = ScheduleType::FLOODING,
                                                             InitType init = InitType::NONE,
                                                             MethodType method = MethodType::NONE,
                                                             OSDType OSD = OSDType::NONE,
                                                             double alpha = 1.0,
                                                             double beta = 0.0);


private:
    // 成员变量
    Eigen::MatrixXi Hx;
    Eigen::MatrixXi Hz;
    Eigen::MatrixXi Hs;
    Eigen::MatrixXi H;
    int m;
    int n;
    int s;
    double px, py, pz, ps;
    double pi;
    int k;

    int max_iter;
    bool flag;

    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> binary_H;

    Eigen::MatrixXd Q_matrix_X;
    Eigen::MatrixXd Q_matrix_Y;
    Eigen::MatrixXd Q_matrix_Z;
    Eigen::MatrixXd Q_matrix_S;

    Eigen::MatrixXd d_message;
    Eigen::MatrixXd d_message_ds;
    Eigen::MatrixXd delta_message;
    Eigen::MatrixXd delta_message_ds;

    std::vector<std::vector<int>> H_rows_nonzero_cols;
    std::vector<std::vector<int>> H_cols_nonzero_rows;

    // 私有方法
    inline double lambda_func(const PauliType W, double px_val, double py_val, double pz_val);

    // 辅助方法：计算 GF(2) 矩阵的秩
    int gf2_rank(Eigen::MatrixXi mat);

    // 辅助方法：在 GF(2) 中求解线性方程组
    Eigen::VectorXi gf2_solve(Eigen::MatrixXi mat, Eigen::VectorXi vec);

    std::vector<int> binary_osd(const Eigen::VectorXi& syndrome,
                                const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>& binary_H,
                                const Eigen::VectorXd& probability,
                                int k,
                                int s = 0); // 合并后的 binary_osd 函数
};

#endif // LLRBP4DECODER_H
