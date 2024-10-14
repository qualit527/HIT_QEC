#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <sstream>
#include "bp_decoder.hpp"

namespace py = pybind11;
using namespace py::literals;

sparse_matrix::Mod2SparseMatrix Mod2SparseMatrixConstructFromNumpy(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> const &matrix)
{
    if (matrix.ndim() != 2)
        throw std::invalid_argument("Input should be 2D numpy array.");
    if (matrix.shape()[1] % 3 != 0)
        throw std::invalid_argument("Input's cols should be 3×n.");

    sparse_matrix::Mod2SparseMatrix result(matrix.shape()[0], matrix.shape()[1]);

    for (size_t i{0ULL}; i < result.row; i++)
    {
        for (size_t j{0ULL}; j < result.col; j++)
        {
            if (matrix.at(i, j) == 0)
            {
                continue;
            }
            else if (matrix.at(i, j) == 1)
            {
                auto item_ptr{::std::make_shared<sparse_matrix::Mod2SparseMatrix::Item>(i, j)};
                result.items_each_row[i].push_back(item_ptr);
                result.items_each_col[j].push_back(item_ptr);
            }
            else
            {
                throw std::invalid_argument("Input's elements should only consist of 0 or 1.");
            }
        }
    }

    return result;
}

py::tuple bpDecodeWrapper(bp_decoder::BpDecoder &self, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> const &array)
{
    if (array.ndim() != 1)
        throw std::invalid_argument("Input should be 2D numpy array.");

    std::vector<uint8_t> syndrome(array.size());
    std::memcpy(syndrome.data(), array.data(), array.size() * sizeof(uint8_t));
    for (auto e : syndrome)
        if (e != 0 && e != 1)
            throw std::invalid_argument("Syndrome should only consist of 0 or 1.");

    auto [run_iter, converge, log_prob_ratios, bp_decoding] = self.bpDecode(syndrome);

    auto result = py::array_t<uint8_t>(bp_decoding.size());
    auto result_buffer = result.request();
    uint8_t *result_ptr = (uint8_t *)result_buffer.ptr;
    std::memcpy(result_ptr, bp_decoding.data(), bp_decoding.size() * sizeof(uint8_t));

    return py::make_tuple(result, run_iter, converge);
}

py::tuple bpOsdDecodeWrapper(bp_decoder::BpDecoder &self, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> const &array)
{
    if (array.ndim() != 1)
        throw std::invalid_argument("Input should be 2D numpy array.");

    std::vector<uint8_t> syndrome(array.size());
    std::memcpy(syndrome.data(), array.data(), array.size() * sizeof(uint8_t));
    for (auto e : syndrome)
        if (e != 0 && e != 1)
            throw std::invalid_argument("Syndrome should only consist of 0 or 1.");

    auto &&[bp_osd_decoding, run_iter, converge] = self.bpOsdDecode(syndrome);

    auto result = py::array_t<uint8_t>(bp_osd_decoding.size());
    auto result_buffer = result.request();
    uint8_t *result_ptr = (uint8_t *)result_buffer.ptr;
    std::memcpy(result_ptr, bp_osd_decoding.data(), bp_osd_decoding.size() * sizeof(uint8_t));

    return py::make_tuple(result, run_iter, converge);
}

PYBIND11_MODULE(bpdecoupling, m)
{
    m.doc() = "BP decoupling decoder module.";

    py::class_<sparse_matrix::Mod2SparseMatrix>(m, "Mod2SparseMatrix")
        .def(py::init(&Mod2SparseMatrixConstructFromNumpy), "matrix"_a)
        .def("__repr__", &sparse_matrix::Mod2SparseMatrix::toString);

    py::implicitly_convertible<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>, sparse_matrix::Mod2SparseMatrix>();

    py::class_<bp_decoder::BpDecoder> decoder(m, "Decoder");

    py::enum_<bp_decoder::BpDecoder::Method>(decoder, "Method")
        .value("MIN_SUM", bp_decoder::BpDecoder::Method::MIN_SUM)
        .value("PRODUCT_SUM", bp_decoder::BpDecoder::Method::PRODUCT_SUM)
        .value("PRODUCT_SUM_NOT_DECOUPLED", bp_decoder::BpDecoder::Method::PRODUCT_SUM_NOT_DECOUPLED)
        .value("Scaling_MIN_SUM", bp_decoder::BpDecoder::Method::Scaling_MIN_SUM)
        .value("Scaling_PRODUCT_SUM", bp_decoder::BpDecoder::Method::Scaling_PRODUCT_SUM);

    decoder
        .def(py::init<bp_decoder::BpDecoder::Method, int, sparse_matrix::Mod2SparseMatrix const &, std::vector<double> const &>(), "bp_method"_a, "max_iter"_a, "h_bar"_a, "channel_error_rate"_a)
        .def("bpDecode", &bpDecodeWrapper, "syndrome"_a, R"(return `decoding_result` and `is_converge`.)")
        .def("bpOsdDecode", &bpOsdDecodeWrapper, "syndrome"_a, R"(return `decoding_result` and `is_converge`.)");
}
