#pragma once

#ifndef _BP_DECODER_HPP_
#define _BP_DECODER_HPP_

#include "sparse_matrix.hpp"

namespace bp_decoder
{
    std::vector<uint8_t> getSyndrome(sparse_matrix::Mod2SparseMatrix const &h_bar, const std::vector<uint8_t> &code_word);
    class BpDecoder
    {
    public: // types
        enum class Method
        {
            MIN_SUM,
            PRODUCT_SUM,
            PRODUCT_SUM_NOT_DECOUPLED,
            Scaling_MIN_SUM,
            Scaling_PRODUCT_SUM
        };

    private: // vars
        Method method;
        size_t max_iter;
        ::sparse_matrix::Mod2SparseMatrix h_bar;
        ::std::vector<double> error_rate;

    private: // utils
        auto init();
        void update(::std::vector<uint8_t> &decoding, ::std::vector<double> &log_prob_ratios, ::std::vector<uint8_t> const &syndrome, ::std::vector<double> const &prob_ratio_initial, size_t iter, ::std::vector<::std::map<::sparse_matrix::Mod2SparseMatrix::Item *, ::sparse_matrix::Mod2SparseMatrix::Item *>> const &same_position);
        size_t hammingWeight(::std::vector<uint8_t> const &src1, ::std::vector<uint8_t> const &src2);
        ::std::vector<uint8_t> osd0PostProceed(::std::vector<uint8_t> const &syndrome, ::std::vector<double> const &probability_distribution);

    public: // apis
        BpDecoder() = default;
        BpDecoder(Method method, int max_iter, ::sparse_matrix::Mod2SparseMatrix const &h_bar, ::std::vector<double> const &error_rate);
        void setParams(Method method, int max_iter, ::sparse_matrix::Mod2SparseMatrix const &h_bar, ::std::vector<double> const &error_rate);
        void setParams(Method method, int max_iter, ::sparse_matrix::Mod2SparseMatrix &&h_bar, ::std::vector<double> &&error_rate);
        ::std::tuple<size_t, bool, ::std::vector<double>, ::std::vector<uint8_t>> bpDecode(::std::vector<uint8_t> const &syndrome);
        ::std::tuple<::std::vector<uint8_t>, size_t, bool> bpOsdDecode(::std::vector<uint8_t> const &syndrome);
    };
}

#endif