#include "bp_decoder.hpp"

#include <iostream>
#include <cmath>
#include <numeric>
#include <map>
#include <ranges>

namespace bp_decoder
{
    std::vector<uint8_t> getSyndrome(sparse_matrix::Mod2SparseMatrix const &h_bar, const std::vector<uint8_t> &code_word)
    {
        return h_bar * code_word;
    }

    auto BpDecoder::init()
    {
        std::vector<double> prob_ratio_initial(error_rate.size());
        for (auto i{0ULL}; i < error_rate.size(); i++)
        {
            auto &item{prob_ratio_initial[i]};
            // if (method == Method::MIN_SUM)
            //     item = log((1 - error_rate[i]) / error_rate[i]);
            // else
            //     item = error_rate[i] / (1 - error_rate[i]);
            item = log((1 - error_rate[i]) / error_rate[i]);
            if (std::isnan(item))
                item = std::numeric_limits<double>::infinity();
        }
        std::vector<std::map<sparse_matrix::Mod2SparseMatrix::Item *, sparse_matrix::Mod2SparseMatrix::Item *>>
            same_position;
        for (auto &row : h_bar.items_each_row)
        {
            std::map<sparse_matrix::Mod2SparseMatrix::Item *, sparse_matrix::Mod2SparseMatrix::Item *>
                same_position_this_row;
            std::map<size_t, sparse_matrix::Mod2SparseMatrix::Item *> idx_item;
            for (auto &item : row)
            {
                item->value.data2parity = prob_ratio_initial[item->col_index];
                item->value.parity2data = 1;

                auto uni_idx{item->col_index % (h_bar.col / 3)};
                auto found{idx_item.find(uni_idx)};
                if (found != idx_item.end())
                {
                    same_position_this_row[item.get()] = found->second;
                    same_position_this_row[found->second] = item.get();
                    idx_item.erase(found);
                }
                else
                {
                    idx_item[uni_idx] = item.get();
                }
            }
            same_position.emplace_back(std::move(same_position_this_row));
        }
        std::vector<double> log_prob_ratios(h_bar.col, 0.);
        std::vector<uint8_t> decoding(h_bar.col, 0);
        return make_tuple(prob_ratio_initial, log_prob_ratios, same_position, decoding);
    }

    void BpDecoder::update(std::vector<uint8_t> &decoding,
                           std::vector<double> &log_prob_ratios,
                           const std::vector<uint8_t> &syndrome,
                           const std::vector<double> &prob_ratio_initial,
                           size_t iter,
                           const std::vector<std::map<
                               sparse_matrix::Mod2SparseMatrix::Item *, sparse_matrix::Mod2SparseMatrix::Item *>> &
                               same_position)
    {
        // Recompute likelihood ratios.
        
        if (method == Method::MIN_SUM)
        {
            double alpha{ 1.0 };
            for (auto i{0ULL}; i < h_bar.row; i++)
            {
                const auto &row{h_bar.items_each_row[i]};
                // calc (sgn * sum) part
                if (syndrome[i] == 0)
                {
                    unsigned row_weight{0U};
                    double min_vals[2]{
                        1e308,
                        1e308,
                    };
                    for (const auto &item : row)
                    {
                        auto dp{item->value.data2parity};
                        // update weight
                        row_weight += (dp < 0);
                        // update min abs
                        auto abs_dp{fabs(dp)};
                        if (abs_dp >= min_vals[1])
                            continue;
                        if (abs_dp >= min_vals[0])
                            min_vals[1] = abs_dp;
                        else // abs_dp < min_val[0]
                            min_vals[1] = min_vals[0], min_vals[0] = abs_dp;
                    }
                    for (const auto &item : row)
                    {
                        // calculate min_sum part
                        auto dp{item->value.data2parity};
                        unsigned weight{row_weight - (dp < 0)};
                        auto sgn{((weight % 2) == 0 ? 1 : -1)};
                        double min_val = (fabs(dp) == min_vals[0]) ? min_vals[1] : min_vals[0];
                        item->value.parity2data = alpha * sgn * min_val;
                    }
                }
                else // syndrome[i] == 1
                {
                    unsigned row_weight{1U};
                    double min_vals[3]{
                        1e308,
                        1e308,
                        1e308,
                    };
                    for (const auto &item : row)
                    {
                        auto dp{item->value.data2parity};
                        // update weight
                        row_weight += (dp < 0);
                        // update min abs
                        auto abs_dp{fabs(dp)};
                        if (abs_dp >= min_vals[2])
                            continue;
                        if (abs_dp >= min_vals[1])
                            min_vals[2] = abs_dp;
                        else if (abs_dp >= min_vals[0])
                            min_vals[2] = min_vals[1], min_vals[1] = abs_dp;
                        else // abs_dp < min_val[0]
                            min_vals[2] = min_vals[1], min_vals[1] = min_vals[0], min_vals[0] = abs_dp;
                    }
                    for (const auto &item : row)
                    {
                        // calculate min_sum part
                        auto dp{item->value.data2parity};
                        auto same_position_dp{same_position[i].at(item.get())->value.data2parity};
                        unsigned weight{row_weight - (dp < 0) - (same_position_dp < 0)};
                        auto sgn{((weight % 2) == 0 ? 1 : -1)};
                        auto abs_dp{fabs(dp)}, abs_dp_same_position{fabs(same_position_dp)};
                        auto [abs_dp_0, abs_dp_1] = std::minmax(abs_dp, abs_dp_same_position);
                        double min_val;
                        if (abs_dp_0 == min_vals[0] && abs_dp_1 == min_vals[1])
                            min_val = min_vals[2];
                        else if (abs_dp_0 == min_vals[0])
                            min_val = min_vals[1];
                        else
                            min_val = min_vals[0];
                        item->value.parity2data = alpha * sgn * min_val;
                    }
                }
                // calc (tanh) part
                double product_all{1.0};
                for (const auto &item : row)
                    product_all *= tanh(item->value.data2parity / 2);
                for (const auto &item : row)
                {
                    auto dp{item->value.data2parity};
                    auto same_position_dp{same_position[i].at(item.get())->value.data2parity};
                    double product_0 = product_all / tanh(dp / 2);
                    if (fabs(dp) < 1e-20)
                    {
                        product_0 = 1;
                        for (const auto &other_item : row)
                            if (other_item->col_index != item->col_index)
                                product_0 *= tanh(other_item->value.data2parity / 2);
                    }
                    double product_1 = product_0 / tanh(same_position_dp / 2);
                    if (fabs(same_position_dp) < 1e-20)
                    {
                        product_1 = 1;
                        for (const auto &other_item : row)
                            if (other_item->col_index != item->col_index && other_item->col_index != same_position[i].at(item.get())->col_index)
                                product_1 *= tanh(other_item->value.data2parity / 2);
                    }
                    if (fabs(1 - product_1) > 1e-20 && fabs(1 - product_0) > 1e-20)
                        item->value.parity2data += log((1 - product_0) / (1 - product_1));
                        //item->value.parity2data += 0.0;
                }
            }
            // Recompute log-probability-ratios for the bits
            for (auto j{ 0ULL }; j < h_bar.col; j++)
            {
                const auto& col{ h_bar.items_each_col[j] };
                auto pr{ prob_ratio_initial[j] };
                for (const auto& item : col)
                {
                    auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
                    pr += item->value.parity2data;
                    if (std::isnan(pr))
                        pr = 0;
                    pr = pr - log(1 - error_rate[item_same_position]);

                }
                log_prob_ratios[j] = pr;
                for (const auto& item : col)
                {
                    auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
                    //item->value.data2parity = pr - item->value.parity2data + log(1 - error_rate[item_same_position]);
                    item->value.data2parity = pr - item->value.parity2data;
                    if (std::isnan(item->value.data2parity))
                        item->value.data2parity = 0;
                    item->value.data2parity += log(1 - error_rate[item_same_position]);
                }
            }
            // Hard decision
            std::fill(decoding.begin(), decoding.end(), 0);
            auto len{ h_bar.col / 3 };
            for (auto k{ 0ULL }; k < len; k++)
            {
                std::vector list_value{ {log_prob_ratios[k], log_prob_ratios[k + len], log_prob_ratios[k + 2 * len]} };
                auto min_elem{ std::min_element(list_value.begin(), list_value.end()) };
                if (*min_elem < 0)
                {
                    auto idx{ distance(list_value.begin(), min_elem) };
                    decoding[k + idx * len] = 1;
                }
            }

        }
        else if ((method == Method::Scaling_MIN_SUM))
        {   
            //auto ite{iter + 1};
            //double alpha{2.0 - pow(2,-ite)};
            double alpha{ 1.0 };
            for (auto i{ 0ULL }; i < h_bar.row; i++)
            {
                const auto& row{ h_bar.items_each_row[i] };
                // calc (sgn * sum) part
                if (syndrome[i] == 0)
                {
                    unsigned row_weight{ 0U };
                    double min_vals[2]{
                        1e308,
                        1e308,
                    };
                    for (const auto& item : row)
                    {
                        auto dp{ item->value.data2parity };
                        // update weight
                        row_weight += (dp < 0);
                        // update min abs
                        auto abs_dp{ fabs(dp) };
                        if (abs_dp >= min_vals[1])
                            continue;
                        if (abs_dp >= min_vals[0])
                            min_vals[1] = abs_dp;
                        else // abs_dp < min_val[0]
                            min_vals[1] = min_vals[0], min_vals[0] = abs_dp;
                    }
                    for (const auto& item : row)
                    {
                        // calculate min_sum part
                        auto dp{ item->value.data2parity };
                        unsigned weight{ row_weight - (dp < 0) };
                        auto sgn{ ((weight % 2) == 0 ? 1 : -1) };
                        double min_val = (fabs(dp) == min_vals[0]) ? min_vals[1] : min_vals[0];
                        //item->value.parity2data = alpha * sgn * min_val;
                        item->value.parity2data = sgn * min_val;
                    }
                }
                else // syndrome[i] == 1
                {
                    unsigned row_weight{ 1U };
                    double min_vals[3]{
                        1e308,
                        1e308,
                        1e308,
                    };
                    for (const auto& item : row)
                    {
                        auto dp{ item->value.data2parity };
                        // update weight
                        row_weight += (dp < 0);
                        // update min abs
                        auto abs_dp{ fabs(dp) };
                        if (abs_dp >= min_vals[2])
                            continue;
                        if (abs_dp >= min_vals[1])
                            min_vals[2] = abs_dp;
                        else if (abs_dp >= min_vals[0])
                            min_vals[2] = min_vals[1], min_vals[1] = abs_dp;
                        else // abs_dp < min_val[0]
                            min_vals[2] = min_vals[1], min_vals[1] = min_vals[0], min_vals[0] = abs_dp;
                    }
                    for (const auto& item : row)
                    {
                        // calculate min_sum part
                        auto dp{ item->value.data2parity };
                        auto same_position_dp{ same_position[i].at(item.get())->value.data2parity };
                        unsigned weight{ row_weight - (dp < 0) - (same_position_dp < 0) };
                        auto sgn{ ((weight % 2) == 0 ? 1 : -1) };
                        auto abs_dp{ fabs(dp) }, abs_dp_same_position{ fabs(same_position_dp) };
                        auto [abs_dp_0, abs_dp_1] = std::minmax(abs_dp, abs_dp_same_position);
                        double min_val;
                        if (abs_dp_0 == min_vals[0] && abs_dp_1 == min_vals[1])
                            min_val = min_vals[2];
                        else if (abs_dp_0 == min_vals[0])
                            min_val = min_vals[1];
                        else
                            min_val = min_vals[0];
                        //item->value.parity2data = alpha * sgn * min_val;
                        item->value.parity2data = sgn * min_val;
                    }
                }
                // calc (tanh) part
                double product_all{ 1.0 };
                for (const auto& item : row)
                    product_all *= tanh(item->value.data2parity / 2);
                for (const auto& item : row)
                {
                    auto dp{ item->value.data2parity };
                    auto same_position_dp{ same_position[i].at(item.get())->value.data2parity };
                    double product_0 = product_all / tanh(dp / 2);
                    if (fabs(dp) < 1e-20)
                    {
                        product_0 = 1;
                        for (const auto& other_item : row)
                            if (other_item->col_index != item->col_index)
                                product_0 *= tanh(other_item->value.data2parity / 2);
                    }
                    double product_1 = product_0 / tanh(same_position_dp / 2);
                    if (fabs(same_position_dp) < 1e-20)
                    {
                        product_1 = 1;
                        for (const auto& other_item : row)
                            if (other_item->col_index != item->col_index && other_item->col_index != same_position[i].at(item.get())->col_index)
                                product_1 *= tanh(other_item->value.data2parity / 2);
                    }
                    if (fabs(1 - product_1) > 1e-20 && fabs(1 - product_0) > 1e-20)
                        //item->value.parity2data = item->value.parity2data + log((1 - product_0) / (1 - product_1));
                        item->value.parity2data = alpha * (item->value.parity2data) + log((1 - product_0) / (1 - product_1));
                }
            }
            // Recompute log-probability-ratios for the bits
            for (auto j{0ULL}; j < h_bar.col; j++)
            {
                const auto& col{ h_bar.items_each_col[j] };
                auto pr{ prob_ratio_initial[j] };
                for (const auto& item : col)
                {
                    auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
                    pr += item->value.parity2data;
                    if (std::isnan(pr))
                        pr = 0;
                    pr = pr - log(1 - error_rate[item_same_position]);

                }
                log_prob_ratios[j] = pr;
                for (const auto& item : col)
                {
                    auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
                    //item->value.data2parity = pr - item->value.parity2data + log(1 - error_rate[item_same_position]);
                    auto alpha_temp{ pr - item->value.parity2data };
                    if (std::isnan(alpha_temp))
                    {
                        alpha_temp = 0;
                    }  
                    alpha_temp += log(1 - error_rate[item_same_position]);
                    if (((alpha_temp > 0.0 && item->value.data2parity < 0.0) || (alpha_temp < 0.0 && item->value.data2parity > 0.0)) && item->value.data2parity != 0.0 )
                    {
                        item->value.data2parity = 0;
                    }
                    else
                    {
                        item->value.data2parity = alpha_temp;
                    }
                    //item->value.data2parity += log(1 - error_rate[item_same_position]);
                    /*item->value.data2parity = pr - item->value.parity2data;
                    if (std::isnan(item->value.data2parity))
                        item->value.data2parity = 0;
                    item->value.data2parity += log(1 - error_rate[item_same_position]);*/
                }
            }
            // Hard decision
            std::fill(decoding.begin(), decoding.end(), 0);
            auto len{ h_bar.col / 3 };
            for (auto k{ 0ULL }; k < len; k++)
            {
                std::vector list_value{ {log_prob_ratios[k], log_prob_ratios[k + len], log_prob_ratios[k + 2 * len]} };
                auto min_elem{ std::min_element(list_value.begin(), list_value.end()) };
                if (*min_elem < 0)
                {
                    auto idx{ distance(list_value.begin(), min_elem) };
                    decoding[k + idx * len] = 1;
                    /*if (idx == 0)
                    {
                        log_prob_ratios[k + len] = fabs(log_prob_ratios[k + len]);
                        log_prob_ratios[k + 2 * len] = fabs(log_prob_ratios[k + 2 * len]);
                    }
                    else if (idx == 1)
                    {
                        log_prob_ratios[k] = fabs(log_prob_ratios[k]);
                        log_prob_ratios[k + 2 * len] = fabs(log_prob_ratios[k + 2 * len]);
                    }
                    else if (idx == 2)
                    {
                        log_prob_ratios[k] = fabs(log_prob_ratios[k]);
                        log_prob_ratios[k + len] = fabs(log_prob_ratios[k + len]);
                    }*/
                }
            }
            //for (auto j{ 0ULL }; j < h_bar.col; j++)
            //{
            //    const auto& col{ h_bar.items_each_col[j] };
            //    auto pr{ log_prob_ratios[j] };
            //    for (const auto& item : col)
            //    {
            //        auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
            //        //item->value.data2parity = pr - item->value.parity2data + log(1 - error_rate[item_same_position]);
            //        auto alpha_temp{ pr - item->value.parity2data };
            //        if (std::isnan(alpha_temp))
            //        {
            //            alpha_temp = 0;
            //        }  
            //        alpha_temp += log(1 - error_rate[item_same_position]);
            //        if (((alpha_temp > 0.0 && item->value.data2parity < 0.0) || (alpha_temp < 0.0 && item->value.data2parity > 0.0)) && item->value.data2parity != 0.0 )
            //        {
            //            item->value.data2parity = 0;
            //        }
            //        else
            //        {
            //            item->value.data2parity = alpha_temp;
            //        }
            //        //item->value.data2parity += log(1 - error_rate[item_same_position]);
            //        /*item->value.data2parity = pr - item->value.parity2data;
            //        if (std::isnan(item->value.data2parity))
            //            item->value.data2parity = 0;
            //        item->value.data2parity += log(1 - error_rate[item_same_position]);*/
            //    }
            //}
        }
        else if ((method == Method::PRODUCT_SUM))// method == Method::PRODUCT_SUM
        {
            double alpha{ 1.0 };
            for (auto i{0ULL}; i < h_bar.row; i++)
            {
                const auto &row{h_bar.items_each_row[i]};
                double product_all{1.0};
                for (const auto &item : row)
                    product_all *= tanh(item->value.data2parity / 2);
                for (const auto &item : row)
                {
                    auto dp{item->value.data2parity};
                    auto same_position_dp{same_position[i].at(item.get())->value.data2parity};
                    double product_0 = product_all / tanh(dp / 2);
                    if (fabs(dp) < 1e-20)
                    {
                        product_0 = 1;
                        for (const auto &other_item : row)
                            if (other_item->col_index != item->col_index)
                                product_0 *= tanh(other_item->value.data2parity / 2);
                    }
                    double product_1 = product_0 / tanh(same_position_dp / 2);
                    if (fabs(same_position_dp) < 1e-20)
                    {
                        product_1 = 1;
                        for (const auto &other_item : row)
                            if (other_item->col_index != item->col_index && other_item->col_index != same_position[i].at(item.get())->col_index)
                                product_1 *= tanh(other_item->value.data2parity / 2);
                    }
                    if (syndrome[i] == 0)
                        item->value.parity2data = alpha * log((1 + product_0) / (1 - product_1));
                    else
                        item->value.parity2data = alpha * log((1 - product_0) / (1 + product_1));
                }
            }
            // Recompute log-probability-ratios for the bits
            for (auto j{ 0ULL }; j < h_bar.col; j++)
            {
                const auto& col{ h_bar.items_each_col[j] };
                auto pr{ prob_ratio_initial[j] };
                for (const auto& item : col)
                {
                    auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
                    pr += item->value.parity2data;
                    if (std::isnan(pr))
                        pr = 0;
                    pr = pr - log(1 - error_rate[item_same_position]);

                }
                log_prob_ratios[j] = pr;
                for (const auto& item : col)
                {
                    auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
                    //item->value.data2parity = pr - item->value.parity2data + log(1 - error_rate[item_same_position]);
                    item->value.data2parity = pr - item->value.parity2data;
                    if (std::isnan(item->value.data2parity))
                        item->value.data2parity = 0;
                    item->value.data2parity += log(1 - error_rate[item_same_position]);
                }
            }
            // Hard decision
            std::fill(decoding.begin(), decoding.end(), 0);
            auto len{ h_bar.col / 3 };
            for (auto k{ 0ULL }; k < len; k++)
            {
                std::vector list_value{ {log_prob_ratios[k], log_prob_ratios[k + len], log_prob_ratios[k + 2 * len]} };
                auto min_elem{ std::min_element(list_value.begin(), list_value.end()) };
                if (*min_elem < 0)
                {
                    auto idx{ distance(list_value.begin(), min_elem) };
                    decoding[k + idx * len] = 1;
                }
            }
            
        }
        else if ((method == Method::Scaling_PRODUCT_SUM))// method == Method::Scaling_PRODUCT_SUM
        {
            auto ite{ iter + 1 };
            auto alpha{ 1.0 };
            for (auto i{ 0ULL }; i < h_bar.row; i++)
            {
                const auto& row{ h_bar.items_each_row[i] };
                double product_all{ 1.0 };
                for (const auto& item : row)
                    product_all *= tanh(item->value.data2parity / 2);
                for (const auto& item : row)
                {
                    auto dp{ item->value.data2parity };
                    auto same_position_dp{ same_position[i].at(item.get())->value.data2parity };
                    double product_0 = product_all / tanh(dp / 2);
                    if (fabs(dp) < 1e-20)
                    {
                        product_0 = 1;
                        for (const auto& other_item : row)
                            if (other_item->col_index != item->col_index)
                                product_0 *= tanh(other_item->value.data2parity / 2);
                    }
                    double product_1 = product_0 / tanh(same_position_dp / 2);
                    if (fabs(same_position_dp) < 1e-20)
                    {
                        product_1 = 1;
                        for (const auto& other_item : row)
                            if (other_item->col_index != item->col_index && other_item->col_index != same_position[i].at(item.get())->col_index)
                                product_1 *= tanh(other_item->value.data2parity / 2);
                    }
                    if (syndrome[i] == 0)
                        item->value.parity2data = alpha * log((1 + product_0) / (1 - product_1));
                    else
                        item->value.parity2data = alpha * log((1 - product_0) / (1 + product_1));
                }
            }
            // Recompute log-probability-ratios for the bits
            for (auto j{ 0ULL }; j < h_bar.col; j++)
            {
                const auto& col{ h_bar.items_each_col[j] };
                auto pr{ prob_ratio_initial[j] };
                //auto pr{ 0.0 };
                for (const auto& item : col)
                {
                    auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
                    pr += item->value.parity2data;
                    if (std::isnan(pr))
                        pr = 0;
                    pr = pr - log(1 - error_rate[item_same_position]);

                }
                log_prob_ratios[j] = pr;
                //log_prob_ratios[j] = pr + col.back()->value.data2parity;
                for (const auto& item : col)
                {
                    auto item_same_position{ same_position[item->row_index].at(item.get())->col_index };
                    //item->value.data2parity = pr - item->value.parity2data + log(1 - error_rate[item_same_position]);
                    item->value.data2parity = pr - item->value.parity2data;
                    if (std::isnan(item->value.data2parity))
                        item->value.data2parity = 0;
                    item->value.data2parity += log(1 - error_rate[item_same_position]);
                }
            }
            // Hard decision
            std::fill(decoding.begin(), decoding.end(), 0);
            auto len{ h_bar.col / 3 };
            for (auto k{ 0ULL }; k < len; k++)
            {
                std::vector list_value{ {log_prob_ratios[k], log_prob_ratios[k + len], log_prob_ratios[k + 2 * len]} };
                auto min_elem{ std::min_element(list_value.begin(), list_value.end()) };
                if (*min_elem < 0)
                {
                    auto idx{ distance(list_value.begin(), min_elem) };
                    decoding[k + idx * len] = 1;
                }
            }

            }
        else // method == Method::PRODUCT_SUM_NOT_DECOUPLED
        {
            double alpha{ 1.0 };
            for (auto i{0ULL}; i < h_bar.row; i++)
            {
                const auto &row{h_bar.items_each_row[i]};
                double product_all{1.0};
                for (const auto &item : row)
                    product_all *= tanh(item->value.data2parity / 2);
                for (const auto &item : row)
                {
                    auto dp{item->value.data2parity};
                    //auto same_position_dp{same_position[i].at(item.get())->value.data2parity};
                    double product_0 = product_all / tanh(dp / 2);
                    if (fabs(dp) < 1e-20)
                    {
                        product_0 = 1;
                        for (const auto &other_item : row)
                            if (other_item->col_index != item->col_index)
                                product_0 *= tanh(other_item->value.data2parity / 2);
                    }
                    //double product_1 = product_0 / tanh(same_position_dp / 2);
                    //if (fabs(same_position_dp) < 1e-20)
                    //{
                        //product_1 = 1;
                        //for (const auto &other_item : row)
                            //if (other_item->col_index != item->col_index && other_item->col_index != same_position[i].at(item.get())->col_index)
                                //product_1 *= tanh(other_item->value.data2parity / 2);
                    //}
                    if (syndrome[i] == 0)
                        item->value.parity2data = alpha * log((1 + product_0) / (1 - product_0));
                    else
                        item->value.parity2data = alpha * log((1 - product_0) / (1 + product_0));
                }
            }
            // Recompute log-probability-ratios for the bits
            for (auto j{ 0ULL }; j < h_bar.col; j++)
            {
                const auto& col{ h_bar.items_each_col[j] };
                auto pr{ prob_ratio_initial[j] };
                for (const auto& item : col)
                {
                    pr += item->value.parity2data;
                    if (std::isnan(pr))
                        pr = 0;

                }
                log_prob_ratios[j] = pr;
                for (const auto& item : col)
                {
                    item->value.data2parity = pr - item->value.parity2data;
                    if (std::isnan(item->value.data2parity))
                        item->value.data2parity = 0;
                }
            }
            // Hard decision
            std::fill(decoding.begin(), decoding.end(), 0);
            auto len{ h_bar.col};
            for (auto k{ 0ULL }; k < len; k++)
            {
                if (log_prob_ratios[k]<0)
                {  
                    decoding[k] = 1;
                }
            }
        }
      

    }

    size_t BpDecoder::hammingWeight(const std::vector<uint8_t> &src1, const std::vector<uint8_t> &src2)
    {
        // assert(src1.size() == src2.size());
        auto size = src1.size();
        auto result{0ULL};
        for (auto i{0ULL}; i < size; i++)
            result += src1[i] ^ src2[i];
        return result;
    }

    std::vector<uint8_t> BpDecoder::osd0PostProceed(const std::vector<uint8_t> &syndrome,
                                                    const std::vector<double> &probability_distribution)
    {
        // LU decomposition
        auto &&[rank, L, U, rows, cols] = h_bar.luDecomposition(probability_distribution);

        // forward substitution
        std::vector<uint8_t> forward_b(L.col, 0);
        for (auto i{0ULL}; i < L.col; i++)
        {
            auto ii{rows[i]};
            // Look at bits in this row, forming inner product with partial solution, and seeing if the diagonal is 1.
            uint8_t diagonal{0u};
            uint8_t b{0u};
            for (const auto &e : L.items_each_row[ii])
            {
                auto j{e->col_index};
                if (j == i)
                    diagonal = 1;
                else
                    b ^= forward_b[j];
            }
            // Check for no solution if the diagonal isn't 1.
            if (diagonal == 0 && b != syndrome[ii])
                break; // no solution

            forward_b[i] = b ^ syndrome[ii];
        }

        // backward substitution
        std::vector<uint8_t> decoding(U.col, 0);
        for (auto i{U.row}; i-- > 0;)
        {
            auto ii{cols[i]};
            // Look at bits in this row, forming inner product with partial solution, and seeing if the diagonal is 1.
            uint8_t d{0}, b{0};
            for (const auto &e : U.items_each_row[i])
            {
                auto j{e->col_index};
                if (j == ii)
                    d = 1;
                else
                    b ^= decoding[j];
            }
            // Check for no solution if the diagonal isn't 1.
            if (!d && b != forward_b[i])
                break; // no solution
            decoding[ii] = b ^ forward_b[i];
        }

        auto len{U.col / 3};
        for (auto i{0ULL}; i < len; i++)
        {
            if (decoding[i] == 1 && decoding[i + len] == 1 && decoding[i + 2 * len] == 0)
                decoding[i] = 0, decoding[i + len] = 0, decoding[i + 2 * len] = 1;
            else if (decoding[i] == 1 && decoding[i + len] == 0 && decoding[i + 2 * len] == 1)
                decoding[i] = 0, decoding[i + len] = 1, decoding[i + 2 * len] = 0;
            else if (decoding[i] == 0 && decoding[i + len] == 1 && decoding[i + 2 * len] == 1)
                decoding[i] = 1, decoding[i + len] = 0, decoding[i + 2 * len] = 0;
        }
        return decoding;
    }

    BpDecoder::BpDecoder(Method method, int max_iter, const sparse_matrix::Mod2SparseMatrix &h_bar,
                         const std::vector<double> &error_rate)
        : method{method}, max_iter{(0 < max_iter || max_iter < h_bar.col) ? max_iter : h_bar.col}, h_bar{h_bar},
          error_rate{error_rate}
    {
    }

    void BpDecoder::setParams(Method method, int max_iter, const sparse_matrix::Mod2SparseMatrix &h_bar,
                              const std::vector<double> &error_rate)
    {
        this->method = method;
        this->max_iter = (0 < max_iter || max_iter < h_bar.col) ? max_iter : h_bar.col;
        this->h_bar = h_bar;
        this->error_rate = error_rate;
    }

    void BpDecoder::setParams(Method method, int max_iter, sparse_matrix::Mod2SparseMatrix &&h_bar,
                              std::vector<double> &&error_rate)
    {
        this->method = method;
        this->max_iter = (0 < max_iter || max_iter < h_bar.col) ? max_iter : h_bar.col;
        this->h_bar = std::move(h_bar);
        this->error_rate = std::move(error_rate);
    }

    std::tuple<size_t, bool, std::vector<double>, std::vector<uint8_t>> BpDecoder::bpDecode(
        const std::vector<uint8_t> &syndrome)
    {
        if (syndrome.size() != this->h_bar.row)
            throw std::invalid_argument("Syndrome length should be equal to the check matrix's rows.");
        // setup
        auto &&[prob_ratio_initial, log_prob_ratios, same_position, decoding] = this->init();
        // run
        for (auto it{0ULL}; it < max_iter; it++)
        {
            this->update(decoding, log_prob_ratios, syndrome, prob_ratio_initial, it, same_position);
            auto candidate_synd{getSyndrome(h_bar, decoding)};
            auto hamming_weight{this->hammingWeight(syndrome, candidate_synd)};
            if (hamming_weight == 0)
                return std::make_tuple(it, true, log_prob_ratios, decoding);
        }
        return std::make_tuple(max_iter, false, log_prob_ratios, decoding);
    }

    std::tuple<std::vector<uint8_t>, size_t, bool> BpDecoder::bpOsdDecode(const std::vector<uint8_t> &syndrome)
    {
        auto [run_iter, converge, log_prob_ratios, bp_decoding] = this->bpDecode(syndrome);
        if (converge)
        {
            run_iter = run_iter + 1;
            return { bp_decoding, run_iter, true };
        }        
        return {this->osd0PostProceed(syndrome, log_prob_ratios), run_iter, false};
    }
}
