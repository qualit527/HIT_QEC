#include "sparse_matrix.hpp"

#include <sstream>


namespace sparse_matrix
{
    std::vector<uint8_t> Mod2SparseMatrix::operator*(const std::vector<uint8_t> &vec) const
    {
        if (this->col != vec.size())
            throw std::runtime_error("Vec length mismatch matrix col."s);
        std::vector<uint8_t> result(this->row, 0);
        for (auto j{0}; j < this->col; j++)
            if (vec[j])
                for (const auto &item : this->items_each_col[j])
                    result[item->row_index] ^= 1;
        return result;
    }

    Mod2SparseMatrix Mod2SparseMatrix::operator+(const Mod2SparseMatrix &that) const
    {
        if (this->row != that.row || this->col != that.col)
            throw std::runtime_error("Matrices do not have same sizes."s);

        Mod2SparseMatrix result;
        result.row = this->row;
        result.col = this->col;
        result.items_each_row.resize(this->row);
        result.items_each_col.resize(this->col);

        for (auto i{0ULL}; i < this->row; i++)
        {
            auto it_this{this->items_each_row[i].begin()}, &&it_this_end{this->items_each_row[i].end()};
            auto it_other{that.items_each_row[i].begin()}, &&it_other_end{that.items_each_row[i].end()};

            while (it_this != it_this_end && it_other != it_other_end)
            {
                auto &&this_item_col{(*it_this)->col_index}, &&other_item_col{(*it_other)->col_index};
                if (this_item_col < other_item_col)
                {
                    auto item_ptr{::std::make_shared<Item>(i, this_item_col)};
                    result.items_each_row[i].emplace_back(item_ptr);
                    result.items_each_col[this_item_col].emplace_back(item_ptr);
                    ++it_this;
                }
                else if (this_item_col > other_item_col)
                {
                    auto item_ptr{::std::make_shared<Item>(i, other_item_col)};
                    result.items_each_row[i].emplace_back(item_ptr);
                    result.items_each_col[other_item_col].emplace_back(item_ptr);
                    ++it_other;
                }
                else
                {
                    // skip
                    ++it_this;
                    ++it_other;
                }
            }

            while (it_this != it_this_end)
            {
                auto &&this_item_col{(*it_this)->col_index};
                auto item_ptr{::std::make_shared<Item>(i, this_item_col)};
                result.items_each_row[i].emplace_back(item_ptr);
                result.items_each_col[this_item_col].emplace_back(item_ptr);
                ++it_this;
            }

            while (it_other != it_other_end)
            {
                auto &&other_item_col{(*it_other)->col_index};
                auto item_ptr{::std::make_shared<Item>(i, other_item_col)};
                result.items_each_row[i].emplace_back(item_ptr);
                result.items_each_col[other_item_col].emplace_back(item_ptr);
                ++it_other;
            }
        }

        return result;
    }

    void Mod2SparseMatrix::addRow(size_t target_row_index, size_t source_row_index)
    {
        // assert(target_row < this->row && source_row < this->row);
        auto target_row{this->items_each_row[target_row_index]};
        auto source_row{this->items_each_row[source_row_index]};

        auto target_iter{target_row.begin()};
        auto source_iter{source_row.begin()};

        auto source_end{source_row.end()};

        while (target_iter != target_row.end() && source_iter != source_end)
        {
            if ((*target_iter)->col_index > (*source_iter)->col_index)
            {
                this->insert({.row_index = target_row_index, .col_index = (*source_iter)->col_index});
                ++source_iter;
            }
            else if ((*target_iter)->col_index == (*source_iter)->col_index)
            {
                this->remove(*target_iter);
                ++source_iter;
                ++target_iter;
            }
            else
            {
                ++target_iter;
            }
        }

        while (source_iter != source_end)
        {
            this->insert({.row_index = target_row_index, .col_index = (*source_iter)->col_index});
            ++source_iter;
        }
    }

    std::tuple<size_t, Mod2SparseMatrix, Mod2SparseMatrix, std::vector<unsigned long long>, std::vector<unsigned long long>> Mod2SparseMatrix::luDecomposition(
        const std::vector<double> &log_prob_ratios) const
    {
        // Size of sub-matrix to find LU decomposition of.
        auto submatrix_size{std::min(this->row, this->col)};

        // Array where row/col indexes are stored.
        std::vector<unsigned long long>
            row_cur2origin(this->row),
            row_origin2cur(this->row),
            col_cur2origin(this->col),
            col_origin2cur(this->col);

        // Set up initial row and col choices.
        for (auto i{0ULL}; i < this->row; i++)
            row_cur2origin[i] = row_origin2cur[i] = i;
        for (auto j{0ULL}; j < this->col; j++)
            col_cur2origin[j] = j;
        std::sort(col_cur2origin.begin(), col_cur2origin.end(), [&log_prob_ratios](size_t a, size_t b)
                  { return log_prob_ratios[a] < log_prob_ratios[b]; });
        for (auto j{0ULL}; j < this->col; j++)
            col_origin2cur[col_cur2origin[j]] = j;

        // Make a copy of self. will be modified then discarded.
        auto copy{*this};

        // Decomposition result.
        Mod2SparseMatrix
            L(this->row, submatrix_size),
            U(submatrix_size, this->col);

        // Find L and U one column at a time.
        auto num_not_found{0ULL};
        for (auto i{0ULL}; i < submatrix_size; i++)
        {
            // Choose the next row and column of 'copy'
            bool found_next_col{false};
            size_t next_col_cur;
            std::shared_ptr<Item> result;
            for (next_col_cur = i; next_col_cur < this->col; next_col_cur++)
            {
                for (const auto &e : copy.items_each_col[col_cur2origin[next_col_cur]])
                {
                    if (row_origin2cur[e->row_index] >= i)
                    {
                        found_next_col = true;
                        result = e;
                        break;
                    }
                }
                if (found_next_col)
                    break;
            }

            if (!found_next_col)
            {
                num_not_found++;
            }
            else
            {
                // Update 'row_cur2origin' and 'col_cur2origin'. Looks at 'next_col_cur' and 'result' found above.
                std::swap(col_cur2origin[next_col_cur], col_cur2origin[i]);
                col_origin2cur[col_cur2origin[next_col_cur]] = next_col_cur;
                col_origin2cur[col_cur2origin[i]] = i;
                auto next_row_cur{row_origin2cur[result->row_index]};
                std::swap(row_cur2origin[next_row_cur], row_cur2origin[i]);
                row_origin2cur[row_cur2origin[next_row_cur]] = next_row_cur;
                row_origin2cur[row_cur2origin[i]] = i;
            }

            // Update 'L', 'U', and 'copy'.
            auto &col{copy.items_each_col[col_cur2origin[i]]};
            for (auto f{0ULL}; f < col.size(); f++)
            {
                auto origin_row{col[f]->row_index};
                if (row_origin2cur[origin_row] > i)
                {
                    copy.addRow(origin_row, result->row_index);
                    if (f >= col.size() || col[f]->row_index != origin_row)
                        f--;
                    L.insert({.row_index = origin_row, .col_index = i});
                }
                else if (row_origin2cur[origin_row] < i)
                {
                    U.insert({.row_index = row_origin2cur[origin_row], .col_index = col_cur2origin[i]});
                }
                else
                {
                    L.insert({.row_index = origin_row, .col_index = i});
                    U.insert({.row_index = i, .col_index = col_cur2origin[i]});
                }
            }
            // // Get rid of all entries in the current column of 'copy', just to save space.
            // for (auto f{0ULL}; f < col.size(); f++)
            //     copy.remove(col[f]);
        }

        // Get rid of all entries in the rows of L past row 'submatrix_size', after reordering.
        for (auto i{submatrix_size}; i < this->row; i++)
        {
            auto row{L.items_each_row[row_cur2origin[i]]};
            for (auto f : L.items_each_row[row_cur2origin[i]])
                L.remove(f);
        }

        return {
            submatrix_size - num_not_found, // rank
            L,
            U,
            row_cur2origin,
            col_cur2origin,
        };
    }

    std::string Mod2SparseMatrix::toString() const
    {
        std::stringstream ss;
        for (auto const &row : this->items_each_row)
        {
            auto elem_iter{row.begin()};
            for (size_t curr_index{0ULL}; curr_index < this->col; curr_index++)
            {
                if (elem_iter == row.end() || (*elem_iter)->col_index > curr_index)
                {
                    ss << "0 ";
                }
                else
                {
                    ss << "1 ";
                    ++elem_iter;
                }
            }
            ss << '\n';
        }
        return ss.str();
    }
}
