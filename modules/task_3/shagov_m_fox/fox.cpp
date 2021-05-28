// Copyright 2021 Shagov Maksim
#include "../../../modules/task_3/shagov_m_fox/fox.h"
#include <tbb/tbb.h>

bool isEqual(double x, double y) {
    return std::fabs(x - y) < 0.001;
}

bool isEqualMatrix(const Matrix& A, const Matrix& B) {
    if ((A.size() <= 0) || (B.size() <= 0))
        throw "Size of matrix must be > 0";
    if (A.size() != B.size())
        throw "Different size of matrix";
    size_t size = A.size();
    for (size_t i = 0; i < size; i++)
        if (!isEqual(A[i], B[i]))
            return false;
    return true;
}

bool isSizeCorrect(size_t size, size_t t_count) {
    size_t blocks_count = static_cast<size_t>(sqrt(t_count));
    if (size % blocks_count != 0) {
        return false;
    }
    if (blocks_count * blocks_count != t_count) {
        return false;
    }
    return true;
}

Matrix createRandomMatrix(size_t size) {
    if (size <= 0)
        throw "Size of matrix must be > 0";
    Matrix result(size, 0);
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<> urd(-100.0, 100.0);
    for (size_t i = 0; i < size; i++) {
        result[i] = static_cast<double>(urd(mersenne));
    }
    return result;
}

Matrix sequentialMatrixMultiplication(const std::vector<double>& A, const std::vector<double>& B, size_t Size) {
    if (Size <= 0)
        throw "Block size of matrix must be > 0";
    if ((A.size() <= 0) || (B.size() <= 0))
        throw "Size of matrix must be > 0";
    if (A.size() != B.size())
        throw "Different size of matrix";
    if ((A.size() != Size) || (B.size() != Size))
        throw "Different param and size";
    size_t BlockSize = static_cast<size_t>(sqrt(Size));
    Matrix result(BlockSize * BlockSize, 0);
    for (size_t i = 0; i < BlockSize; i++)
        for (size_t j = 0; j < BlockSize; j++)
            for (size_t k = 0; k < BlockSize; k++)
                result[i * BlockSize  + j] += A[i * BlockSize + k] * B[k * BlockSize + j];
    return result;
}

Matrix sequentialBlockMatrixMultiplication(const std::vector<double>& A, const std::vector<double>& B, size_t Size) {
    if (Size <= 0)
        throw "Block size of matrix must be > 0";
    if ((A.size() <= 0) || (B.size() <= 0))
        throw "Size of matrix must be > 0";
    if (A.size() != B.size())
        throw "Different size of matrix";
    if ((A.size() != Size) || (B.size() != Size))
        throw "Different param and size";
    if (static_cast<size_t>(sqrt(Size)) * static_cast<size_t>(sqrt(Size)) != Size)
        throw "Size not square";
    size_t BlockSize = static_cast<size_t>(sqrt(Size));
    size_t BlockCount = static_cast<size_t>(BlockSize/static_cast<size_t>(sqrt(4))) == 0
                       ? 1 : static_cast<size_t>(BlockSize/static_cast<size_t>(sqrt(4)));
    Matrix result(BlockSize * BlockSize, 0);
    for (size_t i = 0; i < BlockSize; i += BlockCount)
        for (size_t j = 0; j < BlockSize; j += BlockCount)
            for (size_t k = 0; k < BlockSize; k += BlockCount)
                for (size_t ii = i; ii < ((BlockCount + i) % BlockSize + BlockCount); ii++)
                    for (size_t jj = j; jj < ((BlockCount + j) % BlockSize + BlockCount); jj++)
                        for (size_t kk = k; kk < ((BlockCount + k) % BlockSize + BlockCount); kk++)
                            result[ii * BlockSize  + jj] += A[ii * BlockSize + kk] * B[kk * BlockSize + jj];
    return result;
}

Matrix parallelBlockMatrixMultiplication(const std::vector<double>& A, const std::vector<double>& B,
                                         size_t Size, int threads_count) {
    if (Size <= 0)
        throw "Block size of matrix must be > 0";
    if ((A.size() <= 0) || (B.size() <= 0))
        throw "Size of matrix must be > 0";
    if (A.size() != B.size())
        throw "Different size of matrix";
    if ((A.size() != Size) || (B.size() != Size))
        throw "Different param and size";
    if (static_cast<size_t>(sqrt(Size)) * static_cast<size_t>(sqrt(Size)) != Size)
        throw "Size not square";
    Matrix result(Size, 0);
    tbb::task_scheduler_init init(threads_count);
    tbb::parallel_for(tbb::blocked_range<int>(0, threads_count, 1), [&A, &B, &result, &threads_count, &Size]
    (tbb::blocked_range<int>& r) -> void{
            size_t cols = static_cast<size_t>(sqrt(Size));
            size_t blocks_count = static_cast<size_t>(sqrt(threads_count));
            size_t block_cols_size = cols / blocks_count;
            auto thread_num = r.begin();
            size_t i1 = thread_num / blocks_count, j1 = thread_num % blocks_count;
            auto A1 = A.data();
            auto B1 = B.data();
            auto C1 = result.data();
            for (size_t stage = 0; stage < blocks_count; stage++) {
                A1 = A.data() + (i1 * cols + ((i1 + stage) % blocks_count)) * block_cols_size;
                B1 = B.data() + (((i1 + stage) % blocks_count) * cols + j1) * block_cols_size;
                C1 = result.data() + (i1 * cols + j1) * block_cols_size;
                for (size_t i = 0; i < block_cols_size; i++) {
                    for (size_t j = 0; j < block_cols_size; j++) {
                        double tmp = 0.0;
                        for (size_t k = 0; k < block_cols_size; k++) {
                            tmp += *(A1 + i * cols + k) * *(B1 + k * cols + j);
                        }
                        *(C1 + i * cols + j) += tmp;
                    }
                }
            }
        });
    return result;
}
