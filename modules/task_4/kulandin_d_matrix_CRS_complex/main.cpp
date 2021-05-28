// Copyright 2021 Kulandin Denis
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include "./sparsematrix.h"
#include "../../../3rdparty/unapproved/unapproved.h"

template<class T>
void checkArrays(const std::vector<T>& a, const std::vector<T>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_FLOAT_EQ(a[i], b[i]);
    }
}

template<>
void checkArrays<std::complex<double>>(
        const std::vector<std::complex<double>>& a,
        const std::vector<std::complex<double>>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_FLOAT_EQ(a[i].real(), b[i].real());
        ASSERT_FLOAT_EQ(a[i].imag(), b[i].imag());
    }
}

TEST(matrix_CSR_complex, int_constructor) {
    SparseMatrix a(1);
    ASSERT_EQ(static_cast<int>(a.getSize()),            1);
    ASSERT_EQ(static_cast<int>(a.getValues().size()),   0);
    ASSERT_EQ(static_cast<int>(a.getCols().size()),     0);
    ASSERT_EQ(static_cast<int>(a.getPointers().size()), 0);
}

TEST(matrix_CSR_complex, copy_constructor) {
    SparseMatrix a(1);
    SparseMatrix b(a);
    ASSERT_EQ(b.getSize(),     a.getSize());
    checkArrays(b.getValues(),   a.getValues());
    checkArrays(b.getCols(),     a.getCols());
    checkArrays(b.getPointers(), a.getPointers());
}

TEST(matrix_CSR_complex, operator_equal) {
    SparseMatrix a(1);
    SparseMatrix b(25);
    b = a;
    ASSERT_EQ(b.getSize(),       a.getSize());
    checkArrays(b.getValues(),   a.getValues());
    checkArrays(b.getCols(),     a.getCols());
    checkArrays(b.getPointers(), a.getPointers());
}

TEST(matrix_CSR_complex, matrix_constructor_and_createDenseMatrix) {
    int size = 5;
    std::vector<std::complex<double>> a = {
        {1, 0},  {-1, 0}, {0, 0}, {-3, 0}, {0, 0},
        {-2, 0}, {5, 0},  {0, 0}, {0, 0},  {0, 0},
        {0, 0},  {0, 0},  {4, 0}, {6, 0},  {4, 0},
        {-4, 0}, {0, 0},  {2, 0}, {7, 0},  {0, 0},
        {0, 0},  {8, 0},  {0, 0}, {0, 0},  {-5, 0}
    };
    SparseMatrix b(a, size);
    ASSERT_EQ(b.getSize(), size);
    checkArrays(a, b.getDenseMatrix());
}

TEST(matrix_CSR_complex, transposition_and_getDenseMatrix) {
    int size = 4;
    std::vector<std::complex<double>> a = {
        {0, 0}, {3, 0}, {0, 0},  {7, 0},
        {0, 0}, {0, 0}, {8, 0},  {0, 0},
        {0, 0}, {0, 0}, {0, 0},  {0, 0},
        {9, 0}, {0, 0}, {15, 0}, {16, 0}
    };
    SparseMatrix b(a, size);
    ASSERT_EQ(b.getSize(), size);
    checkArrays(a, b.transposition().transposition().getDenseMatrix());
}

TEST(matrix_CSR_complex, square_matrix_multiplication) {
    int size = 4;
    std::vector<std::complex<double>> a = {
        {0, 0}, {3, 0}, {0, 0},  {7, 0},
        {0, 0}, {0, 0}, {8, 0},  {0, 0},
        {0, 0}, {0, 0}, {0, 0},  {0, 0},
        {9, 0}, {0, 0}, {15, 0}, {16, 0}
    };
    std::vector<std::complex<double>> ans = {
        {63, 0},  {0, 0},  {129, 0}, {112, 0},
        {0, 0},   {0, 0},  {0, 0},   {0, 0},
        {0, 0},   {0, 0},  {0, 0},   {0, 0},
        {144, 0}, {27, 0}, {240, 0}, {319, 0}
    };
    SparseMatrix b(a, size);
    ASSERT_NO_THROW(b.operator*(b));
    SparseMatrix c = b.threadMultiplication(b);
    ASSERT_EQ(c.getSize(), size);
    checkArrays(c.getDenseMatrix(), ans);
    checkArrays(b.operator*(b).getDenseMatrix(), ans);
}

TEST(matrix_CSR_complex, exception_matrix_multiplication) {
    int sizeA = 1;
    std::vector<std::complex<double>> a = {
        {0, 0}
    };
    int sizeB = 2;
    std::vector<std::complex<double>> b = {
        {0, 0}, {0, 0},
        {0, 0}, {0, 0}
    };
    SparseMatrix matrA(a, sizeA);
    SparseMatrix matrB(b, sizeB);
    ASSERT_THROW(matrA.threadMultiplication(matrB), std::string);
}

TEST(matrix_CSR_complex, matrix_multiplication_only0) {
    int sizeA = 2;
    std::vector<std::complex<double>> a = {
        {0, 0}, {0, 0},
        {0, 0}, {0, 0}
    };
    int sizeB = 2;
    std::vector<std::complex<double>> b = {
        {0, 0}, {0, 0},
        {0, 0}, {0, 0}
    };
    std::vector<std::complex<double>> ans = {
        {0, 0}, {0, 0},
        {0, 0}, {0, 0}
    };
    SparseMatrix matrA(a, sizeA);
    SparseMatrix matrB(b, sizeB);
    SparseMatrix res = matrA.threadMultiplication(matrB);
    checkArrays(res.getDenseMatrix(), ans);
}

typedef testing::TestWithParam<std::tuple<int, int>>
    parametrized_matrix_multiplication;

TEST_P(parametrized_matrix_multiplication, mult_small_dimensions) {
    int size       = std::get<0>(GetParam());
    int nonZero    = std::get<1>(GetParam());
    if (nonZero > size) {
        ASSERT_THROW(generateRandomSparseMatrix(size, nonZero), std::string);
        return;
    }
    SparseMatrix a = generateRandomSparseMatrix(size, nonZero);
    SparseMatrix b = generateRandomSparseMatrix(size, nonZero);

    std::cout << "size = " << size <<
        "; nonZeroElementsInEveryRow = " << nonZero << '\n';
    auto begin           = std::chrono::high_resolution_clock::now();
    SparseMatrix seq_res = a * b;
    auto end             = std::chrono::high_resolution_clock::now();
    auto elapsed_ms      =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Sequential time =  " << elapsed_ms << "ms\n";

    begin                = std::chrono::high_resolution_clock::now();
    SparseMatrix tbb_res = a.threadMultiplication(b);
    end                  = std::chrono::high_resolution_clock::now();
    elapsed_ms           =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "std::thread time = " << elapsed_ms << "ms\n";

    ASSERT_EQ(tbb_res.getSize(),       seq_res.getSize());
    checkArrays(tbb_res.getValues(),   seq_res.getValues());
    checkArrays(tbb_res.getCols(),     seq_res.getCols());
    checkArrays(tbb_res.getPointers(), seq_res.getPointers());
}

INSTANTIATE_TEST_SUITE_P(matrix_CSR_complex,
                         parametrized_matrix_multiplication,
                         testing::Combine(
    testing::Values(500),
    testing::Values(1, 5, 10)
));

