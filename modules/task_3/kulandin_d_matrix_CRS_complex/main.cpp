// Copyright 2021 Kulandin Denis
#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include "./sparsematrix.h"

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
    ASSERT_EQ(b.getValues(),   a.getValues());
    ASSERT_EQ(b.getCols(),     a.getCols());
    ASSERT_EQ(b.getPointers(), a.getPointers());
}

TEST(matrix_CSR_complex, operator_equal) {
    SparseMatrix a(1);
    SparseMatrix b(25);
    b = a;
    ASSERT_EQ(b.getSize(),     a.getSize());
    ASSERT_EQ(b.getValues(),   a.getValues());
    ASSERT_EQ(b.getCols(),     a.getCols());
    ASSERT_EQ(b.getPointers(), a.getPointers());
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
    ASSERT_EQ(a, b.getDenseMatrix());
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
    ASSERT_EQ(a, b.transposition().transposition().getDenseMatrix());
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
    SparseMatrix c = b * b;
    ASSERT_EQ(c.getSize(), size);
    ASSERT_EQ(c.getDenseMatrix(), ans);
    ASSERT_EQ(b.operator*(b).getDenseMatrix(), ans);
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
    ASSERT_THROW(matrA * matrB, std::string);
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
    SparseMatrix res = matrA * matrB;
    ASSERT_EQ(res.getDenseMatrix(), ans);
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
    auto begin = tbb::tick_count::now();
    SparseMatrix seq_res = a * b;
    auto end = tbb::tick_count::now();
    auto elapsed_ms = (end - begin).seconds();
    std::cout << "Sequential time = " << elapsed_ms << "s\n";

    begin = tbb::tick_count::now();
    SparseMatrix openmp_res = a.openMPMultiplication(b);
    end = tbb::tick_count::now();
    elapsed_ms = (end - begin).seconds();
    std::cout << "openMP time = " << elapsed_ms << "s\n";

    begin = tbb::tick_count::now();
    SparseMatrix tbb_res = a.TBBMultiplication(b, 4);
    end = tbb::tick_count::now();
    elapsed_ms = (end - begin).seconds();
    std::cout << "TBB time = " << elapsed_ms << "s\n";

    ASSERT_EQ(tbb_res.getSize(),     openmp_res.getSize());
    ASSERT_EQ(tbb_res.getCols(),     openmp_res.getCols());
    ASSERT_EQ(tbb_res.getValues(),   openmp_res.getValues());
    ASSERT_EQ(tbb_res.getPointers(), openmp_res.getPointers());
    ASSERT_EQ(tbb_res.getSize(),     seq_res.getSize());
    ASSERT_EQ(tbb_res.getCols(),     seq_res.getCols());
    ASSERT_EQ(tbb_res.getValues(),   seq_res.getValues());
    ASSERT_EQ(tbb_res.getPointers(), seq_res.getPointers());
}

INSTANTIATE_TEST_SUITE_P(matrix_CSR_complex,
                         parametrized_matrix_multiplication,
                         testing::Combine(
    testing::Values(500),
    testing::Values(5, 10, 15)
));

