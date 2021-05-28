// Copyright 2021 Alibekov Murad
#include "../../../modules/task_3/alibekov_m_ccs_complex_matrix/ccs_complex_matrix.h"

ccs_complex_matrix generate_regular_ccs(int seed, int N, int count_in_col) {
    if ((N <= 0) || (count_in_col <= 0))
        throw -1;

    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(seed));

    ccs_complex_matrix random_matrix(N, count_in_col * N);

    for (int i = 0; i < N; i++) {
        for (int j = i * count_in_col; j < (i + 1) * count_in_col; j++) {
            bool isFound = false;
            do {
                random_matrix.rows[j] = static_cast<unsigned int>(gen()) % N;
                isFound = true;
                for (int k = i * count_in_col; k < j; k++) {
                    if (random_matrix.rows[j] == random_matrix.rows[k])
                        isFound = false;
                    }
            } while (!isFound);
        }

        // BubbleSort
        for (int j = 0; j < count_in_col - 1; j++)
            for (int k = i * count_in_col; k < (i + 1) * count_in_col - 1 - j; k++)
                if (random_matrix.rows[k] > random_matrix.rows[k + 1])
                    std::swap(random_matrix.rows[k], random_matrix.rows[k + 1]);
    }

    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<> rnd(-30, 30);

    for (int i = 0; i < count_in_col * N; i++)
        random_matrix.values[i] = { rnd(mersenne), rnd(mersenne) };

    for (int i = 0; i < N + 1; i++)
        random_matrix.col_indexes[i] = i * count_in_col;

    return random_matrix;
}

ccs_complex_matrix transpose(const ccs_complex_matrix &A) {
    ccs_complex_matrix AT(A.N, A.NZ);

    for (int i = 0; i < A.NZ; i++)
        AT.col_indexes[A.rows[i] + 1]++;

    int S = 0;
    for (int i = 1; i <= A.N; i++) {
        int tmp = AT.col_indexes[i];
        AT.col_indexes[i] = S;
        S = S + tmp;
    }

    for (int i = 0; i < A.N; i++) {
        int row = i;
        for (int j = A.col_indexes[i]; j < A.col_indexes[i + 1]; j++) {
            std::complex<double> AT_V = A.values[j];
            int AT_col_index = A.rows[j];
            int AT_j_index = AT.col_indexes[AT_col_index + 1];
            AT.values[AT_j_index] = AT_V;
            AT.rows[AT_j_index] = row;
            AT.col_indexes[AT_col_index + 1]++;
        }
    }

    return AT;
}

ccs_complex_matrix naive_multiplicate(const ccs_complex_matrix &A, const ccs_complex_matrix &B) {
    ccs_complex_matrix AT = transpose(A);
    if (A.N != B.N)
        throw -1;

    int N = A.N;

    int rows_count = 0;
    std::vector<int> rows;
    std::vector<std::complex<double> > values;
    std::vector<int> col_indexes;

    col_indexes.push_back(0);
    for (int i = 0; i < N; i++) {
        int count_NZ = 0;
        for (int j = 0; j < N; j++) {
            std::complex<double> sum = {0, 0};

            // dot_product
            for (int k = AT.col_indexes[j]; k < AT.col_indexes[j + 1]; k++)
                for (int l = B.col_indexes[i]; l < B.col_indexes[i + 1]; l++)
                    if (AT.rows[k] == B.rows[l]) {
                        sum += AT.values[k] * B.values[l];
                        break;
                    }

            if ((fabs(sum.real()) > ZERO_IN_CCS) || (fabs(sum.imag()) > ZERO_IN_CCS)) {
                rows.push_back(j);
                rows_count++;
                values.push_back(sum);
                count_NZ++;
            }
        }
        col_indexes.push_back(count_NZ + col_indexes[i]);
    }

    ccs_complex_matrix C(N, rows_count);
    for (int j = 0; j < rows_count; j++) {
        C.rows[j] = rows[j];
        C.values[j] = values[j];
    }

    for (int i = 0; i < N + 1; i++)
        C.col_indexes[i] = col_indexes[i];

    return C;
}

ccs_complex_matrix optim_multiplicate(const ccs_complex_matrix &A, const ccs_complex_matrix &B) {
    ccs_complex_matrix AT = transpose(A);
    if (A.N != B.N)
        throw -1;

    int N = A.N;

    std::vector<int> rows;
    std::vector<std::complex<double> > values;
    std::vector<int> col_indexes;

    col_indexes.push_back(0);
    int count_NZ = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> sum = {0, 0};

            // like merging 2 sorted arrays
            int a_idx = AT.col_indexes[j];
            int b_idx = B.col_indexes[i];
            while ((a_idx < AT.col_indexes[j + 1]) && (b_idx < B.col_indexes[i + 1])) {
                if (AT.rows[a_idx] < B.rows[b_idx]) {
                    a_idx++;
                } else {
                    if (AT.rows[a_idx] > B.rows[b_idx])
                        b_idx++;
                    else
                        sum += AT.values[a_idx++] * B.values[b_idx++];
                }
            }

            if ((fabs(sum.real()) > ZERO_IN_CCS) || (fabs(sum.imag()) > ZERO_IN_CCS)) {
                rows.push_back(j);
                values.push_back(sum);
                count_NZ++;
            }
        }
        col_indexes.push_back(count_NZ);
    }

    ccs_complex_matrix C(N, count_NZ);

    for (int j = 0; j < count_NZ; j++) {
        C.rows[j] = rows[j];
        C.values[j] = values[j];
    }

    for (int i = 0; i < N + 1; i++)
        C.col_indexes[i] = col_indexes[i];

    return C;
}

ccs_complex_matrix naive_multiplicate_omp(const ccs_complex_matrix &A, const ccs_complex_matrix &B) {
    ccs_complex_matrix AT = transpose(A);
    if (A.N != B.N)
        throw -1;

    int N = A.N;

    std::vector<std::vector<int> > rows(N);
    std::vector<std::vector<std::complex<double> > > values(N);
    std::vector<int> col_indexes(N + 1);

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> sum = {0, 0};

            // dot_product
            for (int k = AT.col_indexes[j]; k < AT.col_indexes[j + 1]; k++)
                for (int l = B.col_indexes[i]; l < B.col_indexes[i + 1]; l++)
                    if (AT.rows[k] == B.rows[l]) {
                        sum += AT.values[k] * B.values[l];
                        break;
                    }

            if ((fabs(sum.real()) > ZERO_IN_CCS) || (fabs(sum.imag()) > ZERO_IN_CCS)) {
                rows[i].push_back(j);
                values[i].push_back(sum);
                col_indexes[i]++;
            }
        }
    }

    int count_NZ = 0;
    for (int i = 0; i < N; i++) {
        int tmp = col_indexes[i];
        col_indexes[i] = count_NZ;
        count_NZ += tmp;
    }
    col_indexes[N] = count_NZ;

    ccs_complex_matrix C(N, count_NZ);
    int count = 0;
    for (int i = 0; i < N; i++) {
        int size = rows[i].size();
        for (int j = 0; j < size; j++) {
            C.rows[count] = rows[i][j];
            C.values[count] = values[i][j];
            count++;
        }
    }

    for (int i = 0; i < N + 1; i++)
        C.col_indexes[i] = col_indexes[i];

    return C;
}

ccs_complex_matrix optim_multiplicate_omp(const ccs_complex_matrix &A, const ccs_complex_matrix &B) {
    ccs_complex_matrix AT = transpose(A);
    if (A.N != B.N)
        throw -1;

    int N = A.N;

    std::vector<std::vector<int> > rows(N);
    std::vector<std::vector<std::complex<double> > > values(N);
    std::vector<int> col_indexes(N + 1);

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> sum = {0, 0};

            // like merging 2 sorted arrays
            int a_idx = AT.col_indexes[j];
            int b_idx = B.col_indexes[i];
            while ((a_idx < AT.col_indexes[j + 1]) && (b_idx < B.col_indexes[i + 1])) {
                if (AT.rows[a_idx] < B.rows[b_idx]) {
                    a_idx++;
                } else {
                    if (AT.rows[a_idx] > B.rows[b_idx])
                        b_idx++;
                    else
                        sum += AT.values[a_idx++] * B.values[b_idx++];
                }
            }

            if ((fabs(sum.real()) > ZERO_IN_CCS) || (fabs(sum.imag()) > ZERO_IN_CCS)) {
                rows[i].push_back(j);
                values[i].push_back(sum);
                col_indexes[i]++;
            }
        }
    }

    int count_NZ = 0;
    for (int i = 0; i < N; i++) {
        int tmp = col_indexes[i];
        col_indexes[i] = count_NZ;
        count_NZ += tmp;
    }
    col_indexes[N] = count_NZ;

    ccs_complex_matrix C(N, count_NZ);
    int count = 0;
    for (int i = 0; i < N; i++) {
        int size = rows[i].size();
        for (int j = 0; j < size; j++) {
            C.rows[count] = rows[i][j];
            C.values[count] = values[i][j];
            count++;
        }
    }

    for (int i = 0; i < N + 1; i++)
        C.col_indexes[i] = col_indexes[i];

    return C;
}

ccs_complex_matrix naive_multiplicate_tbb(const ccs_complex_matrix &A,
        const ccs_complex_matrix &B,
        int _threads_count,
        int _gransize) {
    ccs_complex_matrix AT = transpose(A);

    if (A.N != B.N)
        throw -1;
    if (_threads_count < 0)
        throw -1;
    if (_gransize < 0)
        throw -1;

    int N = A.N;
    int threads_count = _threads_count;
    int gransize = _gransize;

    std::vector<std::vector<int> > rows(N);
    std::vector<std::vector<std::complex<double> > > values(N);
    std::vector<int> col_indexes(N + 1);

    tbb::task_scheduler_init init(threads_count);
    tbb::parallel_for(tbb::blocked_range<int>(0, N, gransize),
        [&](const tbb::blocked_range<int> &r) {
            int begin = r.begin();
            int end = r.end();
            for (int i = begin; i < end; i++) {
                for (int j = 0; j < N; j++) {
                    std::complex<double> sum = {0, 0};

                    // dot_product
                    for (int k = AT.col_indexes[j]; k < AT.col_indexes[j + 1]; k++)
                        for (int l = B.col_indexes[i]; l < B.col_indexes[i + 1]; l++)
                            if (AT.rows[k] == B.rows[l]) {
                                sum += AT.values[k] * B.values[l];
                                break;
                            }

                    if ((fabs(sum.real()) > ZERO_IN_CCS) || (fabs(sum.imag()) > ZERO_IN_CCS)) {
                        rows[i].push_back(j);
                        values[i].push_back(sum);
                        col_indexes[i]++;
                    }
                }
            }
        });
    init.terminate();

    int count_NZ = 0;
    for (int i = 0; i < N; i++) {
        int tmp = col_indexes[i];
        col_indexes[i] = count_NZ;
        count_NZ += tmp;
    }
    col_indexes[N] = count_NZ;

    ccs_complex_matrix C(N, count_NZ);
    int count = 0;
    for (int i = 0; i < N; i++) {
        int size = rows[i].size();
        for (int j = 0; j < size; j++) {
            C.rows[count] = rows[i][j];
            C.values[count] = values[i][j];
            count++;
        }
    }

    for (int i = 0; i < N + 1; i++)
        C.col_indexes[i] = col_indexes[i];

    return C;
}

ccs_complex_matrix optim_multiplicate_tbb(const ccs_complex_matrix &A,
        const ccs_complex_matrix &B,
        int _threads_count,
        int _gransize) {
    ccs_complex_matrix AT = transpose(A);

    if (A.N != B.N)
        throw -1;
    if (_threads_count < 0)
        throw -1;
    if (_gransize < 0)
        throw -1;

    int N = A.N;
    int threads_count = _threads_count;
    int gransize = _gransize;

    std::vector<std::vector<int> > rows(N);
    std::vector<std::vector<std::complex<double> > > values(N);
    std::vector<int> col_indexes(N + 1);

    tbb::task_scheduler_init init(threads_count);
    tbb::parallel_for(tbb::blocked_range<int>(0, N, gransize),
        [&](const tbb::blocked_range<int> &r) {
            int begin = r.begin();
            int end = r.end();
            for (int i = begin; i < end; i++) {
                for (int j = 0; j < N; j++) {
                    std::complex<double> sum = {0, 0};

                    // like merging 2 sorted arrays
                    int a_idx = AT.col_indexes[j];
                    int b_idx = B.col_indexes[i];
                    while ((a_idx < AT.col_indexes[j + 1]) && (b_idx < B.col_indexes[i + 1])) {
                        if (AT.rows[a_idx] < B.rows[b_idx]) {
                            a_idx++;
                        } else {
                            if (AT.rows[a_idx] > B.rows[b_idx])
                                b_idx++;
                            else
                                sum += AT.values[a_idx++] * B.values[b_idx++];
                        }
                    }

                    if ((fabs(sum.real()) > ZERO_IN_CCS) || (fabs(sum.imag()) > ZERO_IN_CCS)) {
                        rows[i].push_back(j);
                        values[i].push_back(sum);
                        col_indexes[i]++;
                    }
                }
            }
        });
    init.terminate();

    int count_NZ = 0;
    for (int i = 0; i < N; i++) {
        int tmp = col_indexes[i];
        col_indexes[i] = count_NZ;
        count_NZ += tmp;
    }
    col_indexes[N] = count_NZ;

    ccs_complex_matrix C(N, count_NZ);
    int count = 0;
    for (int i = 0; i < N; i++) {
        int size = rows[i].size();
        for (int j = 0; j < size; j++) {
            C.rows[count] = rows[i][j];
            C.values[count] = values[i][j];
            count++;
        }
    }

    for (int i = 0; i < N + 1; i++)
        C.col_indexes[i] = col_indexes[i];

    return C;
}

void PrintCCSMatrix(const ccs_complex_matrix &A, bool isComplex) {
    std::cout << "Matrix [" << &A << "] : \n\tvalues: [ ";
    for (int i = 0; i < A.NZ; i++)
        isComplex ? std::cout << A.values[i] << " " : std::cout << A.values[i].real() << " ";
    std::cout << "]\n\trows: [ ";
    for (int i = 0; i < A.NZ; i++)
        std::cout << A.rows[i] << " ";
    std::cout << "]\n\tcol_indexes: [ ";
    for (int i = 0; i < A.N + 1; i++)
        std::cout << A.col_indexes[i] << " ";
    std::cout << "]\n";
}

void PrintDensificationOfCCSMatrix(const ccs_complex_matrix &A, bool isComplex) {
    int N = A.N;
    std::vector<std::vector<std::complex<double> > > dense_matrix(N);
    for (int i = 0; i < N; i++)
        dense_matrix[i] = std::vector<std::complex<double> >(N);
    for (int i = 0; i < N; i++)
        for (int j = A.col_indexes[i]; j < A.col_indexes[i + 1]; j++)
            dense_matrix[A.rows[j]][i] = A.values[j];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            isComplex ? std::cout << dense_matrix[i][j] << " " :
                std::cout << dense_matrix[i][j].real() << " ";
        }
        std::cout << std::endl;
    }
}

bool operator==(const ccs_complex_matrix &A, const ccs_complex_matrix &B) {
    return ((A.N == B.N) && (A.NZ == B.NZ) && (A.rows == B.rows)
        && (A.values == B.values) && (A.col_indexes == B.col_indexes));
}
