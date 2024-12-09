#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

constexpr int n = 1024;

vector<vector<int>> DataA(n, vector<int>(n, 0));
vector<vector<int>> DataB(n, vector<int>(n, 0));
vector<vector<int>> DataC(n, vector<int>(n, 0));
vector<vector<int>> DataCTruth(n, vector<int>(n, 0));

void initA() {
    for (int h = 0; h < n; h++) {
        for (int w = 0; w < n; w++) {
            DataA[h][w] = rand() % 10;
        }
    }
}

void initB() {
    for (int h = 0; h < n; h++) {
        for (int w = 0; w < n; w++) {
            DataB[h][w] = rand() % 10;
        }
    }
}

void initCTruth() {
    for (int h = 0; h < n; h++) {
        for (int w = 0; w < n; w++) {
            DataCTruth[h][w] = 0;
            for (int k = 0; k < n; k++) {
                DataCTruth[h][w] += DataA[h][k] * DataB[k][w];
            }
        }
    }
}

void init() {
    initA();
    initB();
    initCTruth();
}

void test() {
    for (int h = 0; h < n; h++) {
        for (int w = 0; w < n; w++) {
            assert(DataC[h][w] == DataCTruth[h][w]);
        }
    }
}

void add_matrix(const vector<vector<int>>& DataA, const vector<vector<int>>& DataB, vector<vector<int>>& result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = DataA[i][j] + DataB[i][j];
        }
    }
}

void subtract_matrix(const vector<vector<int>>& DataA, const vector<vector<int>>& DataB, vector<vector<int>>& result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = DataA[i][j] - DataB[i][j];
        }
    }
}

void matmul(const vector<vector<int>>& matA, const vector<vector<int>>& matB, vector<vector<int>>& matRes, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matRes[i][j] = 0;
            for (int k = 0; k < size; k++) {
                matRes[i][j] += matA[i][k] * matB[k][j];
            }
        }
    }
}


void strassen(const vector<vector<int>>& matA, const vector<vector<int>>& matB, vector<vector<int>>& matRes, int size) {
    //if (size == 1) {
    //    matRes[0][0] = matA[0][0] * matB[0][0];
    //    return;
    //}

    if (size <= 64) {
        matmul(matA, matB, matRes, size);
        return;
    }

    int newSize = size / 2;

    vector<vector<int>> A(newSize, vector<int>(newSize));
    vector<vector<int>> B(newSize, vector<int>(newSize));
    vector<vector<int>> C(newSize, vector<int>(newSize));
    vector<vector<int>> D(newSize, vector<int>(newSize));
    vector<vector<int>> E(newSize, vector<int>(newSize));
    vector<vector<int>> F(newSize, vector<int>(newSize));
    vector<vector<int>> G(newSize, vector<int>(newSize));
    vector<vector<int>> H(newSize, vector<int>(newSize));

    vector<vector<int>> S1(newSize, vector<int>(newSize));
    vector<vector<int>> S2(newSize, vector<int>(newSize));
    vector<vector<int>> S3(newSize, vector<int>(newSize));
    vector<vector<int>> S4(newSize, vector<int>(newSize));
    vector<vector<int>> S5(newSize, vector<int>(newSize));
    vector<vector<int>> S6(newSize, vector<int>(newSize));
    vector<vector<int>> S7(newSize, vector<int>(newSize));

    vector<vector<int>> T1(newSize, vector<int>(newSize));
    vector<vector<int>> T2(newSize, vector<int>(newSize));

    // Divide matA and matB into submatrices
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A[i][j] = matA[i][j];
            B[i][j] = matA[i][j + newSize];
            C[i][j] = matA[i + newSize][j];
            D[i][j] = matA[i + newSize][j + newSize];

            E[i][j] = matB[i][j];
            F[i][j] = matB[i][j + newSize];
            G[i][j] = matB[i + newSize][j];
            H[i][j] = matB[i + newSize][j + newSize];
        }
    }

    // Perform Strassen's 7 multiplications
    // S1 = (B - D) * (G + H)
    subtract_matrix(B, D, T1, newSize);
    add_matrix(G, H, T2, newSize);
    strassen(T1, T2, S1, newSize);

    // S2 = (A + D) * (E + H)
    add_matrix(A, D, T1, newSize);
    add_matrix(E, H, T2, newSize);
    strassen(T1, T2, S2, newSize);

    // S3 = (A - C) * (E + F)
    subtract_matrix(A, C, T1, newSize);
    add_matrix(E, F, T2, newSize);
    strassen(T1, T2, S3, newSize);

    // S4 = (A + B) * H
    add_matrix(A, B, T1, newSize);
    strassen(T1, H, S4, newSize);

    // S5 = A * (F - H)
    subtract_matrix(F, H, T1, newSize);
    strassen(A, T1, S5, newSize);

    // S6 = D * (G - E)
    subtract_matrix(G, E, T1, newSize);
    strassen(D, T1, S6, newSize);

    // S7 = (C + D) * E
    add_matrix(C, D, T1, newSize);
    strassen(T1, E, S7, newSize);

    // Combine results into matRes
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            matRes[i][j] = S1[i][j] + S2[i][j] - S4[i][j] + S6[i][j];
            matRes[i][j + newSize] = S4[i][j] + S5[i][j];
            matRes[i + newSize][j] = S6[i][j] + S7[i][j];
            matRes[i + newSize][j + newSize] = S2[i][j] - S3[i][j] + S5[i][j] - S7[i][j];
        }
    }
}

int main() {
    init();
    std::cout << "===== n = " << n << " =====" << std::endl;
    float avg_time = 0.0f;
    for (int iter = 0; iter < 32; iter++) {
        auto t = get_time();
        strassen(DataA, DataB, DataC, n);
        //matmul(DataA, DataB, DataC, n);
        test();
        printf("iter %2d: %f\n", iter + 1, get_time() - t);
        avg_time += get_time() - t;
    }
    printf("Avg Time for Calculation: %f\n", avg_time / 32);
    return 0;
}
