#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

constexpr int n = 1024;
constexpr int TILE_SIZE = 256;

int A[n][n];
int B[n][n];
int C[n][n];
int C_groundtruth[n][n];

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void init() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = rand();
            B[i][j] = rand();
        }
    }
    memset(C_groundtruth, 0, sizeof(C_groundtruth));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C_groundtruth[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void test() {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            assert(C[i][j] == C_groundtruth[i][j]);
        }
    }
}

void matmul() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_ikj() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matmul_unroll() {
    memset(C, 0, sizeof(C));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k += 8) {  // Unroll by 1 to 32
                C[i][j] += A[i][k] * B[k][j];
                C[i][j] += A[i][k + 1] * B[k + 1][j];
                C[i][j] += A[i][k + 2] * B[k + 2][j];
                C[i][j] += A[i][k + 3] * B[k + 3][j];
                C[i][j] += A[i][k + 4] * B[k + 4][j];
                C[i][j] += A[i][k + 5] * B[k + 5][j];
                C[i][j] += A[i][k + 6] * B[k + 6][j];
                C[i][j] += A[i][k + 7] * B[k + 7][j];
                //C[i][j] += A[i][k + 8] * B[k + 8][j];
                //C[i][j] += A[i][k + 9] * B[k + 9][j];
                //C[i][j] += A[i][k + 10] * B[k + 10][j];
                //C[i][j] += A[i][k + 11] * B[k + 11][j];
                //C[i][j] += A[i][k + 12] * B[k + 12][j];
                //C[i][j] += A[i][k + 13] * B[k + 13][j];
                //C[i][j] += A[i][k + 14] * B[k + 14][j];
                //C[i][j] += A[i][k + 15] * B[k + 15][j];
                //C[i][j] += A[i][k + 16] * B[k + 16][j];
                //C[i][j] += A[i][k + 17] * B[k + 17][j];
                //C[i][j] += A[i][k + 18] * B[k + 18][j];
                //C[i][j] += A[i][k + 19] * B[k + 19][j];
                //C[i][j] += A[i][k + 20] * B[k + 20][j];
                //C[i][j] += A[i][k + 21] * B[k + 21][j];
                //C[i][j] += A[i][k + 22] * B[k + 22][j];
                //C[i][j] += A[i][k + 23] * B[k + 23][j];
                //C[i][j] += A[i][k + 24] * B[k + 24][j];
                //C[i][j] += A[i][k + 25] * B[k + 25][j];
                //C[i][j] += A[i][k + 26] * B[k + 26][j];
                //C[i][j] += A[i][k + 27] * B[k + 27][j];
                //C[i][j] += A[i][k + 28] * B[k + 28][j];
                //C[i][j] += A[i][k + 29] * B[k + 29][j];
                //C[i][j] += A[i][k + 30] * B[k + 30][j];
                //C[i][j] += A[i][k + 31] * B[k + 31][j];
            }
        }
    }
}

void matmul_tile() {
    memset(C, 0, sizeof(C));
    for (int ii = 0; ii < n; ii += TILE_SIZE) {
        for (int jj = 0; jj < n; jj += TILE_SIZE) {
            for (int kk = 0; kk < n; kk += TILE_SIZE) {
                for (int i = ii; i < ii + TILE_SIZE; i++) {
                    for (int j = jj; j < jj + TILE_SIZE; j++) {
                        for (int k = kk; k < kk + TILE_SIZE; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    init();
    float avg_time = 0.0f;
    for (int K = 0; K < 32; K++) {
        auto t = get_time();

        //matmul();
        //matmul_ikj();
        //matmul_unroll();
        matmul_tile();

        test();
        printf("Iteration Time: %f\n", get_time() - t);
        avg_time += get_time() - t;
    }
    printf("Avg Time for Calculation: %f\n", avg_time / 32);
    return 0;
}
