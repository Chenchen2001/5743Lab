#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <cassert>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}
// INITIALIZE the matrix size, matrix and parameters
constexpr int I = 1024;
constexpr int K = 1024;
constexpr int J = 1024;

int A[I][K];
int B[K][J];
int AT[K][I];
int BT[J][K];
int C[I][J];
int CTruth[I][J];

void initA() {
    for (int i = 0; i < I; i++) {
        for (int k = 0; k < K; k++) {
            A[i][k] = rand();
        }
    }
}

void initB() {
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < J; j++) {
            B[k][j] = rand();
        }
    }
}

void initCTruth() {
    for (int i = 0; i < I; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++) {
                CTruth[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void initAT() {
    for (int i = 0; i < K; i++) {
        for (int k = 0; k < I; k++) {
            AT[i][k] = A[k][i];
        }
    }
}

void initBT() {
    for (int k = 0; k < J; k++) {
        for (int j = 0; j < K; j++) {
            BT[k][j] = B[j][k];
        }
    }
}

// INITIALIZE the matrixes
void init() {
    initA();
    initB();
    initAT();
    initBT();
    initCTruth();
}

void test() {
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      assert(C[i][j] == CTruth[i][j]);
    }
  }
}

void matmul() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_ikj() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < J; j++) {
        C[i][j] += A[i][k] * B[k][j];    
      }   
    }
  }
}

void matmul_AT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += AT[k][i] * B[k][j];    
      }   
    }
  }
}

void matmul_BT() {
  memset(C, 0, sizeof(C));
  for (int i = 0; i < I; i++) {
    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * BT[j][k];    
      }   
    }
  }
}

int main() {
  init();
  std::cout << "===== I = " << I << "\t K = " << K << "\t J = " << J << " =====" << std::endl;
  float avg_time = 0.0f;
  for (int iter = 0; iter < 32; iter++) {
    auto t = get_time();
    matmul();
     //matmul_ikj();
     //matmul_AT();
     //matmul_BT();
    test();
    printf("%f\n", get_time() - t);
    avg_time += get_time() - t;
  }
  printf("Avg Time for Calculation: %f\n", avg_time / 32);
  return 0;
}