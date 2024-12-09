#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace std;

// INITIALIZE paras
size_t BATCH = 1;
size_t HEIGHT = 56;
size_t WIDTH = 56;
size_t IN_CHANNELS = 3;
size_t OUT_CHANNELS = 64;
size_t KERNEL_SIZE = 3;
size_t STRIDE = 1;
size_t PADDING = 0;
int iterations = 32;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// Convert the feature map to column matrix 
vector<vector<double>> im2col(const vector<vector<vector<vector<double>>>>& input, int KERNEL_SIZE, int STRIDE, int PADDING) {
    // GET input AND output size and CALCULATE output size
    int batch = input.size();
    int in_channels = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();
    int out_height = (height - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1;
    int out_width = (width - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1;

    // INITIALIZE im2col matrix
    int col_height = out_height * out_width;
    int col_width = in_channels * KERNEL_SIZE * KERNEL_SIZE;
    vector<vector<double>> im2col_matrix(batch * col_height, vector<double>(col_width, 0));

    // CALCULATE im2col mattrix
    for (int b = 0; b < batch; ++b) {
        int col_idx = 0;
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                int row_idx = 0;
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                        for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                            int h_offset = h * STRIDE + kh - PADDING;
                            int w_offset = w * STRIDE + kw - PADDING;
                            if (h_offset >= 0 && h_offset < height && w_offset >= 0 && w_offset < width) {
                                im2col_matrix[b * col_height + col_idx][row_idx] = input[b][ic][h_offset][w_offset];
                            }
                            ++row_idx;
                        }
                    }
                }
                ++col_idx;
            }
        }
    }
    return im2col_matrix;
}

// CONVERT kernel to matrix
vector<vector<double>> kernel2matrix(const vector<vector<vector<vector<double>>>>& kernel) {
    int OUT_CHANNELS = kernel.size();
    int IN_CHANNELS = kernel[0].size();
    int KERNEL_SIZE = kernel[0][0].size();
    vector<vector<double>> kernel_matrix(OUT_CHANNELS, vector<double>(IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE, 0));

    for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
        int col_idx = 0;
        for (int ic = 0; ic < IN_CHANNELS; ++ic) {
            for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    kernel_matrix[oc][col_idx] = kernel[oc][ic][kh][kw];
                    ++col_idx;
                }
            }
        }
    }
    return kernel_matrix;
}

vector<vector<double>> add_matrix(const vector<vector<double>>& DataA, const vector<vector<double>>& DataB) {
    int AWidth = DataA[0].size();
    int AHeight = DataA.size();
    int BWidth = DataB[0].size();
    int BHeight = DataB.size();

    assert(AWidth == BWidth);
    assert(AHeight == BHeight);

    vector<vector<double>> result(AHeight, vector<double>(AWidth));
    
    for (int i = 0; i < AHeight; i++) {
        for (int j = 0; j < AWidth; j++) {
            result[i][j] = DataA[i][j] + DataB[i][j];
        }
    }
    return result;
}

vector<vector<double>> subtract_matrix(const vector<vector<double>>& DataA, const vector<vector<double>>& DataB) {
    int AWidth = DataA[0].size();
    int AHeight = DataA.size();
    int BWidth = DataB[0].size();
    int BHeight = DataB.size();

    assert(AWidth == BWidth);
    assert(AHeight == BHeight);

    vector<vector<double>> result(AHeight, vector<double>(AWidth));

    for (int i = 0; i < AHeight; i++) {
        for (int j = 0; j < AWidth; j++) {
            result[i][j] = DataA[i][j] - DataB[i][j];
        }
    }
    return result;
}

vector<vector<double>> multiply_matrix(const vector<vector<double>>& DataA, const vector<vector<double>>& DataB) {
    int AWidth = DataA[0].size();
    int AHeight = DataA.size();
    int BWidth = DataB[0].size();
    int BHeight = DataB.size();

    assert(AWidth == BHeight);

    vector<vector<double>> result(AHeight, vector<double>(BWidth));

    for (int i = 0; i < AHeight; i++) {
        for (int j = 0; j < BWidth; j++) {
            double tmp = 0;
            for (int k = 0; k < AWidth; k++) {
                tmp += DataA[i][k] * DataB[k][j];
            }
            result[i][j] = tmp;
        }
    }
    return result;
}

vector<vector<double>> scale_matrix(const vector<vector<double>>& DataA, double scalar) {
    int AWidth = DataA[0].size();
    int AHeight = DataA.size();
    vector<vector<double>> result(AHeight, vector<double>(AWidth)); 
    for (size_t i = 0; i < DataA.size(); i++) { 
        for (size_t j = 0; j < DataA[0].size(); j++) { 
            result[i][j] = DataA[i][j] * scalar;
        }
    }
    return result;
}

void winograd(const vector<vector<double>>& matA, const vector<vector<double>>& matB, vector<vector<double>>& matRes) {

    int newRows = matA.size() / 2;
    int newCols = matA[0].size() / 3;
    int newKRows = matB[0].size() / 3;
    int newKCols = matB.size();

    vector<vector<double>> D00(newRows, vector<double>(newCols));
    vector<vector<double>> D10(newRows, vector<double>(newCols));
    vector<vector<double>> D20(newRows, vector<double>(newCols));
    vector<vector<double>> D30(newRows, vector<double>(newCols));

    vector<vector<double>> K0(newKRows, vector<double>(newKCols));
    vector<vector<double>> K1(newKRows, vector<double>(newKCols));
    vector<vector<double>> K2(newKRows, vector<double>(newKCols));

    vector<vector<double>> M0(newRows, vector<double>(newKCols));
    vector<vector<double>> M1(newRows, vector<double>(newKCols));
    vector<vector<double>> M2(newRows, vector<double>(newKCols));
    vector<vector<double>> M3(newRows, vector<double>(newKCols));

    vector<vector<double>> R0(newRows, vector<double>(newKCols));
    vector<vector<double>> R1(newRows, vector<double>(newKCols));
    
    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < newCols; j++) {
            D00[i][j] = matA[i][j];
            D10[i][j] = matA[i][j + newCols];
            D20[i][j] = matA[i][j + 2 * newCols];
            D30[i][j] = matA[i + newRows][j + 2 * newCols];
        }
    }

    for (int i = 0; i < newKRows; i++) {
        for (int j = 0; j < newKCols; j++) {    
            K0[i][j] = matB[j][i];
            K1[i][j] = matB[j][i + newKRows];
            K2[i][j] = matB[j][i + 2 * newKRows];
        }
    }

    M0 = multiply_matrix(subtract_matrix(D00, D20), K0);
    M1 = multiply_matrix(add_matrix(D10, D20), scale_matrix(add_matrix(add_matrix(K0, K1), K2), 0.5));
    M2 = multiply_matrix(subtract_matrix(D20, D10), scale_matrix(add_matrix(subtract_matrix(K0, K1), K2), 0.5));
    M3 = multiply_matrix(subtract_matrix(D10, D30), K2);

    R0 = add_matrix(add_matrix(M0, M1), M2);
    R1 = subtract_matrix(subtract_matrix(M1, M2), M3);

    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < newKCols; j++) {
            matRes[i][j] = R0[i][j];
            matRes[i + newKRows][j] = R1[i][j];
        }
    }
}

// EXECUTE im2col multiplication with kernel matrix by winograd
vector<vector<double>> im2col_multi_with_kernel_matrix_for_winograd(const vector<vector<double>>& im2col_matrix, const vector<vector<double>>& kernel_matrix) {
    int rows = im2col_matrix.size();
    int cols = kernel_matrix.size();
    vector<vector<double>> result(rows, vector<double>(cols, 0));
    winograd(im2col_matrix, kernel_matrix, result);
    return result;
}


// RESHAPE result to output format
vector<vector<vector<vector<double>>>> format_col2output(
    const vector<vector<double>>& result, int BATCH, int OUT_CHANNELS, int out_HEIGHT, int out_WIDTH) {
    vector<vector<vector<vector<double>>>> output(BATCH, vector<vector<vector<double>>>(OUT_CHANNELS, vector<vector<double>>(out_HEIGHT, vector<double>(out_WIDTH, 0))));

    for (int b = 0; b < BATCH; ++b) {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
            for (int oh = 0; oh < out_HEIGHT; ++oh) {
                for (int ow = 0; ow < out_WIDTH; ++ow) {
                    output[b][oc][oh][ow] = result[b * (out_HEIGHT * out_WIDTH) + oh * out_WIDTH + ow][oc];
                }
            }
        }
    }
    return output;
}

// EXECUTE Conv2D using im2col
vector<vector<vector<vector<double>>>> conv2d_winograd(
    const vector<vector<vector<vector<double>>>>& input,
    const vector<vector<vector<vector<double>>>>& kernel,
    int STRIDE, int PADDING) {

    int BATCH = input.size();
    int OUT_CHANNELS = kernel.size();
    int KERNEL_SIZE = kernel[0][0].size();
    int HEIGHT = input[0][0].size();
    int WIDTH = input[0][0][0].size();
    int out_HEIGHT = (HEIGHT - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1;
    int out_WIDTH = (WIDTH - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1;

    vector<vector<double>> im2col_matrix = im2col(input, KERNEL_SIZE, STRIDE, PADDING);
    vector<vector<double>> kernel_matrix = kernel2matrix(kernel);
    vector<vector<double>> result = im2col_multi_with_kernel_matrix_for_winograd(im2col_matrix, kernel_matrix);
    return format_col2output(result, BATCH, OUT_CHANNELS, out_HEIGHT, out_WIDTH);
}


int main() {
    // INPUT SIZE is [BATCH, IN_CHANNELS, HEIGHT, WIDTH]
    vector<vector<vector<vector<double>>>> input(BATCH, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(HEIGHT, vector<double>(WIDTH))));
    for (size_t b = 0; b < BATCH; ++b) {
        for (size_t c = 0; c < IN_CHANNELS; ++c) {
            for (size_t h = 0; h < HEIGHT; ++h) {
                for (size_t w = 0; w < WIDTH; ++w) {
                    input[b][c][h][w] = rand() % 256;
                }
            }
        }
    }

    // KERNAL SIZE is [OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE]
    // INITIALIZE kernel by filling 0.5 
    vector<vector<vector<vector<double>>>> kernel(OUT_CHANNELS, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(KERNEL_SIZE, vector<double>(KERNEL_SIZE, 0.5))));

    double avg_time = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        auto t = get_time();
        // RUN conv
        vector<vector<vector<vector<double>>>> output = conv2d_winograd(input, kernel, STRIDE, PADDING);
        cout << "Rnd:" << iter + 1 << "\tTime:" << get_time() - t << "s\tOutput_shape: [" << output.size() << ", " << output[0].size() << ", " << output[0][0].size() << ", " << output[0][0][0].size() << "]" << endl;
        avg_time += get_time() - t;
    }
    cout << "Avg Time for Calculation: " << avg_time / iterations << "s." << endl;
    return 0;
}