#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cassert>

using namespace std;

// INITIALIZE paras
size_t BATCH = 1;
size_t HEIGHT = 64;
size_t WIDTH = 4096;
size_t IN_CHANNELS = 1;
size_t OUT_CHANNELS = 1024;
size_t KERNEL_SIZE = 3;
size_t STRIDE = 1;
size_t PADDING = 0;
int iterations = 32;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

vector<vector<vector<vector<double>>>> input(BATCH, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(HEIGHT, vector<double>(WIDTH))));
// INITIALIZE kernel by filling 0.5 
vector<vector<vector<vector<double>>>> kernel(OUT_CHANNELS, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(KERNEL_SIZE, vector<double>(KERNEL_SIZE, 0.5))));

void init(const string& filename, size_t rows, size_t cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    string line;
    size_t row = 0;

    while (getline(file, line)) { // split the data by line
        if (row >= rows) break;
        stringstream ss(line);
        string value;
        size_t col = 0;

        while (getline(ss, value, ',')) { // split each line by ","
            if (col >= cols) break;
            input[0][0][row][col] = round(stod(value));
            col++;
        }
        row++;
    }

    file.close();
}

// Convert the feature map to column matrix 
vector<vector<double>> im2col(
    const vector<vector<vector<vector<double>>>>& input,
    int KERNEL_SIZE, int STRIDE, int PADDING) {

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

// EXECUTE im2col multiplication with kernel matrix
vector<vector<double>> im2col_multi_with_kernel_matrix(
    const vector<vector<double>>& im2col_matrix,
    const vector<vector<double>>& kernel_matrix) {

    int rows = im2col_matrix.size();
    int cols = kernel_matrix.size();
    int shared_dim = kernel_matrix[0].size(); 

    // Initialize result matrix
    vector<vector<double>> result(rows, vector<double>(cols, 0));

    // Matrix multiplication
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < shared_dim; ++k) {
                result[i][j] += im2col_matrix[i][k] * kernel_matrix[j][k];
            }
        }
    }
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
vector<vector<vector<vector<double>>>> conv2d_im2col(
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
    vector<vector<double>> result = im2col_multi_with_kernel_matrix(im2col_matrix, kernel_matrix);
    return format_col2output(result, BATCH, OUT_CHANNELS, out_HEIGHT, out_WIDTH);
}


int main() {
    string filename = "pointcloud.csv";
    init(filename, HEIGHT, WIDTH);

    cout << endl;
    cout << "===== im2col CONV OUT_CHANNELS = " << OUT_CHANNELS << " =====" << endl;
    double avg_time = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        auto t = get_time();
        // RUN conv
        vector<vector<vector<vector<double>>>> output = conv2d_im2col(input, kernel, STRIDE, PADDING);
        cout << "Rnd:" << iter+1 << "\tTime:" << get_time() - t << "s\tOutput_shape: [" << output.size() << ", " << output[0].size() << ", " << output[0][0].size() << ", " << output[0][0][0].size() << "]" << endl;
        avg_time += get_time() - t;
    }
    cout << "###@@@ Avg Time for Calculation(im2col_conv, out_channel = " << OUT_CHANNELS << "): " << avg_time / iterations << "s." << endl;
    cout << endl;
    
    return 0;
}