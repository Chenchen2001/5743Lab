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

// DEFINE conv function
vector<vector<vector<vector<double>>>> conv2d(
    const vector<vector<vector<vector<double>>>>& input,
    const vector<vector<vector<vector<double>>>>& kernel,
    int STRIDE, int PADDING) {

    // GET input AND kernal size and CALCULATE output size
    int batch = input.size();
    int in_channels = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();
    int out_channels = kernel.size();
    int kernel_size = kernel[0][0].size();
    int out_height = (height - kernel_size + 2 * PADDING) / STRIDE + 1;
    int out_width = (width - kernel_size + 2 * PADDING) / STRIDE + 1;

    // INITIALIZE output
    vector<vector<vector<vector<double>>>> output(batch, vector<vector<vector<double>>>(out_channels, vector<vector<double>>(out_height, vector<double>(out_width, 0))));

    for (int b = 0; b < batch; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int h_offset = oh * STRIDE + kh - PADDING;
                                int w_offset = ow * STRIDE + kw - PADDING;

                                if (h_offset >= 0 && h_offset < height && w_offset >= 0 && w_offset < width) {
                                    output[b][oc][oh][ow] +=
                                        input[b][ic][h_offset][w_offset] * kernel[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}

int main() {
    string filename = "pointcloud.csv";
    init(filename, HEIGHT, WIDTH);

    cout << endl;
    cout << "===== TRADITIONAL CONV OUT_CHANNELS = " << OUT_CHANNELS << " =====" << endl;
    double avg_time = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        auto t = get_time();
        // RUN conv
        vector<vector<vector<vector<double>>>> output = conv2d(input, kernel, STRIDE, PADDING);
        cout << "Rnd:" << iter+1 << "\tTime:" << get_time() - t << "s\tOutput_shape: [" << output.size() << ", " << output[0].size() << ", " << output[0][0].size() << ", " << output[0][0][0].size() << "]" << endl;
        avg_time += get_time() - t;
    }
    cout << "###@@@ Avg Time for Calculation(traditional conv out_channel = " << OUT_CHANNELS << "): " << avg_time / iterations << "s." << endl;
    cout << endl;

    return 0;
}