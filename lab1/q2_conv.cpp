#include <sys/time.h>
#include <iostream>
#include <vector>
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
    // INPUT SIZE is [BATCH, IN_CHANNELS, HEIGHT, WIDTH]
    vector<vector<vector<vector<double>>>> input(BATCH, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(HEIGHT, vector<double>(WIDTH))));
    for (size_t b = 0; b < BATCH; ++b) { // INITIALIZE input feature map with random number from 0 to 255
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
        vector<vector<vector<vector<double>>>> output = conv2d(input, kernel, STRIDE, PADDING);
        cout << "Rnd:" << iter+1 << "\tTime:" << get_time() - t << "s\tOutput_shape: [" << output.size() << ", " << output[0].size() << ", " << output[0][0].size() << ", " << output[0][0][0].size() << "]" << endl;
        avg_time += get_time() - t;
    }
    cout << "Avg Time for Calculation: " << avg_time / iterations << "s."<< endl;
    return 0;
}



/*      ++=====================================++    */
/*      ||   CODES BELOW ARE FOR TEST ONLY     ||    */
/*      ++=====================================++    */

//void print4DVector(const vector<vector<vector<vector<double>>>>& vec) {
//    for (size_t i = 0; i < vec.size(); ++i) {
//        cout << "Batch " << i << ":" << endl;
//        for (size_t j = 0; j < vec[i].size(); ++j) {
//            cout << "  Channel " << j << ":" << endl;
//            for (size_t k = 0; k < vec[i][j].size(); ++k) {
//                for (size_t l = 0; l < vec[i][j][k].size(); ++l) {
//                    cout << vec[i][j][k][l] << " ";
//                }
//                cout << endl;
//            }
//            cout << endl;
//        }
//        cout << endl;
//    }
//}
//
//void test_conv2d() {
//    int BATCH = 1;
//    int IN_CHANNELS = 1;
//    int OUT_CHANNELS = 1;
//    int HEIGHT = 4;
//    int WIDTH = 4;
//    int KERNEL_SIZE = 3;
//    int STRIDE = 1;
//    int PADDING = 0;
//
//    // INPUT SIZE IS [1, 1, 4, 4]£¬ALL FILL IN 1
//    vector<vector<vector<vector<double>>>> input(BATCH, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(HEIGHT, vector<double>(WIDTH, 1.0))));
//
//    // KERNAL SIZE IS [1, 1, 3, 3]£¬ALL FILL IN 0.5
//    vector<vector<vector<vector<double>>>> kernel(OUT_CHANNELS, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(KERNEL_SIZE, vector<double>(KERNEL_SIZE, 0.5))));
//
//    // DO CONV
//    vector<vector<vector<vector<double>>>> output = conv2d(input, kernel, STRIDE, PADDING);
//
//    // CHECK OUTPUT SHAPE
//    assert(output.size() == static_cast<size_t>(BATCH));
//    assert(output[0].size() == static_cast<size_t>(OUT_CHANNELS));
//    assert(output[0][0].size() == static_cast<size_t>((HEIGHT - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1));
//    assert(output[0][0][0].size() == static_cast<size_t>((WIDTH - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1));
//
//    cout << "Test output shape passed!" << endl;
// 
//    cout << "---------- input: ----------" << endl;
//    print4DVector(input);
//    cout << "----------kernel: ----------" << endl;
//    print4DVector(kernel);
//    cout << "---------- output: ----------" << endl;
//    print4DVector(output);
//
//    assert(output[0][0][1][1] == 4.5);
//    cout << "Test output values passed!" << endl;
//}
//
//int main() {
//    cout << "========== NOW DOING TEST ==========\n" << endl;
//    test_conv2d();
//    cout << "============ TEST  DONE ============" << endl;
//    return 0;
//}
