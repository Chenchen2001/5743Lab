#include <sys/time.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

/*
 Implement a C++  version of sparse convolution and record the inference time 
 with different out channel numbers.
*/

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

const int BATCH = 1; // firmed at 1
const int HEIGHT_FEATURE = 64;
const int WIDTH_FEATURE = 4096;
const int IN_CHANNELS = 1; // firmed at 1
const int OUT_CHANNELS = 128;
const int KERNEL_SIZE = 3;
const int STRIDE = 1;
const int PADDING = 0;
const int OUTPUT_HEIGHT = (HEIGHT_FEATURE - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1;
const int OUTPUT_WIDTH = (WIDTH_FEATURE - KERNEL_SIZE + 2 * PADDING) / STRIDE + 1;
const int iterations = 32;

vector<vector<vector<vector<double>>>> cloudData(BATCH, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(HEIGHT_FEATURE, vector<double>(WIDTH_FEATURE))));
// INITIALIZE kernel by filling 0.5 
vector<vector<vector<vector<double>>>> kernel(OUT_CHANNELS, vector<vector<vector<double>>>(IN_CHANNELS, vector<vector<double>>(KERNEL_SIZE, vector<double>(KERNEL_SIZE, 0.5))));

// INITIALIZER ONLY for BATCH, IN_CHANNELS firmed at 1, read pointcloud.csv into cloudData
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
            cloudData[0][0][row][col] = round(stod(value));
            col++;
        }
        row++;
    }

    file.close(); 
}

// As dataset filled with 0 or 1, count number of none-zero items.
int count_none_zeros(const vector<vector<vector<vector<double>>>>& sparseMatrix4D) {
    size_t num_of_none_zeros = 0;
    for (size_t i = 0; i < HEIGHT_FEATURE; i++) {
        for (size_t j = 0; j < WIDTH_FEATURE; j++) {
            if (sparseMatrix4D[0][0][i][j] != 0) num_of_none_zeros++;
        }
    }
    return num_of_none_zeros;
}

// input data non-zero item coordination index storage
vector<vector<double>> generate_none_zero_list(const vector<vector<vector<vector<double>>>>& sparseMatrix4D, size_t none_zero_nums) {
    vector<vector<double>> none_zeros_list(none_zero_nums, vector<double>(2)); // index: none_zero_item_index, value: [input_h, input_w]
    size_t curr_index = 0;
    for (size_t i = 0; i < HEIGHT_FEATURE; i++) {
        for (size_t j = 0; j < WIDTH_FEATURE; j++) {
            if (cloudData[0][0][i][j] != 0) {
                size_t index = curr_index++;
                none_zeros_list[index][0] = i;
                none_zeros_list[index][1] = j;
            }
        }
    }
    return none_zeros_list;
}

// kernel data coordination index storage
vector<vector<int>> generate_kernel_coordinates() {
    vector<vector<int>> kernel_coords;

    for (size_t out_channel = 0; out_channel < OUT_CHANNELS; ++out_channel) {
        for (size_t in_channel = 0; in_channel < IN_CHANNELS; ++in_channel) {
            for (size_t kh = 0; kh < KERNEL_SIZE; ++kh) {
                for (size_t kw = 0; kw < KERNEL_SIZE; ++kw) {
                    kernel_coords.push_back({ static_cast<int>(out_channel), static_cast<int>(in_channel), static_cast<int>(kh), static_cast<int>(kw) });
                }
            }
        }
    }

    return kernel_coords;
}

// output data coordination index storage
vector<vector<int>> generate_output_coordinates(const vector<vector<double>>& none_zeros_list, int none_zero_nums) {
    vector<vector<int>> output_coords;

    for (int i = 0; i < none_zero_nums; i++) {
        int input_h = none_zeros_list[i][0];
        int input_w = none_zeros_list[i][1];

        for (int kh = 0; kh < KERNEL_SIZE; kh++) {
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                // calculate output coordinates
                int output_h = input_h - kh + PADDING;
                int output_w = input_w - kw + PADDING;
                // check output coordinates are valid
                if (output_h >= 0 && output_h < OUTPUT_HEIGHT &&
                    output_w >= 0 && output_w < OUTPUT_WIDTH) {
                    for (size_t batch = 0; batch < BATCH; ++batch) {
                        for (size_t out_channel = 0; out_channel < OUT_CHANNELS; ++out_channel) {
                            output_coords.push_back({ static_cast<int>(batch), static_cast<int>(out_channel), static_cast<int>(output_h), static_cast<int>(output_w) });
                        }
                    }
                }
            }
        }
    }

    return output_coords;
}

// rulebook construction
vector<vector<double>> generate_rulebook(const vector<vector<double>>& none_zeros_list, vector<vector<int>>& output_coordinate_list, size_t none_zero_nums) {
    vector<vector<double>> rulebook(output_coordinate_list.size(), vector<double>(3)); // [none_zero_input_coordinates_index, output_coordinates_index, kernel_coordinates_index]

    for (size_t i = 0; i < output_coordinate_list.size(); ++i) {
        size_t none_zero_input_idx = i / (KERNEL_SIZE * KERNEL_SIZE * OUT_CHANNELS);
        size_t output_idx = i; 
        size_t kernel_idx = i % (KERNEL_SIZE * KERNEL_SIZE * OUT_CHANNELS);

        rulebook[i][0] = none_zero_input_idx;
        rulebook[i][1] = output_idx;
        rulebook[i][2] = kernel_idx;
    }

    return rulebook;
}

// do multiplication
vector<vector<vector<vector<double>>>> do_sparse_conv(vector<vector<double>>& rulebook, vector<vector<double>>& none_zeros_list, vector<vector<int>>& output_coordinate_list, vector<vector<int>>& kernel_coordinate_list) {
    vector<vector<vector<vector<double>>>> output(BATCH, vector<vector<vector<double>>>(OUT_CHANNELS, vector<vector<double>>(OUTPUT_HEIGHT, vector<double>(OUTPUT_WIDTH, 0.0))));
    for (size_t i = 0; i < rulebook.size(); i++){
        size_t input_index = rulebook[i][0];
        size_t output_index = rulebook[i][1];
        size_t kernel_index = rulebook[i][2];

        size_t in_batch = 1;
        size_t in_channels = 1;
        size_t input_h = none_zeros_list[input_index][0];
        size_t input_w = none_zeros_list[input_index][1];
        double input_data = cloudData[in_batch - 1][in_channels - 1][input_h][input_w];

        size_t koutc = kernel_coordinate_list[kernel_index][0];
        size_t kinc = kernel_coordinate_list[kernel_index][1];
        size_t kh = kernel_coordinate_list[kernel_index][2];
        size_t kw = kernel_coordinate_list[kernel_index][3];
        double kernel_data = kernel[koutc][kinc][kh][kw];

        size_t out_batch = output_coordinate_list[output_index][0];
        size_t out_channels = output_coordinate_list[output_index][1];
        size_t oh = output_coordinate_list[output_index][2];
        size_t ow = output_coordinate_list[output_index][3];
        output[out_batch][out_channels][oh][ow] += input_data * kernel_data;
    }
    return output;
}

vector<vector<vector<vector<double>>>> sparse_conv(vector<vector<vector<vector<double>>>>& cloudData) {
    int num_of_none_zeros = count_none_zeros(cloudData);
    vector<vector<double>> none_zeros_list = generate_none_zero_list(cloudData, num_of_none_zeros); // index: none_zero_item_index, value: [input_h, input_w]
    vector<vector<int>> output_coordinate_list = generate_output_coordinates(none_zeros_list, num_of_none_zeros); // none_zero output item map
    vector<vector<int>> kernel_coordinate_list = generate_kernel_coordinates(); // kernel index map
    vector<vector<double>> rulebook = generate_rulebook(none_zeros_list, output_coordinate_list, num_of_none_zeros); // [none_zero_input_index, output_index, kernel_index]
    vector<vector<vector<vector<double>>>> output = do_sparse_conv(rulebook, none_zeros_list, output_coordinate_list, kernel_coordinate_list);
    return output;
}

int main() {
    string filename = "pointcloud.csv"; 
    init(filename, HEIGHT_FEATURE, WIDTH_FEATURE); 
    
    cout << endl;
    cout << "===== SPARSE CONV OUT_CHANNELS = " << OUT_CHANNELS << " =====" << endl;
    double avg_time = 0.0;
    for (int iter = 0; iter < iterations; iter++) {
        auto t = get_time();
        vector<vector<vector<vector<double>>>> output = sparse_conv(cloudData);
        cout << "Rnd:" << iter + 1 << "\tTime:" << get_time() - t << "s\tOutput_shape: [" << output.size() << ", " << output[0].size() << ", " << output[0][0].size() << ", " << output[0][0][0].size() << "]" << endl;
        avg_time += get_time() - t;
    }
    cout << "###@@@ Avg Time for Calculation(sparse_conv, out_channel = " << OUT_CHANNELS << "): " << avg_time / iterations << "s." << endl;
    cout << endl;

    return 0;
}


/*      ++=====================================++    */
/*      ||   CODES BELOW ARE FOR TEST ONLY     ||    */
/*      ++=====================================++    */

//void print2DVector(const vector<vector<double>>& vec) {
//    for (size_t i = 0; i < vec.size(); ++i) {
//        for (size_t j = 0; j < vec[i].size(); ++j) {
//            cout << vec[i][j] << " ";
//        }
//        cout << endl;
//    }
//}
//
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
//int main() {
//    string filename = "pointcloud.csv";
//    init(filename, HEIGHT_FEATURE, WIDTH_FEATURE);
//
//    cout << "========= PART OF INPUT =========" << endl;
//
//    for (size_t i = 30; i < 40; i++) {
//        for (size_t j = 25; j < 35; j++) {
//            cout << cloudData[0][0][i][j] << " ";
//        }
//        cout << endl;
//    }
//
//    return 0;
//}
/*
========= PART OF INPUT =========
   25262728293031323334
30  0 0 0 0 0 0 0 0 0 0
31  0 0 1 1 1 0 0 0 0 0
32  0 0 1 1 1 0 0 0 0 0
33  0 0 0 0 0 0 0 0 0 0
34  0 0 0 0 0 0 0 0 0 0
35  0 0 0 0 0 0 0 0 0 0
36  0 0 0 0 0 0 0 0 0 0
37  0 0 0 0 0 0 0 0 0 0
38  0 0 0 0 0 0 0 0 0 0
39  0 0 0 0 0 0 0 0 0 0
*/