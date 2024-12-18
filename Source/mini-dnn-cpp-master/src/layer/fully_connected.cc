#include "./fully_connected.h"
#include "../../config.h"
#include "cuda_utilities.h"
void FullyConnected::init() {
  weight.resize(dim_in, dim_out);
  bias.resize(dim_out);
  grad_weight.resize(dim_in, dim_out);
  grad_bias.resize(dim_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
}


void FullyConnected::forward(const Matrix& bottom) {
  // z = w' * x + b
  // const int n_sample = bottom.cols();
  // top.resize(dim_out, n_sample);
  // top = weight.transpose() * bottom;
  // top.colwise() += bias;
  switch (config::currentVersion)
  {
  case 1:
    FullyConnected::forwardVersion_1(bottom);
    break;
  case 2:
    FullyConnected::forwardVersion_2(bottom);
    break;
  case 3:
    FullyConnected::forwardVersion_3(bottom);
    break;
  
  default:
    FullyConnected::forwardVersion_1(bottom);
    break;
  }
}

// Sequential Version
void FullyConnected::forwardVersion_1(const Matrix& bottom){
  // z = w' * x + b
  const int n_sample = bottom.cols();
  top.resize(dim_out, n_sample);
  top = weight.transpose() * bottom;
  top.colwise() += bias;
}

// Parallel Version (Not optimized)
void FullyConnected::forwardVersion_2(const Matrix& bottom){
  // std::cout << "đã vào FullyConnected::forwardVersion_2\n";
  const int n_sample = bottom.cols(); // Số lượng mẫu
  top.resize(dim_out, n_sample);      // Kết quả đầu ra: kích thước (dim_out x n_sample)

  // 1. Chuẩn bị dữ liệu trên CPU và GPU
  float* h_bottom = (float*)malloc(dim_in * n_sample * sizeof(float)); // Dữ liệu đầu vào (bottom)
  float* h_C = (float*)calloc(dim_out * n_sample, sizeof(float));      // Ma trận kết quả

  for (int i = 0; i < n_sample; i++) {
    // Trích xuất cột của bottom
    float* columnData = const_cast<float*>(bottom.col(i).data());

    // Chuyển đổi cột bottom sang hàng (transpose nếu cần)
    Matrix columnMatrix = Eigen::Map<Matrix>(columnData, dim_in, 1);
    std::memcpy(h_bottom + i * dim_in, columnMatrix.data(), dim_in * sizeof(float));

    // Thực hiện phép nhân ma trận trên GPU
    matrixMultiplicationGPUWrapper(weight.transpose().data(), h_bottom + i * dim_in, h_C + i * dim_out,
                                   dim_out, dim_in, 1, i, false);

    // Cộng bias vào kết quả GPU
    Eigen::Map<Matrix>(h_C + i * dim_out, dim_out, 1).colwise() += bias;
  }

  // Chuyển kết quả từ GPU về Eigen Matrix
  Matrix result = Eigen::Map<Matrix>(h_C, dim_out, n_sample);
  top = result;

  // Giải phóng bộ nhớ
  free(h_bottom);
  free(h_C);
}
// void FullyConnected::forwardVersion_2(const Matrix& bottom){
//   // z = w' * x + b
    // std::cout << "đã vào FullyConnected::forwardVersion_2\n";
//     const int n_sample = bottom.cols(); // Số lượng mẫu
//     top.resize(dim_out, n_sample);      // Kết quả đầu ra: kích thước (dim_out x n_sample)

//     // 1. Chuẩn bị dữ liệu trên CPU và GPU
//     float* h_weight = weight.transpose().data(); // Trọng số w đã chuyển vị
//     float* h_bottom = const_cast<float*>(bottom.data()); // Dữ liệu đầu vào
//     float* h_top = (float*)calloc(dim_out * n_sample, sizeof(float)); // Ma trận kết quả

//     // 2. Sử dụng matrixMultiplicationGPUWrapper để tính z = w' * x
//     matrixMultiplicationGPUWrapper(h_weight, h_bottom, h_top,
//                                    dim_out, dim_in, n_sample, 0, false);

//     // 3. Chuyển kết quả GPU về Eigen Matrix
//     Matrix result = Eigen::Map<Matrix>(h_top, dim_out, n_sample);

//     // 4. Cộng bias vào từng cột của ma trận kết quả
//     result.colwise() += bias;

//     top = result;
//     free(h_top);
// }

// Parallel Version (optimized)
void FullyConnected::forwardVersion_3(const Matrix& bottom){
  // std::cout << "đã vào FullyConnected::forwardVersion_2\n";
  const int n_sample = bottom.cols(); // Số lượng mẫu
  top.resize(dim_out, n_sample);      // Kết quả đầu ra: kích thước (dim_out x n_sample)

  // 1. Chuẩn bị dữ liệu trên CPU và GPU
  float* h_bottom = (float*)malloc(dim_in * n_sample * sizeof(float)); // Dữ liệu đầu vào (bottom)
  float* h_C = (float*)calloc(dim_out * n_sample, sizeof(float));      // Ma trận kết quả

  for (int i = 0; i < n_sample; i++) {
    // Trích xuất cột của bottom
    float* columnData = const_cast<float*>(bottom.col(i).data());

    // Chuyển đổi cột bottom sang hàng (transpose nếu cần)
    Matrix columnMatrix = Eigen::Map<Matrix>(columnData, dim_in, 1);
    std::memcpy(h_bottom + i * dim_in, columnMatrix.data(), dim_in * sizeof(float));

    // Thực hiện phép nhân ma trận trên GPU
    matrixMultiplicationGPUWrapper(weight.transpose().data(), h_bottom + i * dim_in, h_C + i * dim_out,
                                   dim_out, dim_in, 1, i, true);

    // Cộng bias vào kết quả GPU
    Eigen::Map<Matrix>(h_C + i * dim_out, dim_out, 1).colwise() += bias;
  }

  // Chuyển kết quả từ GPU về Eigen Matrix
  Matrix result = Eigen::Map<Matrix>(h_C, dim_out, n_sample);
  top = result;

  // Giải phóng bộ nhớ
  free(h_bottom);
  free(h_C);
}
// void FullyConnected::forwardVersion_3(const Matrix& bottom){
//   // z = w' * x + b
//     const int n_sample = bottom.cols(); // Số lượng mẫu
//     top.resize(dim_out, n_sample);      // Kết quả đầu ra: kích thước (dim_out x n_sample)

//     // 1. Chuẩn bị dữ liệu trên CPU và GPU
//     float* h_weight = weight.transpose().data(); // Trọng số w đã chuyển vị
//     float* h_bottom = const_cast<float*>(bottom.data()); // Dữ liệu đầu vào
//     float* h_top = (float*)calloc(dim_out * n_sample, sizeof(float)); // Ma trận kết quả

//     // 2. Sử dụng matrixMultiplicationGPUWrapper để tính z = w' * x
//     matrixMultiplicationGPUWrapper(h_weight, h_bottom, h_top,
//                                    dim_out, dim_in, n_sample, 0, true);

//     // 3. Chuyển kết quả GPU về Eigen Matrix
//     Matrix result = Eigen::Map<Matrix>(h_top, dim_out, n_sample);

//     // 4. Cộng bias vào từng cột của ma trận kết quả
//     result.colwise() += bias;

//     top = result;
//     free(h_top);
// }

void FullyConnected::backward(const Matrix& bottom, const Matrix& grad_top) {
  switch (config::currentVersion)
  {
  case 1:
    FullyConnected::backwardVersion_1(bottom, grad_top);
    break;
  case 2:
    FullyConnected::backwardVersion_2(bottom, grad_top);
    break;
  case 3:
    FullyConnected::backwardVersion_3(bottom, grad_top);
    break;
  
  default:
    FullyConnected::backwardVersion_1(bottom, grad_top);
    break;
  }
}

// Sequential Version
void FullyConnected::backwardVersion_1(const Matrix& bottom, const Matrix& grad_top) {
  const int n_sample = bottom.cols();
  // d(L)/d(w') = d(L)/d(z) * x'
  // d(L)/d(b) = \sum{ d(L)/d(z_i) }
  grad_weight = bottom * grad_top.transpose();
  grad_bias = grad_top.rowwise().sum();
  // d(L)/d(x) = w * d(L)/d(z)
  grad_bottom.resize(dim_in, n_sample);
  grad_bottom = weight * grad_top;
}

// Parallel Version (Not optimized)
void FullyConnected::backwardVersion_2(const Matrix& bottom, const Matrix& grad_top) {
  // std::cout << "đã vào FullyConnected::backwardVersion_2\n";
  const int n_sample = bottom.cols();

  // Gradient trọng số: d(L)/d(w') = d(L)/d(z) * x'
  float* h_grad_weight = (float*)calloc(dim_in * dim_out, sizeof(float));
  float* h_bottom = (float*)malloc(dim_in * n_sample * sizeof(float)); // Dữ liệu đầu vào (x)
  float* h_grad_top = (float*)malloc(dim_out * n_sample * sizeof(float)); // Gradient đầu ra (d(L)/d(z))

  for (int i = 0; i < n_sample; i++) {
    // Trích xuất cột của bottom và grad_top
    float* columnDataBottom = const_cast<float*>(bottom.col(i).data());
    float* columnDataGradTop = const_cast<float*>(grad_top.col(i).data());

    // Chuyển đổi dữ liệu cột sang hàng
    Matrix columnMatrixBottom = Eigen::Map<Matrix>(columnDataBottom, dim_in, 1);
    Matrix columnMatrixGradTop = Eigen::Map<Matrix>(columnDataGradTop, dim_out, 1);
    std::memcpy(h_bottom + i * dim_in, columnMatrixBottom.data(), dim_in * sizeof(float));
    std::memcpy(h_grad_top + i * dim_out, columnMatrixGradTop.data(), dim_out * sizeof(float));

    // Tính gradient trọng số trên GPU
    matrixMultiplicationGPUWrapper(h_bottom + i * dim_in, h_grad_top + i * dim_out, h_grad_weight,
                                   dim_in, 1, dim_out, i, false);
  }

  grad_weight = Eigen::Map<Matrix>(h_grad_weight, dim_in, dim_out);

  // Gradient bias: d(L)/d(b) = \sum{d(L)/d(z_i)}
  grad_bias.resize(dim_out, 1);
  for (int i = 0; i < dim_out; ++i) {
    grad_bias(i, 0) = grad_top.row(i).sum();
  }

  // Gradient đầu vào: d(L)/d(x) = w * d(L)/d(z)
  float* h_grad_bottom = (float*)calloc(dim_in * n_sample, sizeof(float));
  float* h_weight = weight.data(); // Trọng số (w)

  for (int i = 0; i < n_sample; i++) {
    // Tính gradient đầu vào trên GPU
    matrixMultiplicationGPUWrapper(h_weight, h_grad_top + i * dim_out, h_grad_bottom + i * dim_in,
                                   dim_in, dim_out, 1, i, false);
  }

  grad_bottom.resize(dim_in, n_sample);
  grad_bottom = Eigen::Map<Matrix>(h_grad_bottom, dim_in, n_sample);

  // 2. Giải phóng bộ nhớ
  free(h_grad_weight);
  free(h_grad_bottom);
  free(h_bottom);
  free(h_grad_top);
}
// void FullyConnected::backwardVersion_2(const Matrix& bottom, const Matrix& grad_top) {
    // std::cout << "đã vào FullyConnected::backwardVersion_2\n";
//     const int n_sample = bottom.cols();

//     // 1. Chuẩn bị dữ liệu GPU
//     float* h_bottom = const_cast<float*>(bottom.data()); // Dữ liệu đầu vào (x)
//     float* h_grad_top = const_cast<float*>(grad_top.data()); // Gradient đầu ra (d(L)/d(z))

//     // Gradient trọng số: d(L)/d(w') = d(L)/d(z) * x'
//     float* h_grad_weight = (float*)calloc(dim_in * dim_out, sizeof(float));
//     matrixMultiplicationGPUWrapper(h_bottom, h_grad_top, h_grad_weight, 
//                                    dim_in, n_sample, dim_out, 0, false);

//     // Cập nhật grad_weight từ GPU về Eigen::Matrix
//     grad_weight = Eigen::Map<Matrix>(h_grad_weight, dim_in, dim_out);

//     // Gradient bias: d(L)/d(b) = \sum{d(L)/d(z_i)}
//     grad_bias.resize(dim_out, 1);
//     for (int i = 0; i < dim_out; ++i) {
//         grad_bias(i, 0) = grad_top.row(i).sum();
//     }

//     // Gradient đầu vào: d(L)/d(x) = w * d(L)/d(z)
//     float* h_grad_bottom = (float*)calloc(dim_in * n_sample, sizeof(float));
//     float* h_weight = weight.data(); // Trọng số (w)
//     matrixMultiplicationGPUWrapper(h_weight, h_grad_top, h_grad_bottom, 
//                                    dim_in, dim_out, n_sample, 0, false);

//     // Cập nhật grad_bottom từ GPU về Eigen::Matrix
//     grad_bottom.resize(dim_in, n_sample);
//     grad_bottom = Eigen::Map<Matrix>(h_grad_bottom, dim_in, n_sample);

//     // 2. Giải phóng bộ nhớ
//     free(h_grad_weight);
//     free(h_grad_bottom);
// }

// Parallel Version (optimized)
// Parallel Version (Not optimized)
void FullyConnected::backwardVersion_3(const Matrix& bottom, const Matrix& grad_top) {
  // std::cout << "đã vào FullyConnected::backwardVersion_2\n";
  const int n_sample = bottom.cols();

  // Gradient trọng số: d(L)/d(w') = d(L)/d(z) * x'
  float* h_grad_weight = (float*)calloc(dim_in * dim_out, sizeof(float));
  float* h_bottom = (float*)malloc(dim_in * n_sample * sizeof(float)); // Dữ liệu đầu vào (x)
  float* h_grad_top = (float*)malloc(dim_out * n_sample * sizeof(float)); // Gradient đầu ra (d(L)/d(z))

  for (int i = 0; i < n_sample; i++) {
    // Trích xuất cột của bottom và grad_top
    float* columnDataBottom = const_cast<float*>(bottom.col(i).data());
    float* columnDataGradTop = const_cast<float*>(grad_top.col(i).data());

    // Chuyển đổi dữ liệu cột sang hàng
    Matrix columnMatrixBottom = Eigen::Map<Matrix>(columnDataBottom, dim_in, 1);
    Matrix columnMatrixGradTop = Eigen::Map<Matrix>(columnDataGradTop, dim_out, 1);
    std::memcpy(h_bottom + i * dim_in, columnMatrixBottom.data(), dim_in * sizeof(float));
    std::memcpy(h_grad_top + i * dim_out, columnMatrixGradTop.data(), dim_out * sizeof(float));

    // Tính gradient trọng số trên GPU
    matrixMultiplicationGPUWrapper(h_bottom + i * dim_in, h_grad_top + i * dim_out, h_grad_weight,
                                   dim_in, 1, dim_out, i, true);
  }

  grad_weight = Eigen::Map<Matrix>(h_grad_weight, dim_in, dim_out);

  // Gradient bias: d(L)/d(b) = \sum{d(L)/d(z_i)}
  grad_bias.resize(dim_out, 1);
  for (int i = 0; i < dim_out; ++i) {
    grad_bias(i, 0) = grad_top.row(i).sum();
  }

  // Gradient đầu vào: d(L)/d(x) = w * d(L)/d(z)
  float* h_grad_bottom = (float*)calloc(dim_in * n_sample, sizeof(float));
  float* h_weight = weight.data(); // Trọng số (w)

  for (int i = 0; i < n_sample; i++) {
    // Tính gradient đầu vào trên GPU
    matrixMultiplicationGPUWrapper(h_weight, h_grad_top + i * dim_out, h_grad_bottom + i * dim_in,
                                   dim_in, dim_out, 1, i, true);
  }

  grad_bottom.resize(dim_in, n_sample);
  grad_bottom = Eigen::Map<Matrix>(h_grad_bottom, dim_in, n_sample);

  // 2. Giải phóng bộ nhớ
  free(h_grad_weight);
  free(h_grad_bottom);
  free(h_bottom);
  free(h_grad_top);
}
// void FullyConnected::backwardVersion_3(const Matrix& bottom, const Matrix& grad_top) {
//     const int n_sample = bottom.cols();

//     // 1. Chuẩn bị dữ liệu GPU
//     float* h_bottom = const_cast<float*>(bottom.data()); // Dữ liệu đầu vào (x)
//     float* h_grad_top = const_cast<float*>(grad_top.data()); // Gradient đầu ra (d(L)/d(z))

//     // Gradient trọng số: d(L)/d(w') = d(L)/d(z) * x'
//     float* h_grad_weight = (float*)calloc(dim_in * dim_out, sizeof(float));
//     matrixMultiplicationGPUWrapper(h_bottom, h_grad_top, h_grad_weight, 
//                                    dim_in, n_sample, dim_out, 0, true);

//     // Cập nhật grad_weight từ GPU về Eigen::Matrix
//     grad_weight = Eigen::Map<Matrix>(h_grad_weight, dim_in, dim_out);

//     // Gradient bias: d(L)/d(b) = \sum{d(L)/d(z_i)}
//     grad_bias.resize(dim_out, 1);
//     for (int i = 0; i < dim_out; ++i) {
//         grad_bias(i, 0) = grad_top.row(i).sum();
//     }

//     // Gradient đầu vào: d(L)/d(x) = w * d(L)/d(z)
//     float* h_grad_bottom = (float*)calloc(dim_in * n_sample, sizeof(float));
//     float* h_weight = weight.data(); // Trọng số (w)
//     matrixMultiplicationGPUWrapper(h_weight, h_grad_top, h_grad_bottom, 
//                                    dim_in, dim_out, n_sample, 0, true);

//     // Cập nhật grad_bottom từ GPU về Eigen::Matrix
//     grad_bottom.resize(dim_in, n_sample);
//     grad_bottom = Eigen::Map<Matrix>(h_grad_bottom, dim_in, n_sample);

//     // 2. Giải phóng bộ nhớ
//     free(h_grad_weight);
//     free(h_grad_bottom);
// }



void FullyConnected::update(Optimizer& opt) {
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(),
                                              grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> FullyConnected::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(),
            res.begin() + weight.size());
  return res;
}

void FullyConnected::set_parameters(const std::vector<float>& param) {
  if (static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> FullyConnected::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(),
            res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
