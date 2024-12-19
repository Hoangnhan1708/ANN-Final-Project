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
    Matrix columnMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(columnData, dim_in, 1);
    std::memcpy(h_bottom + i * dim_in, columnMatrix.data(), dim_in * sizeof(float));

    // Thực hiện phép nhân ma trận trên GPU
    matrixMultiplicationGPUWrapper(weight.transpose().data(), h_bottom + i * dim_in, h_C + i * dim_out,
                                   dim_out, dim_in, 1, i, false);

    // Cộng bias vào kết quả GPU
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_C + i * dim_out, dim_out, 1).colwise() += bias;
  }

  // Chuyển kết quả từ GPU về Eigen Matrix
  Matrix result = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_C, dim_out, n_sample);
  top = result;

  // Matrix top_sequential = weight.transpose() * bottom;
  // top_sequential.colwise() += bias;

  // So sánh top (GPU) với top_sequential (CPU)
  // float error = (top - top_sequential).norm();
  // std::cout << "Forward Version 2: Top ERROR: " << error << std::endl;

  // Giải phóng bộ nhớ
  free(h_bottom);
  free(h_C);
}


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
    Matrix columnMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(columnData, dim_in, 1);
    std::memcpy(h_bottom + i * dim_in, columnMatrix.data(), dim_in * sizeof(float));

    // Thực hiện phép nhân ma trận trên GPU
    matrixMultiplicationGPUWrapper(weight.transpose().data(), h_bottom + i * dim_in, h_C + i * dim_out,
                                   dim_out, dim_in, 1, i, true);

    // Cộng bias vào kết quả GPU
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_C + i * dim_out, dim_out, 1).colwise() += bias;
  }

  // Chuyển kết quả từ GPU về Eigen Matrix
  Matrix result = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_C, dim_out, n_sample);
  top = result;

  // Matrix top_sequential = weight.transpose() * bottom;
  // top_sequential.colwise() += bias;

  // So sánh top (GPU) với top_sequential (CPU)
  // float error = (top - top_sequential).norm();
  // std::cout << "Forward Version 2: Top ERROR: " << error << std::endl;

  // Giải phóng bộ nhớ
  free(h_bottom);
  free(h_C);
}

void FullyConnected::backward(const Matrix& bottom, const Matrix& grad_top) {
  // FullyConnected::backwardVersion_1(bottom, grad_top);
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
    const int n_sample = bottom.cols();

    // Chuẩn bị bộ nhớ
    float* h_bottom = (float*)malloc(dim_in * n_sample * sizeof(float));
    float* h_grad_top = (float*)malloc(dim_out * n_sample * sizeof(float));
    float* h_grad_bottom = (float*)malloc(dim_in * n_sample * sizeof(float));

    // Sao chép dữ liệu từ Eigen sang mảng liên tục
    std::memcpy(h_bottom, bottom.data(), dim_in * n_sample * sizeof(float));
    std::memcpy(h_grad_top, grad_top.data(), dim_out * n_sample * sizeof(float));

    // 1. Tính grad_weight bằng code tuần tự
    grad_weight = bottom * grad_top.transpose();

    // 2. Tính grad_bias = \sum(d(L)/d(z))
    grad_bias.resize(dim_out, 1);
    for (int i = 0; i < dim_out; ++i) {
        grad_bias(i, 0) = grad_top.row(i).sum();
    }

    // 3. Tính grad_bottom = weight * grad_top (sử dụng song song trên GPU)
    matrixMultiplicationGPUWrapper(weight.data(), h_grad_top, h_grad_bottom, dim_in, dim_out, n_sample, 0, false);

    // Chuyển kết quả từ GPU về Eigen Matrix
    grad_bottom = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_grad_bottom, dim_in, n_sample);

    // Debug giá trị
    // Matrix grad_weight_sequential = bottom * grad_top.transpose();
    // Matrix grad_bias_sequential = grad_top.rowwise().sum();
    // Matrix grad_bottom_sequential = weight * grad_top;

    // float error_weight = (grad_weight - grad_weight_sequential).norm();
    // float error_bias = (grad_bias - grad_bias_sequential).norm();
    // float error_bottom = (grad_bottom - grad_bottom_sequential).norm();

    // std::cout << "Backward Version 2: Weight Error: " << error_weight << std::endl;
    // std::cout << "Backward Version 2: Bias Error: " << error_bias << std::endl;
    // std::cout << "Backward Version 2: Bottom Error: " << error_bottom << std::endl;

    // Giải phóng bộ nhớ
    free(h_bottom);
    free(h_grad_top);
    free(h_grad_bottom);
}




// Parallel Version (optimized)
void FullyConnected::backwardVersion_3(const Matrix& bottom, const Matrix& grad_top) {
    const int n_sample = bottom.cols();

    // Chuẩn bị bộ nhớ
    float* h_bottom = (float*)malloc(dim_in * n_sample * sizeof(float));
    float* h_grad_top = (float*)malloc(dim_out * n_sample * sizeof(float));
    float* h_grad_bottom = (float*)malloc(dim_in * n_sample * sizeof(float));

    // Sao chép dữ liệu từ Eigen sang mảng liên tục
    std::memcpy(h_bottom, bottom.data(), dim_in * n_sample * sizeof(float));
    std::memcpy(h_grad_top, grad_top.data(), dim_out * n_sample * sizeof(float));

    // 1. Tính grad_weight bằng code tuần tự
    grad_weight = bottom * grad_top.transpose();

    // 2. Tính grad_bias = \sum(d(L)/d(z))
    grad_bias.resize(dim_out, 1);
    for (int i = 0; i < dim_out; ++i) {
        grad_bias(i, 0) = grad_top.row(i).sum();
    }

    // 3. Tính grad_bottom = weight * grad_top (sử dụng song song trên GPU)
    matrixMultiplicationGPUWrapper(weight.data(), h_grad_top, h_grad_bottom, dim_in, dim_out, n_sample, 0, true);

    // Chuyển kết quả từ GPU về Eigen Matrix
    grad_bottom = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(h_grad_bottom, dim_in, n_sample);

    // Debug giá trị
    // Matrix grad_weight_sequential = bottom * grad_top.transpose();
    // Matrix grad_bias_sequential = grad_top.rowwise().sum();
    // Matrix grad_bottom_sequential = weight * grad_top;

    // float error_weight = (grad_weight - grad_weight_sequential).norm();
    // float error_bias = (grad_bias - grad_bias_sequential).norm();
    // float error_bottom = (grad_bottom - grad_bottom_sequential).norm();

    // std::cout << "Backward Version 2: Weight Error: " << error_weight << std::endl;
    // std::cout << "Backward Version 2: Bias Error: " << error_bias << std::endl;
    // std::cout << "Backward Version 2: Bottom Error: " << error_bottom << std::endl;

    // Giải phóng bộ nhớ
    free(h_bottom);
    free(h_grad_top);
    free(h_grad_bottom);
}



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
