#ifndef SRC_LAYER_FULLY_CONNECTED_H_
#define SRC_LAYER_FULLY_CONNECTED_H_

#include <vector>
#include "../layer.h"

class FullyConnected : public Layer {
 private:
  const int dim_in;
  const int dim_out;

  Matrix weight;  // weight parameter
  Vector bias;  // bias paramter
  Matrix grad_weight;  // gradient w.r.t weight
  Vector grad_bias;  // gradient w.r.t bias

  void init();

 public:
  FullyConnected(const int dim_in, const int dim_out) :
                 dim_in(dim_in), dim_out(dim_out)
  { init(); }

  void forward(const Matrix& bottom);
  void forwardVersion_0(const Matrix& bottom); // Library
  void forwardVersion_1(const Matrix& bottom); // Sequential
  void forwardVersion_2(const Matrix& bottom); // Parallel (Not optimized)
  void forwardVersion_3(const Matrix& bottom); // Parallel (Optimized)
  void backward(const Matrix& bottom, const Matrix& grad_top);
  void backwardVersion_0(const Matrix& bottom, const Matrix& grad_top); // Library
  void backwardVersion_1(const Matrix& bottom, const Matrix& grad_top); // Sequential
  void backwardVersion_2(const Matrix& bottom, const Matrix& grad_top); // Parallel (Not optimized)
  void backwardVersion_3(const Matrix& bottom, const Matrix& grad_top); // Parallel (Optimized)
  void update(Optimizer& opt);
  int output_dim() { return dim_out; }
  std::vector<float> get_parameters() const;
  std::vector<float> get_derivatives() const;
  void set_parameters(const std::vector<float>& param);
  const Matrix& get_weight() const {return weight;}
};

#endif  // SRC_LAYER_FULLY_CONNECTED_H_
