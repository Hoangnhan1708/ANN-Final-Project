#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "src/layer.h"
#include "src/layer/fully_connected.h"
#include "src/layer/relu.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

#include "src/layer/cuda_utilities.h"
#include "config.h"
#include "testImplement.h"
#include <thread>
#include <vector>
#include <numeric>

const bool IS_TRAINING = true;
const bool IS_CREATING_TEST_CASES = false;

namespace config 
{
  int currentVersion = 1;
  int startVersion = 2;
  int endVersion = 2;
  int isTraining = 1;
  float forwardTime = 0;
}

void saveNetworkParameters(Network& network, std::string filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    std::vector<std::vector<float>> parameters = network.get_parameters();

    for (const auto &v: parameters) {
      for (const auto &f: v) {
        file << f << " ";
      }
      file << "\n";
    }

  }
}

void loadNetworkParameters(Network& network, std::string filename) {
  std::ifstream file(filename);
  if (file.is_open()) {
    std::vector<std::vector<float>> parameters = network.get_parameters();

    for (auto &v: parameters) {
      for (auto &f: v) {
        file >> f;
      }
    }

    network.set_parameters(parameters);
    std::cout << "Parameters loaded" << std::endl;
  }
  else {
    std::cout << "Unable to open file " << filename << std::endl;
  }
}

void testing(Network& ann, MNIST& dataset, int epoch) {
  // startTimer();    
  ann.forward(dataset.test_data);
  // std::cout << "Test time: " << stopTimer() << std::endl;
   
  float acc = compute_accuracy(ann.output(), dataset.test_labels);
  std::cout << "Accuracy: " << acc << std::endl;
  std::cout << std::endl;
}



int main(int argc, char** argv) {
  config::startVersion = std::stoi(argv[1]);
  config::endVersion = std::stoi(argv[2]);
  config::isTraining = std::stoi(argv[3]);

  // data
  MNIST dataset("../data/fashion-mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // file to save parameters
  // std::string filename = "../../../Model/parameters_version1.txt";
  
  // ann
  Network ann;
  Layer* F1 = new FullyConnected(784, 128);
  Layer* F2 = new FullyConnected(128, 128);
  Layer* F3 = new FullyConnected(128, 10);
  Layer* relu1 = new ReLU;
  Layer* relu2 = new ReLU;
  Layer* softmax = new Softmax;
  
  ann.add_layer(F1);
  ann.add_layer(relu1);
  ann.add_layer(F2);
  ann.add_layer(relu2);
  ann.add_layer(F3);
  ann.add_layer(softmax);
  
  // loss
  Loss* loss = new CrossEntropy;
  ann.add_loss(loss);
  // train & test
  SGD opt(0.0001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;


  if (config::isTraining == 0) {
    for (int v = config::startVersion; v <= config::endVersion; v++)
    {
      config::currentVersion = v;
      std::string filename = "../../../Model/parameters_version_" + std::to_string(v) + ".txt";
      loadNetworkParameters(ann, filename);
      std::cout << "\nCurrent version: " << config::currentVersion << "\n\n";

      std::cout << "\n\n";

      // Run on the test set
      testing(ann, dataset, 0);
      std::cout << "------------------------------------------\n" << std::endl;

      // if (!config::runAllVersion)
      //   break;
    }

    return 0;
  }

  

  for (int v = config::startVersion; v <= config::endVersion; v++)
  {
    config::currentVersion = v;
    std::cout << "---------------------------Current Version "<< v << "----------------------------------\n";
    startTimer();
    for (int epoch = 0; epoch < n_epoch; epoch ++) 
    {
      std::cout << "---------Epochs "<< epoch << "----------\n";
      shuffle_data(dataset.train_data, dataset.train_labels);
      for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) 
      {
        int ith_batch = start_idx / batch_size;
        Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                      std::min(batch_size, n_train - start_idx));
        Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                      std::min(batch_size, n_train - start_idx));
        Matrix target_batch = one_hot_encode(label_batch, 10);
        if (false && ith_batch % 10 == 1) {
          std::cout << ith_batch << "-th grad: " << std::endl;
          ann.check_gradient(x_batch, target_batch, 10);
        }
        
        ann.forward(x_batch);

        ann.backward(x_batch, target_batch);
        // display
        if (ith_batch % 50 == 0) {
          std::cout << ith_batch << "-th batch, loss: " << ann.get_loss() << std::endl;
        }
        // optimize
        ann.update(opt);
        
        std::string fileParamaters = "../../../Model/parameters_version_" + std::to_string(v) + ".txt";
        saveNetworkParameters(ann, fileParamaters);
      }
    
      // test
      testing(ann, dataset, epoch);
    }
    std::cout << "Train time: " << stopTimer() << std::endl;
  }
  
  return 0;
}

