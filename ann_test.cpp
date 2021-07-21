#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/empty_loss.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error.hpp"

using namespace mlpack::ann;
using namespace mlpack;

int main()
{
  arma::mat input = arma::mat("-0.1; 0.9");
  arma::mat output, gy = arma::mat("2; 1"), g;
  // May be I am not using 'gy' in correct way.

  LogSoftMax<> module;
  module.Forward(input, output);
  output.print("Forward:");
  module.Backward(output, gy, g);
  g.print("Backward:");
}