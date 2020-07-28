
#include <iostream>
#include <math.h>
#include <armadillo>
#include <queue>
#include <cassert>

int main()
{
  arma::colvec b = {0,0,0,0,0,0,0};
  arma::mat a = b;
  std::cout << a.n_rows << std::endl;
  // double vMin = 0, vMax = 200.0;
  // size_t atomSize = 5;
  // size_t batchSize = 6;
  // arma::icolvec isTerminal = arma::zeros<arma::icolvec>(64, 1);
  // arma::colvec sampledRewards = arma::ones<arma::colvec>(64, 1);
  // arma::mat activationGradients = arma::randn(atomSize * 2, batchSize);
  // for (size_t i = 0; i < activationGradients.n_rows; i += atomSize)
  // {
  //   activationGradients.rows(i, i + atomSize - 1) = arma::randn(atomSize, batchSize);
  // }
  // std::cout << activationGradients << std::endl;

  // std::cout << pow(4, 2) << std::endl;

  // arma::vec v = arma::linspace<arma::vec>(10,15,6);

  // X.each_col() += v;         // in-place addition of v to each column vector of X

  // arma::mat Y = X.each_col() + v;

  // A.set_size(2,2);
  // cout << A << '\n';

  // arma::vec B(5);
  // cout << B << '\n';
  // B.fill(4);
  // cout << B.n_rows << '\n';

  //   arma::mat target(1,2);
  // target(0,0) = 100.0;
  // target(0,1) = -100.0;
  // arma::mat gradients;
  // critic.Backward(input, target, gradients);
  // for (double x = 0; x<100; x+=0.1){
  //   if (std::log(1 + std::exp(x)) != (std::log(1 + std::exp(-x)) + x))
  //   {
  //     std::cout << "x: " << x << '\n';
  //     std::cout << "std::log(1 + std::exp(x)): " << std::log(1 + std::exp(x)) << '\n';
  //     std::cout << "std::log(1 + std::exp(-x)) + x: " << std::log(1 + std::exp(-x))+x << '\n';
  //     double diff = std::log(1 + std::exp(-x))+x - std::log(1 + std::exp(x));
  //     std::cout << "diff: " << diff << '\n';

  //   }
  //   if(std::log(1 + std::exp(-x)) == 0)
  //     std::cout << "VALUE OF x WHERE IT TURNS 0: " << x << '\n';
  //   // std::cout << "std::log(1 + std::exp(-x)): " << " , " <<  << '\n';
  // }

  // std::cout << "val: " << std::log(1 + std::exp(std::numeric_limits<double>::max())) << '\n';
  // std::cout << "val: " << arma::log(target) << '\n';
}