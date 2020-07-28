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
    mlpack::math::RandomSeed(std::time(NULL));
    // 3 x 2
    FFN<EmptyLoss<>, GaussianInitialization>
        qNetwork(EmptyLoss<>(), GaussianInitialization(0, 1));
    qNetwork.Add<Linear<>>(3 + 1, 2);
    qNetwork.Add(new ReLULayer<>());
    qNetwork.Add(new Linear<>(2, 1));

    qNetwork.ResetParameters();

    arma::colvec i = {2, -1, 0, 1};
    arma::colvec o;

    arma::colvec b = {2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1};
    qNetwork.Parameters() = b;

    qNetwork.Predict(i, o);
    std::cout << o << std::endl;

    arma::mat gradQ;
    qNetwork.Forward(i, o);
    qNetwork.Backward(i, -1, gradQ);

    size_t hidden1 = boost::get<Linear<> *>(qNetwork.Model()[0])->OutputSize();
    arma::colvec gradQBias = gradQ(i.n_rows * hidden1, 0, arma::size(hidden1, 1));
    std::vector<size_t> indices;
    for (size_t j = 0; j < hidden1; j++)
        indices.push_back(j * i.n_rows);
    arma::uvec p = arma::conv_to<arma::uvec>::from(indices);
    arma::rowvec weightLastLayer = qNetwork.Parameters().rows(p).t();

    arma::mat ans = weightLastLayer * gradQBias;
    std::cout << ans << std::endl;
    std::cout << o << std::endl;
    std::cout << i << std::endl;
}
