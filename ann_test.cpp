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
    qNetwork.Add<Linear<>>(4, 2);
    qNetwork.Add(new ReLULayer<>());
    qNetwork.Add(new Linear<>(2, 4));
    qNetwork.ResetParameters();
}
