#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/empty_loss.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/pendulum.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/sac.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/training_config.hpp"

#include <ensmallen.hpp>
#include <typeinfo>

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;

int main()
{
  mlpack::math::RandomSeed(std::time(NULL));
  // Set up the policy and replay method.
  RandomReplay<Pendulum> replayMethod(32, 10000);

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.TargetNetworkSyncInterval() = 1;
  config.UpdateInterval() = 3;

  FFN<EmptyLoss<>, GaussianInitialization>
      policyNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
  policyNetwork.Add(new Linear<>(3, 128));
  policyNetwork.Add(new ReLULayer<>());
  policyNetwork.Add(new Linear<>(128, 1));
  policyNetwork.Add(new TanHLayer<>());

  FFN<EmptyLoss<>, GaussianInitialization>
      qNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
  qNetwork.Add(new Linear<>(3+1, 128));
  qNetwork.Add(new ReLULayer<>());
  qNetwork.Add(new Linear<>(128, 1));

    FFN<EmptyLoss<>, GaussianInitialization>
      qNetwork2(EmptyLoss<>(), GaussianInitialization(0, 0.1));
  qNetwork2.Add(new Linear<>(3+1, 128));
  qNetwork2.Add(new ReLULayer<>());
  qNetwork2.Add(new Linear<>(128, 1));

  // Set up Soft actor-critic agent.
  SAC<Pendulum, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
      agent(config, qNetwork, qNetwork2, policyNetwork, replayMethod);

  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    agent.Deterministic() = false;
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;

    if (episodes > 1000)
    {
      std::cout << "Pendulum failed." << std::endl;
      converged = false;
      break;
    }
    std::cout << "Average return: " << averageReturn.mean()
        << " Episode return: " << episodeReturn << std::endl;
    
    if(episodes % 10 == 0)
    {
      arma::running_stat<double> averageTestReturn;
      agent.Deterministic() = true;
      for (size_t i = 0; i < 10; i++)
      {
        episodeReturn = agent.Episode();
        averageTestReturn(episodeReturn);
        std::cout << episodeReturn << " ";
      }
      std::cout << std::endl;
      std::cout << "TEST Average return: " << averageTestReturn.mean() << std::endl;
    }
  }
}