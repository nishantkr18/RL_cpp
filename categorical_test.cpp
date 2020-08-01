#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/acrobot.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/mountain_car.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/cart_pole.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/q_learning.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/q_networks/categorical_dqn.hpp"
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
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 1.0);
  RandomReplay<CartPole> replayMethod(32, 10000);

  TrainingConfig config;
  config.StepSize() = 0.01;
  config.Discount() = 0.99;
  config.TargetNetworkSyncInterval() = 100;
  config.ExplorationSteps() = 32;
  config.DoubleQLearning() = false;
  config.IsCategorical() = true;

  FFN<EmptyLoss<>, GaussianInitialization> module(
      EmptyLoss<>(), GaussianInitialization(0, 0.1));
  module.Add<Linear<>>(4, 128);
  module.Add<ReLULayer<>>();
  module.Add<Linear<>>(128, 2 * 51);

  CategoricalDQN<> network(module);

  // Set up DQN agent.
  QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy),
            decltype(replayMethod)>
      agent(config, network, policy, replayMethod);

  std::vector<double> returnList;
  size_t episodes = 0;
  bool converged = true;
  size_t consecutiveEpisodesTest = 50;
  while (true)
  {
    double episodeReturn = agent.Episode();
    returnList.push_back(episodeReturn);
    episodes += 1;

    if (returnList.size() <= consecutiveEpisodesTest)
    {
      std::cout << episodeReturn << std::endl;
      continue;
    }
    else
      returnList.erase(returnList.begin());
    double averageReturn = std::accumulate(returnList.begin(),
                                           returnList.end(), 0.0) /
                           returnList.size();

    std::cout << "Average return in last " << consecutiveEpisodesTest
              << " consecutive episodes: " << averageReturn
              << " Episode return: " << episodeReturn << std::endl;

    if (episodes > 1000)
    {
      std::cout << "Cart Pole with DQN failed." << std::endl;
      converged = false;
      break;
    }
    if (averageReturn > 150)
      break;
  }
  return 0;
}