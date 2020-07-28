#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/empty_loss.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/acrobot.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/cart_pole.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/double_pole_cart.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/mountain_car.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/q_learning.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/q_networks/simple_dqn.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/q_networks/dueling_dqn.hpp"
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
  GreedyPolicy<CartPole> policy(1.0, 1000, 0.1, 0.99);
  RandomReplay<CartPole> replayMethod(32, 10000);

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.Discount() = 0.99;
  config.TargetNetworkSyncInterval() = 50;
  config.ExplorationSteps() = 32;
  config.StepLimit() = 200;
  config.DoubleQLearning() = false;
  config.NoisyQLearning() = true;

  DuelingDQN<> network(4, 64, 64, 2, true);

  // Set up DQN agent.
  QLearning<CartPole, decltype(network), AdamUpdate, decltype(policy)>
      agent(config, network, policy, replayMethod);

  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;

    if (episodes > 1000)
    {
      std::cout << "Cart Pole with DQN failed." << std::endl;
      converged = false;
      break;
    }
    std::cout << "Average return: " << averageReturn.mean()
              << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 100)
    {
      agent.Deterministic() = true;
      arma::running_stat<double> testReturn;
      for (size_t i = 0; i < 10; ++i)
        testReturn(agent.Episode());

      std::cout << "Average return in deterministic test: "
                << testReturn.mean() << std::endl;
      break;
    }
  }
}