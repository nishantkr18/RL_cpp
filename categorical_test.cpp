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
  config.StepSize() = 0.001;
  config.Discount() = 0.99;
  config.TargetNetworkSyncInterval() = 200;
  config.ExplorationSteps() = 64;
  config.DoubleQLearning() = false;
  config.StepLimit() = 200;
  config.IsCategorical() = true;

  CategoricalDQN<> model(4, 128, 128, 2);

  // Set up DQN agent.
  QLearning<CartPole, decltype(model), AdamUpdate, decltype(policy)>
      agent(config, model, policy, replayMethod);

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

    /**
     * Reaching running average return 35 is enough to show it works.
     * For the speed of the test case, I didn't set high criterion.
     */
    std::cout << "Average return: " << averageReturn.mean()
        << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 150)
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