#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/empty_loss.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/env_type.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/cart_pole.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/acrobot.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/ppo.hpp"
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
  DiscreteActionEnv env();
  DiscreteActionEnv::State::dimension = 4;

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.Discount() = 0.99;
  config.StepLimit() = 200;

  FFN<EmptyLoss<>, GaussianInitialization>
      actor(EmptyLoss<>(), GaussianInitialization(0, 1));
  actor.Add(new Linear<>(4, 128));
  actor.Add(new ReLULayer<>());
  actor.Add(new Linear<>(128, 2));

  FFN<EmptyLoss<>, GaussianInitialization>
      critic(EmptyLoss<>(), GaussianInitialization(0, 1));
  critic.Add(new Linear<>(4, 128));
  critic.Add(new ReLULayer<>());
  critic.Add(new Linear<>(128, 1));

  // Set up DQN agent.
  PPO<CartPole, decltype(actor), decltype(critic), AdamUpdate>
      agent(config, actor, critic);

  arma::running_stat<double> averageReturn;
  size_t episodes = 0;
  bool converged = true;
  while (true)
  {
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes += 1;

    if (episodes > 10000)
    {
      std::cout << "Cart Pole with DQN failed." << std::endl;
      converged = false;
      break;
    }

    /**
     * Reaching running average return 35 is enough to show it works.
     * For the speed of the test case, I didn't set high criterion.
     */
    std::cout << "Episode: " << episodes
              << "Average return: " << averageReturn.mean()
              << " Episode return: " << episodeReturn << std::endl;
    if (averageReturn.mean() > 190)
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