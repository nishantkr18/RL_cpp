#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/empty_loss.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/pendulum.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/training_config.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/ppo.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp"

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
  GreedyPolicy<Pendulum> policy(1.0, 1000, 0.1, 0.99);
  RandomReplay<Pendulum> replayMethod(64, 100);

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.UpdateInterval() = 32;
  config.ActorUpdateStep() = 10;
  config.Epsilon() = 0.2;

  FFN<EmptyLoss<>, GaussianInitialization> actor(
      EmptyLoss<>(), GaussianInitialization(0, 0.001));
  actor.Add<Linear<>>(2, 128);
  actor.Add<ReLULayer<>>();
  actor.Add<Linear<>>(128, 2);

  FFN<MeanSquaredError<>, GaussianInitialization> critic(
      MeanSquaredError<>(), GaussianInitialization(0, 0.001));
  critic.Add<Linear<>>(2, 128);
  critic.Add<ReLULayer<>>();
  critic.Add<Linear<>>(128, 1);

  // Set up PPO agent.
  PPO<Pendulum, decltype(actor), decltype(critic), AdamUpdate,
      decltype(policy)>
      agent(config, actor, critic, policy, replayMethod);

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

    if (episodes % 10 == 0)
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