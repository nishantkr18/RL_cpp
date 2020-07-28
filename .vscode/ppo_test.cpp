#include "reinforcement_learning/environment/continuous_mountain_car.hpp"
#include "reinforcement_learning/environment/lunarlander.hpp"
#include "reinforcement_learning/environment/pendulum.hpp"
#include "reinforcement_learning/policy/greedy_policy.hpp"
#include "reinforcement_learning/ppo.hpp"
#include "reinforcement_learning/training_config.hpp"
#include <mlpack/methods/ann/dists/normal_distribution.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/empty_loss.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>

#include <ensmallen.hpp>
#include <math.h>
#include <mlpack/core.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;
// using namespace std;

// int testing()
// {
//   mlpack::math::RandomSeed(std::time(NULL));
//   FFN<EmptyLoss<>, GaussianInitialization> critic(
//       EmptyLoss<>(), GaussianInitialization(0, 0.1));

//   critic.Add<Linear<>>(2, 64);
//   critic.Add<ReLULayer<>>();
//   critic.Add<Linear<>>(64, 128);
//   critic.Add<ReLULayer<>>();
//   critic.Add<Linear<>>(128, 2);
//   critic.ResetParameters();
//   cout << "Parameters:" << critic.Parameters() << '\n';

//   arma::mat input(2, 1);
//   input.fill(1);
//   cout << "input: " << input << '\n';

//   arma::mat output;
//   critic.Forward(input, output);
//   cout << "output: " << output << '\n';

//   //   critic.Predict(input, output);
//   //   cout << "Parameters:" << critic.Parameters() << '\n';

//   //   cout << "output: " << output << '\n';

//   //   arma::mat target(1,2);
//   output(0, 0) += 2;
//   output(0, 1) += 2;
//   cout << "target: " << output << '\n';
//   arma::mat gradients;
//   critic.Backward(input, output, gradients);
//   cout << "gradients: " << gradients << '\n';
// }

// PENDULUM
int pendulumTest()
{
  // mlpack::math::RandomSeed(457);
  mlpack::math::RandomSeed(std::time(NULL));
  size_t episodes = 0;
  bool converged = false;
  // Set up the network.
  FFN<MeanSquaredError<>, GaussianInitialization> critic(
      MeanSquaredError<>(), GaussianInitialization(0, 1));

  critic.Add<Linear<>>(3, 64);
  critic.Add<ReLULayer<>>();
  critic.Add<Linear<>>(64, 256);
  critic.Add<ReLULayer<>>();
  critic.Add<Linear<>>(256, 1);

  FFN<EmptyLoss<>, GaussianInitialization> actor(
      EmptyLoss<>(), GaussianInitialization(0, 1));
  actor.Add<Linear<>>(3, 64);
  actor.Add<ReLULayer<>>();
  actor.Add<Linear<>>(64, 256);
  actor.Add<ReLULayer<>>();
  actor.Add<Linear<>>(256, 2);

  // Set up the policy and replay method.
  GreedyPolicy<Pendulum> policy(1.0, 1000, 0.1, 0.99);
  RandomReplay<Pendulum> replayMethod(32, 2000);

  TrainingConfig config;
  config.StepSize() = 0.001;
  config.Discount() = 0.9;
  config.Epsilon() = 0.2;
  config.StepLimit() = 200;
  config.UpdateInterval() = 32;
  config.ActorUpdateStep() = 10;

  // Set up the PPO agent.
  PPO<Pendulum, decltype(actor), decltype(critic), AdamUpdate,
      decltype(policy)>
      agent(std::move(config), std::move(actor), std::move(critic),
            std::move(policy), std::move(replayMethod));

  arma::running_stat<double> averageReturn;

  for (episodes = 0; episodes <= 10000; ++episodes)
  {
    // agent.Deterministic() = true;
    double episodeReturn = agent.Episode();
    if (std::isnan(episodeReturn))
    {
      std::cout << "STOP!!!!!!!!!!!!!!!!" << '\n';
      break;
    }
    averageReturn(episodeReturn);

    /**
     * I am using a threshold of -136.16 to check convergence.
     */
    if (episodes % 10 == 0)
      std::cout << "Average return: " << averageReturn.mean()
                << " Episode return: " << episodeReturn << "Episode" << episodes << std::endl;
    if (averageReturn.mean() > -136.16)
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
// // CONTINUOUS MOUNTAIN CAR
// int contMoun()
// {
//   mlpack::math::RandomSeed(std::time(NULL));
//   size_t episodes = 0;
//   bool converged = false;
//   for (size_t trial = 0; trial < 4; ++trial) {
//     // Set up the network.
//     FFN<MeanSquaredError<>, GaussianInitialization> critic(
//       MeanSquaredError<>(), GaussianInitialization(0, 0.001));

//     critic.Add<Linear<>>(2, 128);
//     critic.Add<ReLULayer<>>();
//     critic.Add<Linear<>>(128, 1);

//     FFN<EmptyLoss<>, GaussianInitialization> actor(
//       EmptyLoss<>(), GaussianInitialization(0, 0.001));

//     actor.Add<Linear<>>(2, 128);
//     actor.Add<ReLULayer<>>();
//     actor.Add<Linear<>>(128, 2);

//     // Set up the policy and replay method.
//     GreedyPolicy<ContinuousMountainCar> policy(1.0, 1000, 0.1, 0.99);
//     RandomReplay<ContinuousMountainCar> replayMethod(64, 10000);

//     TrainingConfig config;
//     config.StepSize() = 0.0001;
//     config.Discount() = 0.99;
//     config.Epsilon() = 0.2;
//     config.StepLimit() = 1000;
//     config.UpdateInterval() = 64;
//     config.ActorUpdateStep() = 10;

//     // Set up the PPO agent.
//     PPO<ContinuousMountainCar, decltype(actor), decltype(critic), AdamUpdate,
//         decltype(policy)>
//         agent(std::move(config), std::move(actor), std::move(critic),
//         std::move(policy), std::move(replayMethod));

//     arma::running_stat<double> averageReturn;

//     for (episodes = 0; episodes <= 1000; ++episodes) {
//       // agent.Deterministic() = true;
//       double episodeReturn = agent.Episode();
//       if(std::isnan(episodeReturn*-1)) {std::cout << "STOP!!!!!!!!!!!!!!!!" << '\n'; break; }
//       averageReturn(episodeReturn);

//       /**
//        * I am using a threshold of 120 to check convergence.
//        */
//       std::cout << "Average return: " << averageReturn.mean()
//                  << " Episode return: " << episodeReturn << std::endl;
//       if (averageReturn.mean() > 1200) {
//         agent.Deterministic() = true;
//         arma::running_stat<double> testReturn;
//         for (size_t i = 0; i < 10; ++i)
//           testReturn(agent.Episode());
//         std::cout << "Average return in deterministic test: "
//                    << testReturn.mean() << std::endl;
//         break;
//       }
//     }

//     if (episodes < 1000) {
//       converged = true;
//       break;
//     }
//   }
// }

// // LunarLander
// int main3()
// {
//   mlpack::math::RandomSeed(std::time(NULL));
//   size_t episodes = 0;
//   bool converged = false;
//   for (size_t trial = 0; trial < 4; ++trial) {
//     // Set up the network.
//     FFN<MeanSquaredError<>, GaussianInitialization> critic(
//       MeanSquaredError<>(), GaussianInitialization(0, 0.001));

//     critic.Add<Linear<>>(4, 64);
//     critic.Add<ReLULayer<>>();
//     critic.Add<Linear<>>(64, 32);
//     critic.Add<ReLULayer<>>();
//     critic.Add<Linear<>>(32, 1);

//     FFN<EmptyLoss<>, GaussianInitialization> actor(
//       EmptyLoss<>(), GaussianInitialization(0, 0.001));

//     actor.Add<Linear<>>(4, 64);
//     actor.Add<ReLULayer<>>();
//     actor.Add<Linear<>>(64, 32);
//     actor.Add<ReLULayer<>>();
//     actor.Add<Linear<>>(32, 2);

//     // Set up the policy and replay method.
//     GreedyPolicy<LunarLander> policy(1.0, 1000, 0.1, 0.99);
//     RandomReplay<LunarLander> replayMethod(64, 10000);

//     TrainingConfig config;
//     config.StepSize() = 0.0001;
//     config.Discount() = 0.99;
//     config.Epsilon() = 0.2;
//     config.StepLimit() = 1000;
//     config.UpdateInterval() = 64;
//     config.ActorUpdateStep() = 10;

//     // Set up the PPO agent.
//     PPO<LunarLander, decltype(actor), decltype(critic), AdamUpdate,
//         decltype(policy)>
//         agent(std::move(config), std::move(actor), std::move(critic),
//         std::move(policy), std::move(replayMethod));

//     arma::running_stat<double> averageReturn;

//     for (episodes = 0; episodes <= 1000; ++episodes) {
//       // agent.Deterministic() = true;
//       double episodeReturn = agent.Episode();
//       if(std::isnan(episodeReturn*-1)) {std::cout << "STOP!!!!!!!!!!!!!!!!" << '\n'; break; }
//       averageReturn(episodeReturn);

//       /**
//        * I am using a threshold of 120 to check convergence.
//        */
//       std::cout << "Average return: " << averageReturn.mean()
//                  << " Episode return: " << episodeReturn << std::endl;
//       if (averageReturn.mean() > 1200) {
//         agent.Deterministic() = true;
//         arma::running_stat<double> testReturn;
//         for (size_t i = 0; i < 10; ++i)
//           testReturn(agent.Episode());
//         std::cout << "Average return in deterministic test: "
//                    << testReturn.mean() << std::endl;
//         break;
//       }
//     }

//     if (episodes < 1000) {
//       converged = true;
//       break;
//     }
//   }
// }

int main()
{
  pendulumTest();
  // testing();
}
