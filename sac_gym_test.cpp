#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/empty_loss.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/env_type.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/sac.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/training_config.hpp"

#include <ensmallen.hpp>
#include <iostream>

#include "../gym_tcp_api/cpp/environment.hpp"

using namespace gym;

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::rl;

int main(int argc, char *argv[])
{
    mlpack::math::RandomSeed(std::time(NULL));
    ContinuousActionEnv::State::dimension = 24;
    ContinuousActionEnv::Action::size = 4;

    TrainingConfig config;
    config.ExplorationSteps() = 2000;
    config.StepSize() = 0.001;
    config.TargetNetworkSyncInterval() = 1;
    config.UpdateInterval() = 1;

    // Set up the policy and replay method.
    RandomReplay<ContinuousActionEnv> replayMethod(32, 10000);

    const std::string environment = "BipedalWalker-v3";
    const std::string host = "127.0.0.1";
    const std::string port = "4040";

    Environment env(host, port, environment);

    env.compression(9);
    env.monitor.start("./dummy/", true, true);

    env.reset();
    env.render();

    for (int episodes = 50; episodes < 1000; episodes += 50)
    {
        FFN<EmptyLoss<>, GaussianInitialization>
            policyNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));

        FFN<EmptyLoss<>, GaussianInitialization>
            qNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));

        data::Load("./" + std::to_string(episodes) + "qNetwork.xml", "episode", qNetwork);
        data::Load("./" + std::to_string(episodes) + "policyNetwork.xml", "episode", policyNetwork);

        // Set up Soft actor-critic agent.
        SAC<ContinuousActionEnv, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
            agent(config, qNetwork, policyNetwork, replayMethod);
        double currentReward = 0;
        size_t currentStep = 0;
        env.reset();
        agent.Deterministic() = true;
        while (1)
        {
            agent.State().Data() = env.observation;
            agent.SelectAction();
            arma::mat action = {agent.Action().action};

            env.step(action);
            currentReward += env.reward;
            currentStep += 1;

            if (env.done)
            {
                std::cout << "current reward: " << currentReward << " episode: " << episodes << std::endl;
                break;
            }

            // std::cout << "Current step: " << currentStep << " current reward: "
            //           << currentReward << std::endl;
        }
    }

    std::cout << "Video: https://kurg.org/media/gym/" << env.instance
              << " (it might take some minutes before the video is accessible)."
              << std::endl;

    return 0;
}