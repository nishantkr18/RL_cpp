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
    ContinuousActionEnv::State::dimension = 3;
    ContinuousActionEnv::Action::size = 1;

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
    qNetwork.Add(new Linear<>(3 + 1, 128));
    qNetwork.Add(new ReLULayer<>());
    qNetwork.Add(new Linear<>(128, 1));

    FFN<EmptyLoss<>, GaussianInitialization>
        qNetwork2(EmptyLoss<>(), GaussianInitialization(0, 0.1));
    qNetwork2.Add(new Linear<>(3 + 1, 128));
    qNetwork2.Add(new ReLULayer<>());
    qNetwork2.Add(new Linear<>(128, 1));

    // Set up the policy and replay method.
    RandomReplay<ContinuousActionEnv> replayMethod(32, 10000);

    // Set up Soft actor-critic agent.
    SAC<ContinuousActionEnv, decltype(qNetwork), decltype(policyNetwork), AdamUpdate>
        agent(config, qNetwork, qNetwork2, policyNetwork, replayMethod);

    const std::string environment = "Pendulum-v0";
    const std::string host = "127.0.0.1";
    const std::string port = "4040";

    Environment env(host, port, environment);

    std::vector<double> returnList;
    size_t episodes = 0;
    bool converged = true;
    size_t consecutiveEpisodesTest = 50;
    while (true)
    {
        double episodeReturn = 0;
        env.reset();
        do
        {
            agent.State().Data() = env.observation;
            agent.SelectAction();
            arma::mat action = {agent.Action().action[0]};

            env.step(action);
            ContinuousActionEnv::State nextState;
            nextState.Data() = env.observation;

            replayMethod.Store(agent.State(), agent.Action(), env.reward, nextState, env.done, 0.99);
            episodeReturn += env.reward;
            agent.TotalSteps()++;
            if (agent.Deterministic() || agent.TotalSteps() < config.ExplorationSteps())
                continue;
            for(size_t i = 0; i < config.UpdateInterval(); i++)
                agent.Update();
        } while (!env.done);
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
        if (averageReturn > -400)
            break;
    }
    env.compression(9);
    env.monitor.start("./dummy/", true, true);

    env.reset();
    env.render();

    for (int i = 0; i < 50; i++)
    {
        double currentReward = 0;
        size_t currentStep = 0;
        env.reset();
        agent.Deterministic() = true;
        while (1)
        {
            agent.State().Data() = env.observation;
            agent.SelectAction();
            arma::mat action = {agent.Action().action[0]};

            env.step(action);
            currentReward += env.reward;
            currentStep += 1;

            if (env.done)
            {
                std::cout << "current reward: " << currentReward << " iteration: " << i << std::endl;
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