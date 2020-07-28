#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error.hpp"

#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/env_type.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/q_learning.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/q_networks/simple_dqn.hpp"
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
    DiscreteActionEnv::State::dimension = 3;
    DiscreteActionEnv::Action::size = 3;

    // Set up the network.
    FFN<MeanSquaredError<>, GaussianInitialization> network(
        MeanSquaredError<>(), GaussianInitialization(0, 1));
    network.Add<Linear<>>(DiscreteActionEnv::State::dimension, 128);
    network.Add<ReLULayer<>>();
    network.Add<Linear<>>(128, DiscreteActionEnv::Action::size);
    // Set up the network.
    SimpleDQN<> model(network);

    // Set up the policy and replay method.
    GreedyPolicy<DiscreteActionEnv> policy(1.0, 1000, 0.1, 0.99);
    RandomReplay<DiscreteActionEnv> replayMethod(32, 10000);

    TrainingConfig config;
    config.ExplorationSteps() = 100;

    const std::string environment = "Pendulum-v0";
    const std::string host = "127.0.0.1";
    const std::string port = "4040";

    Environment env(host, port, environment);
    // Set up DQN agent.
    QLearning<DiscreteActionEnv, decltype(model), AdamUpdate, decltype(policy), decltype(replayMethod)>
        agent(config, model, policy, replayMethod);

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
            arma::mat action = {double(agent.Action().action) - 1.0};

            env.step(action);
            action = env.action_space.sample();
            DiscreteActionEnv::State nextState;
            nextState.Data() = env.observation;

            replayMethod.Store(agent.State(), agent.Action(), env.reward, nextState, env.done, 0.99);
            episodeReturn += env.reward;
            agent.TotalSteps()++;
            if (agent.Deterministic() || agent.TotalSteps() < config.ExplorationSteps())
                continue;
            agent.TrainAgent();
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
            arma::mat action = {double(agent.Action().action) - 1.0};

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