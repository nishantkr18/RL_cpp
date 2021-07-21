#include <mlpack/core.hpp>

#include "../mlpack/src/mlpack/methods/ann/ffn.hpp"
#include "../mlpack/src/mlpack/methods/ann/init_rules/gaussian_init.hpp"
#include "../mlpack/src/mlpack/methods/ann/layer/layer.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/empty_loss.hpp"
#include "../mlpack/src/mlpack/methods/ann/loss_functions/mean_squared_error.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/environment/env_type.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/training_config.hpp"
#include "../mlpack/src/mlpack/methods/reinforcement_learning/replay/random_replay.hpp"

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
    config.StepSize() = 0.001;
    config.TargetNetworkSyncInterval() = 1;
    config.UpdateInterval() = 3;

    FFN<EmptyLoss<>, GaussianInitialization>
        learningQNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
    learningQNetwork.Add(new Linear<>(ContinuousActionEnv::State::dimension + ContinuousActionEnv::Action::size, 128));
    learningQNetwork.Add(new ReLULayer<>());
    learningQNetwork.Add(new Linear<>(128, 1));

    FFN<EmptyLoss<>, GaussianInitialization>
        targetQNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
    targetQNetwork = learningQNetwork;
    // targetQNetwork.Add(new Linear<>(3 + 1, 128));
    // targetQNetwork.Add(new ReLULayer<>());
    // targetQNetwork.Add(new Linear<>(128, 1));

    FFN<EmptyLoss<>, GaussianInitialization>
        policyNetwork(EmptyLoss<>(), GaussianInitialization(0, 0.1));
    policyNetwork.Add(new Linear<>(ContinuousActionEnv::State::dimension, 128));
    policyNetwork.Add(new ReLULayer<>());
    policyNetwork.Add(new Linear<>(128, ContinuousActionEnv::Action::size));
    policyNetwork.Add(new TanHLayer<>());

    if (learningQNetwork.Parameters().is_empty())
        learningQNetwork.ResetParameters();
    if (targetQNetwork.Parameters().is_empty())
        targetQNetwork.ResetParameters();
    if (policyNetwork.Parameters().is_empty())
        policyNetwork.ResetParameters();
    targetQNetwork.Parameters() = learningQNetwork.Parameters();

    MeanSquaredError<> lossFunction;

    AdamUpdate qNetworkUpdater;
    typename AdamUpdate::template Policy<arma::mat, arma::mat> *
        qNetworkUpdatePolicy;
    AdamUpdate policyNetworkUpdater;
    typename AdamUpdate::template Policy<arma::mat, arma::mat> *
        policyNetworkUpdatePolicy;
    qNetworkUpdatePolicy = new typename AdamUpdate::template Policy<arma::mat, arma::mat>(qNetworkUpdater,
                                                                                          learningQNetwork.Parameters().n_rows,
                                                                                          learningQNetwork.Parameters().n_cols);
    policyNetworkUpdatePolicy = new typename AdamUpdate::template Policy<arma::mat, arma::mat>(policyNetworkUpdater,
                                                                                               policyNetwork.Parameters().n_rows,
                                                                                               policyNetwork.Parameters().n_cols);
    // Set up the policy and replay method.
    RandomReplay<ContinuousActionEnv> replayMethod(32, 10000);

    Environment env("127.0.0.1", "4040", "BipedalWalker-v3");

    std::vector<double> returnList;
    size_t episodes = 0;
    bool converged = true;
    size_t consecutiveEpisodesTest = 50;
    size_t totalSteps = 0;
    while (true)
    {
        double episodeReturn = 0;
        env.reset();
        do
        {
            ContinuousActionEnv::State state;
            ContinuousActionEnv::Action action;
            state.Data() = env.observation;

            // SELECTING ACTION
            arma::mat outputAction;
            policyNetwork.Predict(state.Encode(), outputAction);
            arma::mat noise = arma::randn<arma::mat>(1) * 0.1;
            noise = arma::clamp(noise, -0.25, 0.25);
            outputAction = outputAction + noise;
            action.action[0] = outputAction[0];

            arma::mat actionEnv = {action.action[0]};

            env.step(actionEnv);
            ContinuousActionEnv::State nextState;
            nextState.Data() = env.observation;

            replayMethod.Store(state, action, env.reward, nextState, env.done, 0.99);
            episodeReturn += env.reward;
            for (size_t i = 0; i < config.UpdateInterval(); i++)
            {
                // Sample from previous experience.
                arma::mat sampledStates;
                std::vector<ContinuousActionEnv::Action> sampledActions;
                arma::rowvec sampledRewards;
                arma::mat sampledNextStates;
                arma::irowvec isTerminal;

                replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
                                    sampledNextStates, isTerminal);

                // Critic network update.

                // Get the actions for sampled next states, from policy.
                arma::mat nextStateActions;
                // std::cout << "sampledNextStates:" << sampledNextStates << std::endl;
                // std::cout << "policyNetwork.Parameters():" << policyNetwork.Parameters() << std::endl;
                policyNetwork.Predict(sampledNextStates, nextStateActions);

                // std::cout << "nextStateActions: " << nextStateActions << std::endl;

                arma::mat targetQInput = arma::join_vert(nextStateActions,
                                                         sampledNextStates);
                // std::cout << "targetQInput: " << targetQInput << std::endl;
                arma::rowvec Q1, Q2;
                // std::cout << "TargetParameter:" << targetQNetwork.Parameters() << std::endl;
                targetQNetwork.Predict(targetQInput, Q1);
                arma::rowvec maskedQ1 = (1 - isTerminal) % Q1;
                arma::rowvec nextQ = sampledRewards + config.Discount() * maskedQ1;

                arma::mat sampledActionValues(action.size, sampledActions.size());
                for (size_t i = 0; i < sampledActions.size(); i++)
                    sampledActionValues.col(i) = sampledActions[i].action[0];
                arma::mat learningQInput = arma::join_vert(sampledActionValues,
                                                           sampledStates);
                learningQNetwork.Forward(learningQInput, Q1);

                arma::mat gradQ1Loss, gradQ2Loss;
                lossFunction.Backward(Q1, nextQ, gradQ1Loss);

                // Update the critic networks.
                arma::mat gradientQ1;
                learningQNetwork.Backward(learningQInput, gradQ1Loss, gradientQ1);

                qNetworkUpdatePolicy->Update(learningQNetwork.Parameters(),
                                             config.StepSize(), gradientQ1);

                arma::mat gradient;
                for (size_t i = 0; i < sampledStates.n_cols; i++)
                {
                    arma::mat grad, gradQ, q;
                    arma::colvec singleState = sampledStates.col(i);
                    arma::colvec singlePi;
                    policyNetwork.Forward(singleState, singlePi);

                    arma::colvec input = arma::join_vert(singlePi, singleState);

                    // std::cout << "input: " << input << std::endl;
                    learningQNetwork.Forward(input, q);
                    learningQNetwork.Backward(input, -1, gradQ);

                    // std::cout << "gradQ:" << gradQ << std::endl;

                    size_t hidden1 = boost::get<mlpack::ann::Linear<> *>(learningQNetwork.Model()[0])->OutputSize();
                    arma::colvec gradQBias = gradQ(input.n_rows * hidden1, 0, arma::size(hidden1, 1));
                    arma::rowvec weightLastLayer = learningQNetwork.Parameters().rows(0, hidden1 - 1).t();
                    arma::mat gradPolicy = weightLastLayer * gradQBias;
                    policyNetwork.Backward(singleState, gradPolicy, grad);
                    if (i == 0)
                    {
                        gradient.copy_size(grad);
                        gradient.fill(0.0);
                    }
                    // std::cout << "grad: " << grad << std::endl;

                    gradient += grad;
                }
                gradient /= sampledStates.n_cols;

                // std::cout << "gradient for policy" << gradient << std::endl;

                policyNetworkUpdatePolicy->Update(policyNetwork.Parameters(),
                                                  config.StepSize(), gradient);
                // std::cout << "policy Parameters:" << policyNetwork.Parameters() << std::endl;

                // Update target network
                targetQNetwork.Parameters() = (1.0 - 0.005) * targetQNetwork.Parameters() +
                                              0.005 * learningQNetwork.Parameters();
            }
        } while (!env.done);
        returnList.push_back(episodeReturn);
        episodes += 1;

        if (returnList.size() > consecutiveEpisodesTest)
            returnList.erase(returnList.begin());

        double averageReturn = std::accumulate(returnList.begin(),
                                               returnList.end(), 0.0) /
                               returnList.size();

        std::cout << "Average return in last " << returnList.size()
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

    return 0;
}