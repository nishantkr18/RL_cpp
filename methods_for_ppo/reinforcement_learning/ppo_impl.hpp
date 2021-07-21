/**
 * @file ppo.hpp
 * @author Xiaohong Ji
 *
 * This file is the implementation of PPO class, which implements
 * proximal policy optimization algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RL_PPO_IMPL_HPP
#define MLPACK_METHODS_RL_PPO_IMPL_HPP

#include <mlpack/prereqs.hpp>

#include "ppo.hpp"

namespace mlpack {
namespace rl {

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::PPO(TrainingConfig config,
       ActorNetworkType actor,
       CriticNetworkType critic,
       PolicyType policy,
       ReplayType replayMethod,
       UpdaterType updater,
       EnvironmentType environment):
  config(std::move(config)),
  actorNetwork(std::move(actor)),
  criticNetwork(std::move(critic)),
  actorUpdater(std::move(updater)),
  #if ENS_VERSION_MAJOR >= 2
  actorUpdatePolicy(NULL),
  #endif
  criticUpdater(std::move(updater)),
  #if ENS_VERSION_MAJOR >= 2
  criticUpdatePolicy(NULL),
  #endif
  policy(std::move(policy)),
  replayMethod(std::move(replayMethod)),
  environment(std::move(environment)),
  totalSteps(0),
  deterministic(false)
{
  // Set up actor and critic network.
  if (actorNetwork.Parameters().is_empty()){
    actorNetwork.ResetParameters();
    std::cout << "Initialize actor: " << '\n';
  }

  if (criticNetwork.Parameters().is_empty()){
    criticNetwork.ResetParameters();
    std::cout << "Initialize critic: " << '\n';

  }

  #if ENS_VERSION_MAJOR == 1
  this->criticUpdater.Initialize(criticNetwork.Parameters().n_rows,
                                 criticNetwork.Parameters().n_cols);
  #else
  this->criticUpdatePolicy = new typename UpdaterType::template
  Policy<arma::mat, arma::mat>(this->criticUpdater,
                               criticNetwork.Parameters().n_rows,
                               criticNetwork.Parameters().n_cols);
  #endif

  #if ENS_VERSION_MAJOR == 1
  this->actorUpdater.Initialize(actorNetwork.Parameters().n_rows,
                                actorNetwork.Parameters().n_cols);
  #else
  this->actorUpdatePolicy = new typename UpdaterType::template
  Policy<arma::mat, arma::mat>(this->actorUpdater,
                               actorNetwork.Parameters().n_rows,
                               actorNetwork.Parameters().n_cols);
  #endif

  oldActorNetwork = actorNetwork;
  oldActorNetwork.ResetParameters();
  actorNetwork.ResetParameters();
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::~PPO()
{
  #if ENS_VERSION_MAJOR >= 2
  delete actorUpdatePolicy;
  delete criticUpdatePolicy;
  #endif
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
void PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Update()
{
  arma::mat sampledStates;
  std::vector<ActionType> sampledActions;
  arma::colvec sampledRewards;
  arma::mat sampledNextStates;
  arma::icolvec isTerminal;

  replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
                      sampledNextStates, isTerminal);


  for (size_t step = 0; step < config.ActorUpdateStep(); step ++) {
    // Update the oldActorNetwork, synchronize the parameter.
    // std::cout << "oldActorNetwork.Parameters()[0]: " << oldActorNetwork.Parameters()[0] << '\n';
    // std::cout << "actorNetwork.Parameters()[0]: " << actorNetwork.Parameters()[0] << '\n';
    oldActorNetwork.Parameters() = actorNetwork.Parameters();
    // std::cout << "oldActorNetwork.Parameters()[0]: " << oldActorNetwork.Parameters()[0] << '\n';
    // std::cout << "actorNetwork.Parameters()[0]: " << actorNetwork.Parameters()[0] << '\n';

    arma::rowvec discountedRewards(sampledRewards.n_rows);
    
    arma::mat nextActionValue;
    criticNetwork.Predict(state.Encode(), nextActionValue);
    // std::cout << "state.Encode(): " << state.Encode() << '\n';
    // std::cout << "sampledNextStates[31].Encode(): " << sampledNextStates << '\n';
    // std::cout << "sampledNextStates[32].Encode(): " << sampledNextStates[32] << '\n';
    // double values = nextActionValue[0];
    double values = 0;
    // std::cout << "values: " << values << '\n';
    // if(values>10.0)
    //   {std::cout << "values: " << values << '\n';}
    for (int i = sampledRewards.n_rows - 1; i >= 0; --i)
    {
      // std::cout << "i: " << i << '\n';
      values = sampledRewards[i] + values * config.Discount();
      discountedRewards[i] = values;
    }
    // std::cout << "discountedRewards: " << discountedRewards << '\n';

    arma::mat actionValues, advantages, criticGradients, actorGradients;
    criticNetwork.Forward(sampledStates, actionValues);

    advantages = arma::conv_to<arma::mat>::
                 from(discountedRewards) - actionValues;

    // Update the critic.
    criticNetwork.Backward(sampledStates, advantages, criticGradients);
    #if ENS_VERSION_MAJOR == 1
    criticUpdater.Update(criticNetwork.Parameters(), config.StepSize(),
                         criticGradients);
    #else
    criticUpdatePolicy->Update(criticNetwork.Parameters(), config.StepSize(),
                               criticGradients);
    #endif




    // calculate the ratio.
    arma::mat actionParameter, sigma, mu;
    // std::cout << "sampledStates: " << sampledStates << '\n';
    oldActorNetwork.Predict(sampledStates, actionParameter);
    // std::cout << "OLDactionParameter: " << actionParameter << '\n';
    ann::TanhFunction::Fn(actionParameter.row(0), mu);
    ann::SoftplusFunction::Fn(actionParameter.row(1), sigma);
    sigma += 0.001;
    mu = mu * 2;
    // std::cout << "OLDmu: " << mu << '\n';
    // std::cout << "OLDsigma: " << sigma << '\n';
    ann::NormalDistribution oldNormalDist =
      ann::NormalDistribution(vectorise(mu, 0), vectorise(sigma, 0));

    actorNetwork.Forward(sampledStates, actionParameter);
    // std::cout << "NEWactionParameter: " << actionParameter << '\n';
    ann::TanhFunction::Fn(actionParameter.row(0), mu);
    ann::SoftplusFunction::Fn(actionParameter.row(1), sigma);
    mu = mu * 2;
    sigma += 0.001;
    // std::cout << "NEWmu: " << mu << '\n';
    // std::cout << "NEWsigma: " << sigma << '\n';
    ann::NormalDistribution normalDist =
      ann::NormalDistribution(vectorise(mu, 0), vectorise(sigma, 0));

    // Update the actor.
    // observation use action.
    arma::vec prob, oldProb;
    arma::colvec observation(sampledActions.size());
    for (size_t i = 0; i < sampledActions.size(); i++)
    {
      observation[i] = sampledActions[i].action[0];
    }
    // std::cout << "observation: " << observation.t() << '\n';
    normalDist.LogProbability(observation, prob);
    oldNormalDist.LogProbability(observation, oldProb);
    // std::cout << "prob: " << prob.t() << '\n';
    // std::cout << "oldProb: " << oldProb.t() << '\n';

    arma::mat ratio = arma::exp((prob - oldProb).t());
    // std::cout << "ratio " << ratio << '\n';

    arma::mat L1 = ratio % advantages;

    arma::mat L2 = arma::clamp(ratio, 1 - config.Epsilon(),
                              1 + config.Epsilon()) % advantages;
    arma::mat surroLoss = - arma::min(L1, L2);
    // std::cout << "actorNetwork.Parameters()[0]: " << actorNetwork.Parameters()[0] << '\n';
    // std::cout << "oldActorNetwork.Parameters()[0]: " << oldActorNetwork.Parameters()[0] << '\n';

    // std::cout << "surroLoss: " << surroLoss << '\n';

    // std::cout << "advantages: " << advantages << '\n';
    // std::cout << "L1: " << L1 << '\n';
    // std::cout << "L2: " << L2 << '\n';

    // backward the gradient
    arma::mat dL1 = (L1 < L2) % advantages;
    arma::mat dL2 = (L1 >= L2) % (ratio >= (1 - config.Epsilon())) %
                        (ratio <= (1 + config.Epsilon())) % advantages;
    // std::cout << "dL1: " << dL1 << '\n';
    // std::cout << "dL2: " << dL2 << '\n';
    arma::mat dSurroLoss = -(dL1 + dL2);
    // std::cout << "dSurroLoss: " << dSurroLoss << '\n';

    arma::mat dmu = (observation.t() - mu) / (arma::square(sigma)) % ratio;
    arma::mat dsigma = (- 1.0 / sigma +
                       (arma::square(observation.t() - mu) / arma::pow(sigma, 3)))
                        % ratio;
    // std::cout << "dmu: " << dmu << '\n';
    // std::cout << "dsigma: " << dsigma << '\n';
    arma::mat dTanh, dSoftP;
    ann::TanhFunction::Deriv(mu/2.0, dTanh);
    ann::SoftplusFunction::Deriv(actionParameter.row(1), dSoftP);

    arma::mat dLoss1, dLoss2;
    dLoss1 = 2 * dTanh % dmu % dSurroLoss;
    dLoss2 = dSoftP % dsigma % dSurroLoss;

    arma::mat dLoss = arma::join_cols(dLoss1, dLoss2);

    for(int i = 0; i < surroLoss.n_cols; i++)
    {
      // std::cout << "i: " << i << '\n';
      if(std::isnan(surroLoss[i]) || std::isinf(surroLoss[i])){
        std::cout << "NAN ERROR IN SURROLOSS!" << '\n';
        std::cout << "observation: " << observation.t() << '\n';
        std::cout << "prob: " << prob.t() << '\n';
        std::cout << "oldProb: " << oldProb.t() << '\n';
        std::cout << "(prob - oldProb).t(): " << (prob - oldProb).t() << '\n';
        std::cout << "ratio: " << ratio << '\n';
        std::cout << "advantages: " << advantages << '\n';
        std::cout << "dSurroLoss: " << dSurroLoss << '\n';
        std::cout << "dmu: " << dmu << '\n';
        std::cout << "dsigma: " << dsigma << '\n';
        std::cout << "dTanh: " << dTanh << '\n';
        std::cout << "dSoftP: " << dSoftP << '\n';
        std::cout << "dLoss: " << dLoss << '\n';
        exit(0);
      }
    }
    // std::cout << "observation.t(): " << observation.t() << '\n';
    // std::cout << "mu: " << mu << '\n';
    // std::cout << "sigma: " << sigma << '\n';
    // std::cout << "ratio: " << ratio << '\n';
    // std::cout << "dSurroLoss: " << dSurroLoss << '\n';
    // std::cout << "dmu: " << dmu << '\n';
    // std::cout << "dsigma: " << dsigma << '\n';
    // std::cout << "dmu * dSurroLoss: " << dmu % dSurroLoss << '\n';
    // std::cout << "dsigma * dSurroLoss: " << dsigma % dSurroLoss << '\n';
    actorNetwork.Backward(sampledStates, dLoss, actorGradients);
    // std::cout << "======================================================================" << '\n';

    #if ENS_VERSION_MAJOR == 1
    actorUpdater.Update(actorNetwork.Parameters(), config.StepSize(),
                        actorGradients);
    #else
    actorUpdatePolicy->Update(actorNetwork.Parameters(), config.StepSize(),
                               actorGradients);
    #endif
  }
  replayMethod.Clear();

}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
double PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Step()
{
  // Get the action value for each action at current state.
  arma::mat actionParameter, sigma, mu;

//  std::cout << "state: " << state.Encode() << std::endl;
      // std::cout << "actorNetwork.Parameters()[0]: " << actorNetwork.Parameters()[0] << '\n';

  actorNetwork.Predict(state.Encode(), actionParameter);
  for(int i=0; i<actionParameter.n_rows; i++){
    if(std::isnan(actionParameter[i])){
      std::cout << "NAN Error in ActionParameter!" << '\n';
      std::cout << "state.Encode(): " << state.Encode() << "actionParameter" << actionParameter << '\n';
      exit(0);
    }
  }
  // std::cout << "actionParameter: " << actionParameter << '\n';

  ann::TanhFunction::Fn(actionParameter.row(0), mu);
  ann::SoftplusFunction::Fn(actionParameter.row(1), sigma);

  mu *= 2;
  sigma += 0.001;

 // std::cout << "mu sigma: "<< mu << sigma << std::endl;

  ann::NormalDistribution normalDist =
    ann::NormalDistribution(vectorise(mu, 0), vectorise(sigma, 0));

  ActionType action;
  action.action[0] = normalDist.Sample()[0];

 // std::cout << "action: " << action.action[0] << std::endl;

  // Interact with the environment to advance to next state.
  StateType nextState;
  double reward = environment.Sample(state, action, nextState);
  reward = (reward + 8.1)/8.1;
 // std::cout << "reward: " << reward << std::endl;

  // Store the transition for replay.
  replayMethod.Store(state, action, reward, nextState,
      environment.IsTerminal(nextState));

  // Update current state.
  state = nextState;

  return (reward * 8.1)-8.1;
}

template<
  typename EnvironmentType,
  typename ActorNetworkType,
  typename CriticNetworkType,
  typename UpdaterType,
  typename PolicyType,
  typename ReplayType
>
double PPO<
  EnvironmentType,
  ActorNetworkType,
  CriticNetworkType,
  UpdaterType,
  PolicyType,
  ReplayType
>::Episode()
{
  // Get the initial state from environment.
  state = environment.InitialSample();

  replayMethod.Clear();

  // Track the steps in this episode.
  size_t steps = 0;

  // Track the return of this episode.
  double totalReturn = 0.0;

  // Running until get to the terminal state.
  while (!environment.IsTerminal(state))
  {
    if (config.StepLimit() && steps >= config.StepLimit())
      break;

    totalReturn += Step();
    if(std::isnan(totalReturn))
      {std::cout << "NAN in step!: " << '\n'; exit(0);}

    steps++;
    totalSteps++;

    if (deterministic)
      continue;

    if (steps % config.UpdateInterval() == 0 || steps == 199)
    {
      Update();
    }
  }

  return totalReturn;
}

} // namespace rl
} // namespace mlpack
#endif
