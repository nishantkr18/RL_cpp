/**
 * @file normal_distribution_impl.hpp
 * @author xiaohong ji
 *
 * Implementation of the Normal distribution class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_IMPL_HPP
#define MLPACK_METHODS_ANN_DISTRIBUTIONS_NORMAL_DISTRIBUTION_IMPL_HPP

// In case it hasn't yet been included.
#include "normal_distribution.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

NormalDistribution::NormalDistribution()
{
  // Nothing to do here.
}

NormalDistribution::NormalDistribution(
    const arma::vec& mu,
    const arma::vec& sigma) :
    mu(mu),
    sigma(sigma)
{
}

arma::vec NormalDistribution::Sample() const
{
  return sigma * arma::randn<arma::vec>(mu.n_elem) + mu;
}

arma::vec NormalDistribution::LogProbability(
    const arma::vec& observation) const
{
  const arma::vec variance = arma::square(sigma);
  arma::vec v1 = arma::log(sigma) + std::log(std::sqrt(2 * pi));
  arma::vec v2 = arma::square(observation - mu) / (2 * variance);
  arma::mat q = (-v1 - v2).t();
  for (int i= 0; i<q.n_cols; i++)
    if(std::isnan(q[i]) || std::isnan(q[i]))
    {
      std::cout << "nan in LogProb of normal_distribution " << q[i] << '\n';
      std::cout << "sigma: " << sigma << '\n';
      std::cout << "mu: " << mu << '\n';
      std::cout << "v1: " << v1 << '\n';
      std::cout << "v2: " << v2 << '\n';
      exit(0);
    }
  return  (-v1 - v2);
}

} // namespace ann
} // namespace mlpack

#endif
