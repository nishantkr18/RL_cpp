#include <iostream>
#include <math.h>
#include <armadillo>
#include <queue>
#include <cassert>

int main()
{
    arma::mat a = {1,2,5,5};
    std::cout << arma::accu(a) << std::endl;
}