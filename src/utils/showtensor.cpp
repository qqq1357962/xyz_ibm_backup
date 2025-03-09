#include "showtensor.h"

void showtensor(torch::Tensor t)
{
    std::cout << "===================================" << std::endl;
    std::cout << t << std::endl;
    std::cout << "===================================" << std::endl;
}

void showtensor(torch::Tensor t, int from, int to)
{
    std::cout << "===================================" << std::endl;
    std::cout << t.slice(0, from, to) << std::endl;
    std::cout << "===================================" << std::endl;
}