#include "boost/variant.hpp"
#include <iostream>

class Apple
{
public:
    int a;
    Apple(int a) : a(a) {}
    void appleFunc()
    {
        std::cout << "HEY" << std::endl;
    }
};

class Mango
{
public:
    int a;
    Mango(int a) : a(a) {}
    void mangoFunc()
    {
        std::cout << "HEY" << std::endl;
    }
};

class my_visitor : public boost::static_visitor<double>
{
public:
    int operator()(int i) const
    {
        return i;
    }

    int operator()(const std::string &str) const
    {
        return str.length();
    }
    double operator()(Apple* a) const
    {
        a->appleFunc();
        return 5.55;
    }
    double operator()(Mango* a) const
    {
        a->mangoFunc();
        return 6.66;
    }
};

int main()
{
    using LayerType = boost::variant<std::string, double, Apple *, Mango*>;
    LayerType u(new Mango(123));
    std::cout << u << '\n'; // output: hello world

    boost::get<Mango*>(u)->mangoFunc();

    // double result = boost::apply_visitor(my_visitor(), u);
    // std::cout << result; // output: 11 (i.e., length of "hello world")
}