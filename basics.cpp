#include <iostream>
using namespace std;

class person
{
public:
    string name;
    person() : name()
    {
        std::cout << "Default Constructor" << std::endl;
    }
    person(string name) : name(name)
    {
        std::cout << "Parameterized constructor of " << name << std::endl;
    }
    person(const person &p) : name(p.name)
    {
        std::cout << "Copy Constructor" << std::endl;
    }
    void operator=(const person &p)
    {
        name = p.name;
        std::cout << "Assigment operator" << std::endl;
    }
    ~person()
    {
        // std::cout << "Destructor containing " << name << std::endl;
    }
};

class bus
{
public:
    person &a;
    bus(person &a) : a(a)
    {
    }
};
class Action
{
public:
    enum actions
    {
        negativeTorque,
        zeroTorque,
        positiveTorque,
    };
    // To store the action.
    Action::actions action;

    // Track the size of the action space.
    static const size_t size = 3;
};
int main()
{
    using E = typename Action::actions;
    Action o;
    o.action = static_cast<(typename Action::actions)>(Action::size);

    // std::cout << o.size << std::endl;

    person a("a");

    // person b(a); // Copy constructor called
    // b(a); // error: no match for call to ‘(person) (person&)’

    bus b1(a);

    // person b = a; // Copy constructor called

    // person b; // Default Constructor called
    // b = a;    // assignment operator called
    // b = std::move(a); // first move, then assignment opertor called

    std::cout << a.name << std::endl;
    std::cout << b1.a.name << std::endl;
    // std::cout << b.name << std::endl;
}