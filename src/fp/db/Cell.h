#pragma once
#include "Database.h"

namespace fp {

class Cell {
private:
    string _name;

public:
    int id = -1;
    Point* pos;


    Cell(const string& name = "");
    ~Cell();

    const std::string& name() const { return _name; }

    friend ostream& operator<<(ostream& os, const Cell& c) {
        return os << c._name << "\t(" << ')';
    }
};

}