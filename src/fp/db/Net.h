#pragma once
#include "Database.h"


namespace fp {

struct pin
{   
    pin() {}
    pin(int min_x, int min_y)
        : min_x(min_x), min_y(min_y) {}
    pin(int min_x, int min_y, int max_x, int max_y) 
        : min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y) {}
    int min_x = INT_MAX;
    int min_y = INT_MAX;
    int max_x = 0;
    int max_y = 0;
};

class Net {
private:
    string _name;

public:
    int min_x_;
    int min_y_;
    int max_x_;
    int max_y_;
    int id = -1;
    bool cut = false;
    vector<Cell*> cell_list;
    vector<Macro*> macro_list;

    vector<int> macro_list_by_id;


    Net(const std::string& name = "") : _name(name) { }
    ~Net();

    const std::string& name() const { return _name; }

    void init_bx();

    int ComputeWirelength(
        std::vector<std::pair<Point, Point>> macro_bounding_box_by_id) const;

    friend ostream& operator<<(ostream& os, const Net& c) {
        return os << c._name << "\n";
    }
};

}
