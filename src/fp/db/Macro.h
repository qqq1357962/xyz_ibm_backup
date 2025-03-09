#pragma once
#include "Database.h"

namespace fp {

class Macro {
private:
    string _name;
    int _width[2];
    int _height[2];

public:
    int id = -1;
    int isFromSTD = 0;
    int rectNum = 0;
    bool rotated = 0;
    vector<int> rects;

    Macro(const string& name = "") : _name(name) {}
    ~Macro();

    const std::string& name() const { return _name; }

    // void init();

    int width(int die_id) {return _width[die_id];}
    int height(int die_id) {return _height[die_id];}

    void setWidth(int width0, int width1) {this->_width[0]=width0;this->_width[1]=width1;}
    void setHeight(int height0, int height1) {this->_height[0]=height0;this->_height[1]=height1;}

    friend ostream& operator<<(ostream& os, const Macro& c) {
        return os << c._name << "\t(" << ')';
    }
};

}