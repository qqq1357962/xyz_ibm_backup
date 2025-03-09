#pragma once
#include "Database.h"

namespace fp {

class Point {
private:
    int x_ = 0;
    int y_ = 0;

public:
    static int HPWL(const Point& point_a, const Point& point_b);
    static Point Center(const Point& point_a, const Point& point_b);

    Point();
    Point(int x, int y);

    void Print(std::ostream& os = std::cout, int indent_level = 0) const;

    int x() const;
    int y() const;
    
    void set_pos(int pos_x, int pos_y);
};

}