#pragma once
#include "global.h"
#include "fp/db/Database.h"

namespace fp {

class Contour {
public:
    Contour();
    Contour(const int width) {box.resize(width * 2);}

    vector<int> box;
    int box_x_max = 0;
    int box_y_max = 0;

    void Print(std::ostream& os = std::cout, int indent_level = 0) const;

    double max_x() const;
    double max_y() const;

    std::pair<fp::Point, fp::Point> Update(double x, double width, double height);

private:

    double max_y_;
    std::list<fp::Point> coordinates_;

};

}