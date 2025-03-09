#include "contour.h"

using namespace fp;

Contour::Contour() : max_y_(0.0), coordinates_(1, fp::Point(0.0, 0.0)) {}

void Contour::Print(ostream& os, int indent_level) const {
    const int indent_size = 2;
    os << string(indent_level * indent_size, ' ') << "Contour:" << endl;
    ++indent_level;
    os << string(indent_level * indent_size, ' ') << "max_y_: " << max_y_ << endl;
    os << string(indent_level * indent_size, ' ') << "coordinates_:" << endl;
    ++indent_level;
    for (const fp::Point& point : coordinates_) {
        point.Print(os, indent_level);
    }
}

double Contour::max_x() const { return coordinates_.back().x(); }

double Contour::max_y() const { return max_y_; }


