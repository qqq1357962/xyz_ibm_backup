#include "Net.h"

using namespace fp;

void Net::init_bx()
{
    min_x_ = numeric_limits<int>::max();
    min_y_ = numeric_limits<int>::max();
    max_x_ = numeric_limits<int>::lowest();
    max_y_ = numeric_limits<int>::lowest();
    for (Cell* cell : cell_list) {
        const int terminal_x = cell->pos->x();
        const int terminal_y = cell->pos->y();
        if (terminal_x < min_x_) {
            min_x_ = terminal_x;
        }
        if (terminal_y < min_y_) {
            min_y_ = terminal_y;
        }
        if (terminal_x > max_x_) {
            max_x_ = terminal_x;
        }
        if (terminal_y > max_y_) {
            max_y_ = terminal_y;
        }
    }
}

int Net::ComputeWirelength(
    vector<pair<Point, Point>> macro_bounding_box_by_id) const {
        
    int min_x = min_x_;
    int min_y = min_y_;
    int max_x = max_x_;
    int max_y = max_y_;
    //for (Macro* macro : macro_list) {
        //auto bounding_box = macro_bounding_box_by_id.at(macro->id);
    for (int macro_id : macro_list_by_id) {
        auto bounding_box = macro_bounding_box_by_id.at(macro_id);
        Point center = Point::Center(bounding_box.first, bounding_box.second);
        int x = center.x();
        int y = center.y();
        if (x < min_x) {
            min_x = x;
        }
        if (x > max_x) {
            max_x = x;
        }
        if (y < min_y) {
            min_y = y;
        }
        if (y > max_y) {
            max_y = y;
        }
    }
    
    return Point::HPWL(Point(min_x, min_y), Point(max_x, max_y));
}