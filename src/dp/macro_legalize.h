#pragma once

#include "detailed_place_db.h"
#include "global.h"

namespace dp {
/*
template <typename T>
struct Interval {
    T xl;
    T xh;

    Interval(T l, T h) : xl(l), xh(h) {}

    void intersect(T rhs_xl, T rhs_xh) {
        xl = std::max(xl, rhs_xl);
        xh = std::min(xh, rhs_xh);
    }
};
template <typename T>
struct Blank {
    T xl;
    T yl;
    T xh;
    T yh;

    void intersect(const Blank& rhs) {
        xl = std::max(xl, rhs.xl);
        xh = std::min(xh, rhs.xh);
        yl = std::max(yl, rhs.yl);
        yh = std::min(yh, rhs.yh);
    }
};
*/
bool macroLegalization(NodeData& data, DetailedPlaceData& db, int num_bins_x, int num_bins_y);

}  // namespace dp