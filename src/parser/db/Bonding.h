#pragma once

#include <set>

namespace db {
class Bonding {
private:
    int _lx = std::numeric_limits<int>::min();
    int _ly = std::numeric_limits<int>::min();
    int _width = 0;
    int _height = 0;
    int _netId = -1;

public:
    bool valid = false;
    Bonding(int lx, int ly, int width, int height, int netId)
        : _lx(lx), _ly(ly), _width(width), _height(height), _netId(netId) {}
    void place(int lx, int ly) {
        _lx = lx;
        _ly = ly;
        valid = true;
    }
    void remove() {
        int _lx = std::numeric_limits<int>::min();
        int _ly = std::numeric_limits<int>::min();
        valid = false;
    }
    int netId() const { return _netId; }
    int lx() const { return _lx; }
    int ly() const { return _ly; }
    int width() const { return _width; }
    int height() const { return _height; }
    int hx() const { return lx() + width(); }
    int hy() const { return ly() + height(); }
    int cx() const { return lx() + width() / 2; }
    int cy() const { return ly() + height() / 2; }
};
}  // namespace db
