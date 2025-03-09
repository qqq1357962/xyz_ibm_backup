#pragma once

#include <set>

namespace db {
class Via {
public:
    int x;
    int y;
    ViaType* type;
    Via(ViaType* type, int x, int y) : x(x), y(y), type(type) {}
};

class ViaType {
private:
    bool isDef_ = false;

public:
    string name = "";
    set<Geometry> rects;

    ViaType(const string& name = "", const bool isDef = false) : isDef_(isDef), name(name) {}

    template <class... Args>
    void addRect(Args&&... args) {
        rects.emplace(args...);
    }
    void isDef(bool b) { isDef_ = b; }
    bool isDef() const { return isDef_; }
};
} // namespace db
