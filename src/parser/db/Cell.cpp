#include "Database.h"
using namespace db;

/***** Cell *****/

Cell::~Cell() {
    for (Pin* pin : _pins) {
        delete pin;
    }
    _pins.clear();
}

Pin* Cell::pin(const string& name) const {
    for (Pin* pin : _pins) {
        if (pin->type->name() == name) {
            return pin;
        }
    }
    return nullptr;
}

void Cell::ctype(CellType* t) {
    if (!t) {
        return;
    }
    if (_type) {
        printlog(LOG_ERROR, "type of cell %s already set", _name.c_str());
        return;
    }
    _type = t;
    ++(_type->usedCount);
    _pins.resize(_type->pins.size(), nullptr);
    for (unsigned i = 0; i != _pins.size(); ++i) {
        _pins[i] = new Pin(this, i);
    }
}

int Cell::lx() const { return _lx; }
int Cell::ly() const { return _ly; }
int Cell::orient() const { return _orient; }
bool Cell::flipX() const { return _flipX; }
bool Cell::flipY() const { return _flipY; }
bool Cell::placed() const { return (lx() != INT_MIN) && (ly() != INT_MIN); }
// int Cell::siteWidth() const { return width() / database.siteW; }
// int Cell::siteHeight() const { return height() / database.siteH; }

void Cell::place(int x, int y) {
    if (_fixed) {
        printlog(LOG_WARN, "moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
}

void Cell::rotate(int orient) {
    if (_fixed) {
        printlog(LOG_WARN, "rotate fixed cell %s to %d", _name.c_str(), orient);
    }
    _orient = orient;
}

void Cell::place(int x, int y, int dieId) {
    if (_fixed) {
        printlog(LOG_WARN, "moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
    _dieId = dieId;
}

void Cell::place_with_orient(int x, int y, int dieId, int cell_orient) {
    if (_fixed) {
        printlog(LOG_WARN, "moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
    _dieId = dieId;
    _orient = cell_orient;
}

void Cell::place(int x, int y, bool flipX, bool flipY) {
    if (_fixed) {
        printlog(LOG_WARN, "moving fixed cell %s to (%d,%d)", _name.c_str(), x, y);
    }
    _lx = x;
    _ly = y;
    _flipX = flipX;
    _flipY = flipY;
}

void Cell::unplace() {
    if (_fixed) {
        printlog(LOG_WARN, "unplace fixed cell %s", _name.c_str());
    }
    _lx = _ly = INT_MIN;
    _flipX = _flipY = false;
}

/***** Cell Type *****/

CellType::~CellType() {
    for (PinType* pin : pins) {
        delete pin;
    }
}

PinType* CellType::addPin(const string& name, const char direction, const char type) {
    PinType* newpintype = new PinType(name, direction, type);
    newpintype->pintype_id = pins.size();
    pins.push_back(newpintype);
    return newpintype;
}

PinType* CellType::getPin(string& name) {
    for (int i = 0; i < (int)pins.size(); i++) {
        if (pins[i]->name() == name) {
            return pins[i];
        }
    }
    return nullptr;
}

void CellType::addType(CellType* celltype){
    _type.push_back(celltype);
}

CellType * CellType::getType(int id){
    if (id > 3) {
        return nullptr;
    }
    return _type[id];
}

void CellType::setOrigin(int x, int y) {
    _originX = x;
    _originY = y;
}

bool CellType::operator==(const CellType& r) const {
    if (width != r.width || height != r.height) {
        return false;
    } else if (_originX != r.originX() || _originY != r.originY() || _symmetry != r.symmetry() ||
               pins.size() != r.pins.size()) {
        return false;
    } else if (edgetypeL != r.edgetypeL || edgetypeR != r.edgetypeR) {
        return false;
    } else {
        //  return PinType::comparePin(pins, r.pins);
        for (unsigned i = 0; i != pins.size(); ++i) {
            if (*pins[i] != *r.pins[i]) {
                return false;
            }
        }
    }
    return true;
}
