
#include "Cell.h"
#include "Database.h"

using namespace fp;


Cell::Cell(const string& name){
    _name = name;
    pos = new Point();
}