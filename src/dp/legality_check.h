#pragma once

#include "detailed_place_db.h"
#include "global.h"

namespace dp {

bool boundaryCheck(const DetailedPlaceData& db, float scale_factor, int num_movable_nodes);

bool siteAlignmentCheck(const DetailedPlaceData& db, float scale_factor, int num_movable_nodes);

bool overlapCheck(const DetailedPlaceData& db, float scale_factor, int num_nodes, int num_movable_nodes);

bool legalityCheckKernelCPU(const DetailedPlaceData& db, float scale_factor, int num_nodes, int num_movable_nodes);

bool legalityCheck_main(const DetailedPlaceData& db);

} // namespace dp