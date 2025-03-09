#include"myUtils.h"
#include "placer/database.h"

torch::Tensor load_from_file(std::string path, NodeData& data) {
    printlog(LOG_INFO, "start parsing dataset file: %s", path.c_str());
    std::ifstream infile(path);
    if (infile.is_open() == false) {
        printlog(LOG_ERROR, "Cannot open input file!");
        exit(1);
    }

    // start parser
    string topPlacement;
    string bottomPlacement;
    int numTopPlacement;
    int numBottomPlacement;
    infile >> topPlacement >> numTopPlacement;
    auto new_orient = data.node_orient_top.clone();
    for(int i=0;i<numTopPlacement;i++)
    {
        string tmp;
        infile>>tmp;
        // string cellName;
        // infile>>cellName;
        char C;
        int c_id;
        infile>>C>>c_id;
        c_id--;
        float x,y;
        infile>>x>>y;
        char R;
        infile>>R;
        int orientation;
        infile>>orientation;
        orientation = orientation/90;
        data.node_pos[c_id][0] = x;
        data.node_pos[c_id][1] = y;
        new_orient[c_id] = orientation;
        data.node_die[c_id] = 1;
    }
    infile >> bottomPlacement >> numBottomPlacement;
    for(int i=0;i<numBottomPlacement;i++)
    {
        string tmp;
        infile>>tmp;
        // string cellName;
        // infile>>cellName;
        char C;
        int c_id;
        infile>>C>>c_id;
        c_id--;
        float x,y;
        infile>>x>>y;
        char R;
        infile>>R;
        int orientation;
        infile>>orientation;
        orientation = orientation/90;
        if(orientation!=0){
            int debuggggg=0;
        }
        data.node_pos[c_id][0] = x;
        data.node_pos[c_id][1] = y;
        new_orient[c_id] = orientation;
        data.node_die[c_id] = 0;
    }

    data.update_macro_orientation(new_orient);

    for(int i=0;i<data.cell_mov_rhs;i++)
    {
        if(data.node_die[i].item<int>()==1)
        {
            data.node_pos[i] += data.node_size_top[i]/2;
        }
        else{
            data.node_pos[i] += data.node_size_bot[i]/2;
        }
    }
    return data.node_pos;
}