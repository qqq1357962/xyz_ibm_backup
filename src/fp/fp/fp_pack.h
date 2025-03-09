#pragma once
#include "global.h"
#include "fp/db/Database.h"

#include "b_star_tree.h"
#include "contour.h"

namespace fp {

class Packer {
public:
    
    Packer(fp::Database* database_);
    fp::Database* database;

    Contour contour;

    int root;
    
    struct drawContent{
        int lx,ly,ux,uy;
        int id;
        drawContent(int lx,int ly, int ux,int uy,int id)
        {
            this->lx=lx;this->ly=ly;this->ux=ux;this->uy=uy;this->id=id;
        }
    };

    vector<drawContent> drawList;

    int num_macros() const;
    // double width() const;
    double width1() const;
    double width2() const;
    // double height() const;
    double height1() const;
    double height2() const;
    double area() const;
    double wirelength() const;
    const pair<fp::Point, fp::Point>& macro_bounding_box(int macro_id) const;
    vector<fp::pin> macro_pins;

    void Perturb();
    void PackInt();
    void print();
    void PrintMacroPins();

    void write(const string& output_path);

    vector<int> macro_id_by_node_id_;//布局
    vector<int> macro_die_;//布局
    BStarTree b_star_tree_;
    vector<bool> is_macro_rotated_by_id_;
    vector<bool> is_macro_rotatable_by_id_;
    vector<pair<fp::Point, fp::Point>> macro_bounding_box_by_id_;
    void traverseTree(int cur_node, bool left) ;
    void traverseTree_vis() ;
    double not_overlap_ratio;

private:
    double area_;
    // double width_;
    double width1_;
    double width2_;
    // double height_;
    double height1_;
    double height2_;
    double wirelength_;
    
};

}