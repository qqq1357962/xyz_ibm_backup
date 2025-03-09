#pragma once

#include "global.h"
#include "fp/db/Database.h"

namespace fp {

class Node {
public:
    Node(int parent_id, int left_child_id, int right_child_id);
    int id;
    int parent_id_;
    int left_child_id_;
    int right_child_id_;
};

class BStarTree {
public:
    BStarTree(fp::Database* database_);
    fp::Database* database;

    void Print(std::ostream& os = std::cout, int indent_level = 0) const;

    int num_nodes() const;
    //int root_id() const;
    int root_id1() const;
    int root_id2() const;
    int parent_id(int node_id) const;
    int left_child_id(int node_id) const;
    int right_child_id(int node_id) const;


    void DeleteAndInsert(int deleted_node_id, int target_node_id,
                        std::pair<int, int> inserted_positions);

    std::vector<Node> nodes_;
    int root_id1_;
    int root_id2_;
    const Node& node(int node_id) const;
    Node& node(int node_id);
    void Delete(int deleted_node_id);
    void Insert(int inserted_node_id, int target_node_id,
                std::pair<int, int> inserted_positions);
private:


    
};

}