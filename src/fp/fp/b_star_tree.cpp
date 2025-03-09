#include "b_star_tree.h"

using namespace fp;

//---------------------------------------------------------------------

BStarTree::BStarTree(fp::Database* database_)
    :   database(database_),
        //root_id_(0), 
        nodes_(database_->nMacros, Node(-1, -1, -1))
{
    //root_id1_ = 0;
    //root_id2_ = database_->nMacros1;
    /* tobe implemented in fp_pack
    int num_macros = database_->nMacros;
    vector<int> inserted(num_macros, 0);
    root_id_ = rand() % num_macros;
    inserted[root_id_] = 1;
    
    int row_node = root_id_;
    int col_node = root_id_;
    int width = database->macros[root_id_]->width();
    int inserted_cnt = 1;

    while (inserted_cnt != num_macros) {
        int node;
        do {
            node = rand() % num_macros;
        } while (inserted[node]);
        if (width > database->outline_width / 2 ) {
            nodes_[node].parent_id_= row_node;
            nodes_[row_node].right_child_id_ = node;
            row_node = node;
            col_node = node;
            width = database->macros[node]->width();
        }
        else {
            nodes_[node].parent_id_ = col_node;
            nodes_[col_node].left_child_id_ = node;
            col_node = node;
            width += database->macros[node]->width();
        }

        inserted[node] = 1;
        inserted_cnt++;
    }
    */

    // int num_macros = database_->nMacros;
    // for (int i = 0; i < num_macros - 1; ++i) {
    //     nodes_[i].left_child_id_ = i + 1;
    // }
    // for (int i = 1; i < num_macros; ++i) {
    //     nodes_[i].parent_id_ = i - 1;
    // }
    return;
} //END MODULE

//---------------------------------------------------------------------

void BStarTree::Print(ostream& os, int indent_level) const {
    const int indent_size = 2;
    os << string(indent_level * indent_size, ' ') << "BStarTree:" << endl;
    indent_level++;
    os << string(indent_level * indent_size, ' ') << "root_id1_:" << root_id1_ << endl;
    os << string(indent_level * indent_size, ' ') << "root_id2_:" << root_id2_ << endl;
    os << string(indent_level * indent_size, ' ') << "nodes_:" << endl;
    indent_level++;
    for (int i = 0; i < nodes_.size(); i++) {
        const Node& node = nodes_[i];
        os << string(indent_level * indent_size, ' ') << "Node: " << i << endl;
        indent_level++;
        os << string(indent_level * indent_size, ' ')
            << "parent_id_: " << node.parent_id_ << endl;
        os << string(indent_level * indent_size, ' ')
            << "left_child_id_: " << node.left_child_id_ << endl;
        os << string(indent_level * indent_size, ' ')
            << "right_child_id_: " << node.right_child_id_ << endl;
        indent_level--;
    }
} //END MODULE

//---------------------------------------------------------------------

int BStarTree::num_nodes() const { return nodes_.size(); } //END MODULE

//---------------------------------------------------------------------

//int BStarTree::root_id() const { return root_id_; } //END MODULE

int BStarTree::root_id1() const { return root_id1_; } //END MODULE

int BStarTree::root_id2() const { return root_id2_; } //END MODULE

//---------------------------------------------------------------------

int BStarTree::parent_id(int node_id) const { return node(node_id).parent_id_; } //END MODULE

//---------------------------------------------------------------------

int BStarTree::left_child_id(int node_id) const {
  return node(node_id).left_child_id_;
} //END MODULE

//---------------------------------------------------------------------

int BStarTree::right_child_id(int node_id) const {
  return node(node_id).right_child_id_;
} //END MODULE

//---------------------------------------------------------------------

void BStarTree::DeleteAndInsert(int deleted_node_id, int target_node_id,
                                pair<int, int> inserted_positions) {
    Delete(deleted_node_id);
    Insert(deleted_node_id, target_node_id, inserted_positions);
} //END MODULE

//---------------------------------------------------------------------

// Private
Node::Node(int parent_id, int left_child_id, int right_child_id)
    :   parent_id_(parent_id),
        left_child_id_(left_child_id),
        right_child_id_(right_child_id)
        {} //END MODULE

//---------------------------------------------------------------------

Node& BStarTree::node(int node_id) { return nodes_.at(node_id); } //END MODULE

//---------------------------------------------------------------------

const Node& BStarTree::node(int node_id) const {
    return nodes_.at(node_id);
} //END MODULE

//---------------------------------------------------------------------

void BStarTree::Delete(int deleted_node_id) {
    Node& deleted_node = node(deleted_node_id);

    int child_id = -1;
    if (deleted_node.left_child_id_ != -1 && deleted_node.right_child_id_ != -1) {
        child_id = deleted_node.left_child_id_;
        int current_node_id = child_id;
        while (left_child_id(current_node_id) != -1 &&
            right_child_id(current_node_id) != -1) 
        {
            current_node_id = left_child_id(current_node_id);
        }
        if (right_child_id(current_node_id) != -1) {
            Node& current_node = node(current_node_id);
            current_node.left_child_id_ = current_node.right_child_id_;
            current_node.right_child_id_ = -1;
        }
        while (current_node_id != deleted_node_id) {
            Node& current_node = node(current_node_id);
            Node& current_parent = node(current_node.parent_id_);
            Node& current_parent_right_child = node(current_parent.right_child_id_);
            current_node.right_child_id_ = current_parent.right_child_id_;
            current_parent_right_child.parent_id_ = current_node_id;
            current_node_id = current_node.parent_id_;
        }
    } 
    else if (deleted_node.left_child_id_ != -1) {
        child_id = deleted_node.left_child_id_;
    } 
    else if (deleted_node.right_child_id_ != -1) {
        child_id = deleted_node.right_child_id_;
    }

    if (child_id != -1) {
        node(child_id).parent_id_ = deleted_node.parent_id_;
    }

    if (deleted_node.parent_id_ == -1) {
        if(deleted_node_id< database->nMacros1)
        {
            root_id1_ = child_id;
        }
        else{
            root_id2_ = child_id;
        }
    } 
    else {
        Node& parent = node(deleted_node.parent_id_);
        if (parent.left_child_id_ == deleted_node_id) {
            parent.left_child_id_ = child_id;
        } 
        else {
            parent.right_child_id_ = child_id;
        }
    }

    deleted_node.parent_id_ = -1;
    deleted_node.left_child_id_ = -1;
    deleted_node.right_child_id_ = -1;
    //this->print();//删除看起来还算正常
} //END MODULE

//---------------------------------------------------------------------

void BStarTree::Insert(int inserted_node_id, int target_node_id,
                       pair<int, int> inserted_positions) {
    Node& inserted_node = node(inserted_node_id);
    Node& target_node = node(target_node_id);
    inserted_node.parent_id_ = target_node_id;
    if (inserted_positions.first % 2 == 0) {
        if (inserted_positions.second % 2 == 0) {
            inserted_node.left_child_id_ = target_node.left_child_id_;
            if( inserted_node.left_child_id_>=0&&
                ((inserted_node_id<database->nMacros1) + (inserted_node.left_child_id_<database->nMacros1))==1)
            {
                int debuggggg=1;
            }
        } 
        else {
            inserted_node.right_child_id_ = target_node.left_child_id_;
            if( inserted_node.right_child_id_>=0 &&
                ((inserted_node_id<database->nMacros1) + (inserted_node.right_child_id_<database->nMacros1))==1)
            {
                int debuggggg=1;
            }
        }
        if (target_node.left_child_id_ != -1) {
            node(target_node.left_child_id_).parent_id_ = inserted_node_id;
        }
        target_node.left_child_id_ = inserted_node_id;
        if(target_node.left_child_id_>=0 &&
            ((target_node_id<database->nMacros1) + (target_node.left_child_id_<database->nMacros1))==1)
        {
                int debuggggg=1;
        }
    } 
    else {
        if (inserted_positions.second % 2 == 0) {
            inserted_node.left_child_id_ = target_node.right_child_id_;
            if(inserted_node.left_child_id_>=0 &&
                ((inserted_node_id<database->nMacros1) + (inserted_node.left_child_id_<database->nMacros1))==1)
            {
                int debuggggg=1;
            }
        }
        else {
            inserted_node.right_child_id_ = target_node.right_child_id_;
            if( inserted_node.right_child_id_>=0 &&
                ((inserted_node_id<database->nMacros1) + (inserted_node.right_child_id_<database->nMacros1))==1)
            {
                int debuggggg=1;
            }
        }
        if (target_node.right_child_id_ != -1) {
            node(target_node.right_child_id_).parent_id_ = inserted_node_id;
        }
        target_node.right_child_id_ = inserted_node_id;
        if(target_node.right_child_id_>=0 &&
            ((target_node_id<database->nMacros1) + (target_node.right_child_id_<database->nMacros1))==1)
        {
                int debuggggg=1;
        }
    }
} //END MODULE

//---------------------------------------------------------------------
