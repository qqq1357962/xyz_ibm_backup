#include "fp_pack.h"

using namespace fp;

//---------------------------------------------------------------------
void shuffleArray(std::vector<int>& arr, int start, int end) {
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::shuffle(arr.begin() + start, arr.begin() + end);//, gen);
    std::random_shuffle(arr.begin() + start, arr.begin() + end);
}

Packer::Packer(fp::Database* database_)
    :  //@@
      width1_(0.0),
      height1_(0.0),
      width2_(0.0),
      height2_(0.0),
      wirelength_(0.0),
      not_overlap_ratio(0.0),
      b_star_tree_(database_),
      macro_id_by_node_id_(database_->nMacros, -1),
      macro_die_(database_->nMacros, -1),
      is_macro_rotated_by_id_(database_->nMacros, false),
      is_macro_rotatable_by_id_(database_->nMacros, false),
      macro_bounding_box_by_id_(database_->nMacros, make_pair(fp::Point(0, 0), fp::Point(0, 0))) {
    for (int i = 0; i < database_->nMacros1; i++) {
        macro_die_[i] = 0;
    }
    for (int i = database_->nMacros1; i < database_->nMacros; i++) {
        macro_die_[i] = 1;
    }
    for (int i = 0; i < macro_id_by_node_id_.size(); i++) {     // 树结构数组
        macro_id_by_node_id_[i] = i;                            // sequence_num
        int node_die = macro_die_.at(macro_id_by_node_id_[i]);  // 顺序结构数组
        is_macro_rotatable_by_id_[i] =
            (database_->macros[i]->width(node_die) != database_->macros[i]->height(node_die));
    }
    macro_pins.resize(macro_id_by_node_id_.size());
    database_->nMacros = database_->nMacros1 + database_->nMacros2;
    database = database_;

    // init B* tree
    vector<int> node_ids;
    int num_macros_all = database_->nMacros1 + database_->nMacros2;
    for (int i = 0; i < num_macros_all; i++) {
        node_ids.push_back(i);
    }
    shuffleArray(node_ids, 0, database_->nMacros1);
    shuffleArray(node_ids, database_->nMacros1, num_macros_all);

    b_star_tree_.root_id1_ = node_ids[0];

    int row_node = b_star_tree_.root_id1_;
    int col_node = b_star_tree_.root_id1_;
    int node_die_root1 = macro_die_.at(b_star_tree_.root_id1_);
    int width = database->macros[b_star_tree_.root_id1_]->width(node_die_root1);  // macros:顺序结构数组

    for (int node_pos = 1; node_pos < database_->nMacros1; node_pos++) {
        int node = node_ids[node_pos];
        if (width > database->outline_width / 2) {
            b_star_tree_.nodes_[node].parent_id_ = row_node;
            b_star_tree_.nodes_[row_node].right_child_id_ = node;
            row_node = node;
            col_node = node;
            int node_die = macro_die_.at(node);
            width = database->macros[node]->width(node_die);  // macros:顺序结构数组
        } else {
            b_star_tree_.nodes_[node].parent_id_ = col_node;
            b_star_tree_.nodes_[col_node].left_child_id_ = node;
            col_node = node;
            int node_die = macro_die_.at(node);
            width += database->macros[node]->width(node_die);
        }
    }

    int num_macros = database_->nMacros2;
    vector<int> inserted(num_macros, 0);
    // root_id_ = rand() % num_macros;
    if (database_->nMacros2 == 0) {
        return;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    b_star_tree_.root_id2_ = node_ids[database_->nMacros1];
    row_node = b_star_tree_.root_id2_;
    col_node = b_star_tree_.root_id2_;
    int node_die_root2 = macro_die_[b_star_tree_.root_id2_];
    width = database->macros[b_star_tree_.root_id2_]->width(node_die_root2);

    for (int node_pos = database_->nMacros1 + 1; node_pos < database_->nMacros1 + database_->nMacros2; node_pos++) {
        int node = node_ids[node_pos];
        if (width > database->outline_width / 2) {
            b_star_tree_.nodes_[node].parent_id_ = row_node;
            b_star_tree_.nodes_[row_node].right_child_id_ = node;
            row_node = node;
            col_node = node;
            width = database->macros[node]->width(macro_die_[node]);
        } else {
            b_star_tree_.nodes_[node].parent_id_ = col_node;
            b_star_tree_.nodes_[col_node].left_child_id_ = node;
            col_node = node;
            width += database->macros[node]->width(macro_die_[node]);
        }
    }
    cout << "after init:" << endl;
    // this->print();
}  // END MODULE

//---------------------------------------------------------------------

void Packer::print() {
    const int indent_size = 2;
    int indent_level = 0;
    cout << string(indent_level * indent_size, ' ') << "BStarTree:" << endl;
    indent_level++;
    cout << string(indent_level * indent_size, ' ')
         << "root_name1_:" << database->macros[macro_id_by_node_id_.at(b_star_tree_.root_id1())]->name() << endl;
    cout << string(indent_level * indent_size, ' ')
         << "root_name2_:" << database->macros[macro_id_by_node_id_.at(b_star_tree_.root_id2())]->name() << endl;
    cout << string(indent_level * indent_size, ' ') << "nodes_:" << endl;
    indent_level++;
    for (int i = 0; i < b_star_tree_.nodes_.size(); i++) {
        const Node& node = b_star_tree_.nodes_[i];
        /*
        cout << string(indent_level * indent_size, ' ')
             << "Node"<<i<<": " << macro_id_by_node_id_.at(i)<<" "<<
        database->macros[macro_id_by_node_id_.at(i)]->name() << endl; indent_level++; if (node.parent_id_ != -1) cout <<
        string(indent_level * indent_size, ' ')
                 << "parent_name_: " << node.parent_id_
                 <<" " <<database->macros[macro_id_by_node_id_.at(node.parent_id_)]->name() << endl;
        if (node.left_child_id_ != -1)
            cout << string(indent_level * indent_size, ' ')
                 << "left_child_name_: " << node.left_child_id_
                 <<" "<<database->macros[macro_id_by_node_id_.at(node.left_child_id_)]->name()
                 << endl;
        if (node.right_child_id_ != -1)
            cout << string(indent_level * indent_size, ' ')
                 << "right_child_name_: " << node.right_child_id_ <<" "
                 <<database->macros[macro_id_by_node_id_.at(node.right_child_id_)]->name()
                 << endl;
        */
        cout << "Node: " << i << " " << node.parent_id_ << " " << node.left_child_id_ << " " << node.right_child_id_
             << endl;
        if (i == node.left_child_id_ || i == node.right_child_id_) {
            int debuggg = 1;
        }
        if (i == database->nMacros1 - 1) {
            cout << "=======" << endl;
        }
        indent_level--;
    }
}  // END MODULE

//---------------------------------------------------------------------

int Packer::num_macros() const { return b_star_tree_.num_nodes(); }  // END MODULE

//---------------------------------------------------------------------

// double Packer::width() const { return width_; }  // END MODULE

double Packer::width1() const { return width1_; }  // END MODULE

double Packer::width2() const { return width2_; }  // END MODULE

//---------------------------------------------------------------------

// double Packer::height() const { return height_; }  // END MODULE

double Packer::height1() const { return height1_; }  // END MODULE

double Packer::height2() const { return height2_; }  // END MODULE

//---------------------------------------------------------------------

double Packer::area() const { return area_; }  // END MODULE

//---------------------------------------------------------------------

double Packer::wirelength() const { return wirelength_; }  // END MODULE

//---------------------------------------------------------------------

const pair<fp::Point, fp::Point>& Packer::macro_bounding_box(int macro_id) const {
    return macro_bounding_box_by_id_.at(macro_id);
}  // END MODULE

//---------------------------------------------------------------------

void Packer::Perturb() {
    const int num_nodes = b_star_tree_.num_nodes();
    const int num_macros = num_nodes;
    int operation_num = 3;
    if (st::setting.is_fp_permit_change_cross_chip) {
        operation_num = 4;
    }
    const int op = rand() % operation_num;
    // cout<<op;
    switch (op) {
        case 0: {  // case0: 旋转
            // TODO: all rotatable?
            // const int macro_id = rand() % num_macros;
            const int macro_id = [&]() {
                int node_id = rand() % database->numRotatable;
                node_id = database->macro_sequence_rotatable[node_id];
                return node_id;
            }();
            // [&]() {
            // int macro_id = rand() % num_macros;
            //     while (!is_macro_rotatable_by_id_[macro_id]) {
            //         macro_id = rand() % num_macros;
            //     }
            //     return macro_id;
            // }();
            // logger.info("rotating %d", macro_id);
            // this->print();
            is_macro_rotated_by_id_.at(macro_id) = !is_macro_rotated_by_id_.at(macro_id);
            // this->print();
            break;
        }
        case 1: {  // case1: 交换 isFromSTD
            const int node_a_id = rand() % num_nodes;
            if(node_a_id<database->nMacros1 && database->nMacros1==1)
            {
                break;
            }
            if(node_a_id>=database->nMacros1 && database->nMacros2==1)
            {
                break;
            }
            const int node_b_id = [&]() {
                int node_id = rand() % num_nodes;
                while (node_id == node_a_id ||
                       (node_id >= database->nMacros1) + (node_a_id >= database->nMacros1) == 1) {
                    node_id = rand() % num_nodes;
                }
                return node_id;
            }();
            // TODO:
            int& node_a_macro_id = macro_id_by_node_id_.at(node_a_id);
            int& node_b_macro_id = macro_id_by_node_id_.at(node_b_id);
            // logger.info("swaping %d and %d", node_a_id, node_b_id);
            // this->print();
            swap(node_a_macro_id, node_b_macro_id);
            // this->print();
            break;
        }
        case 2: {  // case2: 移动
            const int node_a_id = rand() % num_nodes;
            if(node_a_id<database->nMacros1 && database->nMacros1==1)
            {
                break;
            }
            if(node_a_id>=database->nMacros1 && database->nMacros2==1)
            {
                break;
            }
            const int node_b_id = [&]() {
                int node_id = rand() % num_nodes;
                while (node_id == node_a_id ||
                       (node_id >= database->nMacros1) + (node_a_id >= database->nMacros1) == 1) {
                    node_id = rand() % num_nodes;
                }
                return node_id;
            }();
            auto inserted_positions = make_pair(rand(), rand());
            // logger.info("deleting %d and inserting %d", node_a_id, node_b_id);
            //  logger.info("%d, %d, %d; %d, %d, %d", node_a_id, b_star_tree_.node(node_a_id).left_child_id_,
            //  b_star_tree_.node(node_a_id).right_child_id_,
            //                                        node_b_id, b_star_tree_.node(node_b_id).left_child_id_,
            //                                        b_star_tree_.node(node_b_id).right_child_id_);
            // this->print();
            b_star_tree_.DeleteAndInsert(node_a_id, node_b_id, inserted_positions);
            // this->print();
            break;
        }
        case 3: {  // case1: 交换 isFromSTD
            const int node_a_id = [&]() {
                int pos = rand() % num_macros;
                while (true) {
                    int macro_id = macro_id_by_node_id_.at(pos);
                    if (is_macro_rotatable_by_id_[macro_id]) {
                        return pos;
                    }
                    pos = rand() % num_macros;
                }
            }();
            const int node_b_id = [&]() {
                int pos = rand() % num_macros;
                while (true) {
                    int macro_id = macro_id_by_node_id_.at(pos);
                    if (is_macro_rotatable_by_id_[macro_id] && pos != node_a_id) {
                        return pos;
                    }
                    pos = rand() % num_macros;
                }
            }();
            int& node_a_macro_id = macro_id_by_node_id_.at(node_a_id);  // tree结构数组
            int& node_b_macro_id = macro_id_by_node_id_.at(node_b_id);
            int& node_die_a = macro_die_.at(node_a_macro_id);  // 顺序数组
            int& node_die_b = macro_die_.at(node_b_macro_id);
            // logger.info("swaping %d and %d", node_a_id, node_b_id);
            // this->print();
            swap(node_die_a, node_die_b);
            swap(node_a_macro_id, node_b_macro_id);
            break;
        }
        default:
            break;
    }
}  // END MODULE

//---------------------------------------------------------------------

void Packer::traverseTree(int current_node_id, bool left) {
    Node& current_node = b_star_tree_.nodes_[current_node_id];
    const int current_macro_id = macro_id_by_node_id_[current_node_id];
    fp::Macro* current_macro = database->macros[current_macro_id];

    // parent node and macro
    int parent_node_id = current_node.parent_id_;
    const int parent_macro_id = macro_id_by_node_id_[parent_node_id];
    fp::Macro* parent_macro = database->macros[parent_macro_id];

    const pair<fp::Point, fp::Point> parent_macro_bounding_box = macro_bounding_box_by_id_[parent_macro_id];

    const bool is_current_macro_rotated = is_macro_rotated_by_id_[current_macro_id];
    int current_macro_width = current_macro->width(macro_die_[current_macro_id]);
    int current_macro_height = current_macro->height(macro_die_[current_macro_id]);
    if (is_current_macro_rotated) {
        swap(current_macro_width, current_macro_height);
    }

    // left or right child of parent
    int lower_left_x;
    int lower_left_y;
    if (left)
        lower_left_x = parent_macro_bounding_box.second.x();
    else
        lower_left_x = parent_macro_bounding_box.first.x();

    int x_start = lower_left_x;
    int x_end = x_start + current_macro_width;
    if (x_end > contour.box.size()) contour.box.resize(x_end);
    int y_max = 0;
    for (int i = x_start; i < x_end; i++)
        if (contour.box[i] > y_max) y_max = contour.box[i];  // 检查该位置的macro底下的最高的位置

    lower_left_y = y_max;

    int upper_right_x = lower_left_x + current_macro_width;
    int upper_right_y = lower_left_y + current_macro_height;

    if (upper_right_x > contour.box_x_max) contour.box_x_max = upper_right_x;
    if (upper_right_y > contour.box_y_max) contour.box_y_max = upper_right_y;

    y_max += current_macro_height;
    for (int i = x_start; i < x_end; i++) contour.box[i] = y_max;

    macro_bounding_box_by_id_[current_macro_id] =
        make_pair(fp::Point(lower_left_x, lower_left_y), fp::Point(upper_right_x, upper_right_y));
    macro_pins[current_macro_id] =
        fp::pin((int)(lower_left_x + upper_right_x) / 2, (int)(lower_left_y + upper_right_y) / 2);

    if (current_node.left_child_id_ != -1) traverseTree(current_node.left_child_id_, true);
    if (current_node.right_child_id_ != -1) traverseTree(current_node.right_child_id_, false);

}  // END MODULE

void Packer::traverseTree_vis() {
    // if(current_node_id==root)
    // {
    //     drawList.clear();
    // }

    // Node &current_node = b_star_tree_.nodes_[current_node_id];
    // const int current_macro_id = macro_id_by_node_id_[current_node_id];
    // fp::Macro* current_macro = database->macros[current_macro_id];

    // // parent node and macro
    // int parent_node_id = current_node.parent_id_;
    // const int parent_macro_id = macro_id_by_node_id_[parent_node_id];
    // fp::Macro* parent_macro = database->macros[parent_macro_id];

    // const pair<fp::Point, fp::Point> parent_macro_bounding_box =
    //         macro_bounding_box_by_id_[parent_macro_id];

    // const bool is_current_macro_rotated = is_macro_rotated_by_id_[current_macro_id];
    // int current_macro_width = current_macro->width();
    // int current_macro_height = current_macro->height();
    // if (is_current_macro_rotated) {
    //     swap(current_macro_width, current_macro_height);
    // }

    // // left or right child of parent
    // int lower_left_x;
    // int lower_left_y;
    // if(current_node_id==root)
    // {
    //     drawList.clear();
    //     lower_left_x=0;
    // }
    // if (left)
    //     lower_left_x = parent_macro_bounding_box.second.x();
    // else
    //     lower_left_x = parent_macro_bounding_box.first.x();

    // int x_start = lower_left_x;
    // int x_end = x_start + current_macro_width;
    // if (x_end > contour.box.size())
    //     contour.box.resize(x_end);
    // int y_max = 0;
    // for (int i = x_start; i < x_end; i++)
    //     if (contour.box[i] > y_max)
    //         y_max = contour.box[i];//检查该位置的macro底下的最高的位置

    // lower_left_y = y_max;

    // int upper_right_x = lower_left_x + current_macro_width;
    // int upper_right_y = lower_left_y + current_macro_height;

    // if (upper_right_x > contour.box_x_max)
    //     contour.box_x_max = upper_right_x;
    // if (upper_right_y > contour.box_y_max)
    //     contour.box_y_max = upper_right_y;

    // y_max += current_macro_height;
    // for (int i = x_start; i < x_end; i++)
    //     contour.box[i] = y_max;

    // macro_bounding_box_by_id_[current_macro_id] =
    //         make_pair(fp::Point(lower_left_x, lower_left_y),
    //                 fp::Point(upper_right_x, upper_right_y));
    // macro_pins[current_macro_id] = fp::pin(
    //             (int)(lower_left_x + upper_right_x) / 2,
    //             (int)(lower_left_y + upper_right_y) / 2);
    // drawList.push_back(drawContent(lower_left_x,lower_left_y,upper_right_x,upper_right_y,current_macro_id));
    // if (current_node.left_child_id_ != -1)
    //     traverseTree_vis(current_node.left_child_id_, true);
    // if (current_node.right_child_id_ != -1)
    //     traverseTree_vis(current_node.right_child_id_, false);
    // if(current_node_id==root)
    // {
    int bounding_x = database->outline_width;
    int bounding_y = database->outline_height;
    string py_cmd = "python plot.py 0 ";
    for (int i = 0; i < database->nMacros; i++) {
        if (macro_die_[i] == 1) {
            continue;
        }
        pair<fp::Point, fp::Point> macro_bounding_box = macro_bounding_box_by_id_[i];
        int lx, ly, ux, uy;
        int id;
        lx = macro_bounding_box.first.x();
        ux = macro_bounding_box.second.x();
        ly = macro_bounding_box.first.y();
        uy = macro_bounding_box.second.y();
        id = i;

        py_cmd += to_string(lx);
        py_cmd += ",";
        py_cmd += to_string(ly);
        py_cmd += ",";
        py_cmd += to_string(ux);
        py_cmd += ",";
        py_cmd += to_string(uy);
        string name = database->macros[id]->name();
        py_cmd += ",", py_cmd += name;  // to_string(id);
        py_cmd += "lll";
    }
    py_cmd += to_string(0);
    py_cmd += ",";
    py_cmd += to_string(0);
    py_cmd += ",";
    py_cmd += to_string(bounding_x);
    py_cmd += ",";
    py_cmd += to_string(bounding_y);
    py_cmd += ",", py_cmd += to_string(-1);
    py_cmd += "lll";
    // cout << py_cmd << endl;
    system(py_cmd.c_str());

    py_cmd = "python plot.py 1 ";
    for (int i = 0; i < database->nMacros; i++) {
        if (macro_die_[i] == 0) {
            continue;
        }
        pair<fp::Point, fp::Point> macro_bounding_box = macro_bounding_box_by_id_[i];
        int lx, ly, ux, uy;
        int id;
        lx = macro_bounding_box.first.x();
        ux = macro_bounding_box.second.x();
        ly = macro_bounding_box.first.y();
        uy = macro_bounding_box.second.y();
        id = i;

        py_cmd += to_string(lx);
        py_cmd += ",";
        py_cmd += to_string(ly);
        py_cmd += ",";
        py_cmd += to_string(ux);
        py_cmd += ",";
        py_cmd += to_string(uy);
        string name = database->macros[id]->name();
        py_cmd += ",", py_cmd += name;  // to_string(id);
        py_cmd += "lll";
    }
    py_cmd += to_string(0);
    py_cmd += ",";
    py_cmd += to_string(0);
    py_cmd += ",";
    py_cmd += to_string(bounding_x);
    py_cmd += ",";
    py_cmd += to_string(bounding_y);
    py_cmd += ",", py_cmd += to_string(-1);
    py_cmd += "lll";
    // cout << py_cmd << endl;
    system(py_cmd.c_str());
    for (int i = 0; i < database->nMacros; i++) {
        logger.info("%d, pos: %d, %d, name: %s, die: %d",
                    i,
                    macro_bounding_box_by_id_[i].first.x(),
                    macro_bounding_box_by_id_[i].first.y(),
                    database->macros[i]->name().c_str(),
                    macro_die_[i]);
    }
    //}
}  // END MODULE

void Packer::PrintMacroPins() {
    // for (int i = 0; i < database->nNets; i++) {
    //     const vector<int>& net = database->nets_by_id[i];
    //     cout<<"net "<<i<<endl;
    //     for (const int id : net) {
    //         cout<<"  "<<macro_pins[id].min_x<<" "<<macro_pins[id].min_y<<endl;
    //     }
    // }
    for (int i = 0; i < macro_pins.size(); i++) {
        cout << i << "  " << macro_pins[i].min_x << " " << macro_pins[i].min_y << endl;
    }
    for (int i = 0; i < macro_die_.size(); i++) {
        cout << i << "  " << macro_die_[i] << endl;
    }
}

//---------------------------------------------------------------------
// FIXME:
// TODO: extract codes bellow into a function
void Packer::PackInt() {
    // find root node and macro
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    int root_id = b_star_tree_.root_id1();
    // cout<<"logg1 "<<root_id<<endl;
    // this->print();
    Node& root_node = b_star_tree_.nodes_[root_id];
    int root_macro_id = macro_id_by_node_id_[root_id];
    fp::Macro* root_macro = database->macros[root_macro_id];
    // cout<<"logg2 "<<root_macro_id<<endl;
    // this->print();
    // set box
    int root_macro_width = root_macro->width(macro_die_[root_macro_id]);
    int root_macro_height = root_macro->height(macro_die_[root_macro_id]);
    bool is_root_macro_rotated = is_macro_rotated_by_id_[root_macro_id];
    if (is_root_macro_rotated) {
        swap(root_macro_width, root_macro_height);
    }
    macro_bounding_box_by_id_[root_macro_id] =
        make_pair(fp::Point(0, 0), fp::Point(root_macro_width, root_macro_height));

    macro_pins[root_macro_id] = fp::pin((int)root_macro_width / 2, (int)root_macro_height / 2);

    contour = Contour(database->outline_width);
    if (root_macro_width > contour.box_x_max) contour.box_x_max = root_macro_width;
    if (root_macro_height > contour.box_y_max) contour.box_y_max = root_macro_height;
    for (int i = 0; i < root_macro_width; i++) contour.box[i] = root_macro_height;
    // cout<<"logg2"<<endl;
    // this->print();
    if (root_node.left_child_id_ != -1) traverseTree(root_node.left_child_id_, true);
    if (root_node.right_child_id_ != -1) traverseTree(root_node.right_child_id_, false);

    width1_ = contour.box_x_max;
    height1_ = contour.box_y_max;

    // this->print();//这里还正常
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    root_id = b_star_tree_.root_id2();
    Node& root_node2 = b_star_tree_.nodes_[root_id];
    root_macro_id = macro_id_by_node_id_[root_id];
    fp::Macro* root_macro2 = database->macros[root_macro_id];
    // this->print();
    //  set box
    root_macro_width = root_macro2->width(macro_die_[root_macro_id]);
    root_macro_height = root_macro2->height(macro_die_[root_macro_id]);

    // cout<<"logg32"<<endl;
    // this->print();

    is_root_macro_rotated = is_macro_rotated_by_id_[root_macro_id];
    if (is_root_macro_rotated) {
        swap(root_macro_width, root_macro_height);
    }

    // cout<<"logg33"<<endl;
    // this->print();

    macro_bounding_box_by_id_[root_macro_id] =
        make_pair(fp::Point(0, 0), fp::Point(root_macro_width, root_macro_height));

    // cout<<"logg4"<<endl;
    // this->print();
    macro_pins[root_macro_id] = fp::pin((int)root_macro_width / 2, (int)root_macro_height / 2);

    contour = Contour(database->outline_width);
    
    if (root_macro_width > contour.box_x_max) contour.box_x_max = root_macro_width;
    if (root_macro_height > contour.box_y_max) contour.box_y_max = root_macro_height;
    
    for (int i = 0; i < root_macro_width; i++) contour.box[i] = root_macro_height;
    // cout<<"logg5"<<endl;
    // this->print();
    if (root_node2.left_child_id_ != -1) traverseTree(root_node2.left_child_id_, true);
    if (root_node2.right_child_id_ != -1) traverseTree(root_node2.right_child_id_, false);

    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // set Packer outline
    width2_ = contour.box_x_max;
    height2_ = contour.box_y_max;

    // area_ = width_ * height_

    // logger.log() << width_ << " | " << height_ << " " <<
    //        width_*height_<< endl;

    area_ = 0;
    int idx = 0;
    for (int& i : contour.box) {
        idx++;
        if (i < database->outline_height && idx <= database->outline_width)
            area_ += database->outline_height;
        else
            area_ += i;
    }
    // logger.log() << area_ << endl;
    // exit(1);

    not_overlap_ratio = 1;
    if (st::setting.numPart == 1) {
        int index1 = 0;
        auto bounding_box1 = macro_bounding_box(index1);
        const fp::Point& lower_left1 = bounding_box1.first;
        int& node_die_a = macro_die_.at(index1);
        int x_low_1 = lower_left1.x();
        int y_low_1 = lower_left1.y();
        int w1 = database->macros[index1]->width(node_die_a);
        int h1 = database->macros[index1]->height(node_die_a);
        int x_up_1 = lower_left1.x() + w1;
        int y_up_1 = lower_left1.y() + h1;

        int index2 = database->nMacros1;
        auto bounding_box2 = macro_bounding_box(index2);
        const fp::Point& lower_left2 = bounding_box1.first;
        int& node_die_b = macro_die_.at(index1);
        int x_low_2 = lower_left2.x();
        int y_low_2 = lower_left2.y();
        int w2 = database->macros[index2]->width(node_die_b);
        int h2 = database->macros[index2]->height(node_die_b);
        int x_up_2 = lower_left2.x() + w2;
        int y_up_2 = lower_left2.y() + h2;

        int left = max(x_low_1, x_low_2);
        int bottom = max(y_low_1, y_low_2);
        int right = min(x_up_1, x_up_2);
        int top = min(y_up_1, y_up_2);

        float overlap_area = 0;
        if (top > bottom && right > left) {
            overlap_area += (top - bottom) * (right - left);
        }
        float total_area = h1 * w1 + h2 * w2;
        not_overlap_ratio = (total_area - overlap_area) / total_area;
    }

    wirelength_ = 0.0;
    // cout<<"show nets"<<endl;
    for (int i = 0; i < database->nNets; i++) {
        const vector<int>& net = database->nets_by_id[i];
        if (net.size() <= 1) {
            continue;
        }
        const fp::pin& terminal_box = database->net_terminals[i];

        int min_x = terminal_box.min_x;
        int min_y = terminal_box.min_y;
        int max_x = terminal_box.max_x;
        int max_y = terminal_box.max_y;

        for (const int id : net) {
            if (net[0] == 0 && net[1] == 4) {
                int debugg = 0;
            }
            // cout<<id<<" ";
            const int x = macro_pins[id].min_x;
            const int y = macro_pins[id].min_y;
            if (x < min_x) {
                min_x = x;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (y > max_y) {
                max_y = y;
            }
        }
        // cout<<endl;
        if ((max_x - min_x) + (max_y - min_y) < 0 ||
            (max_x - min_x) + (max_y - min_y) > 4 * (database->outline_width + database->outline_height)) {
            int debuggg = 1;
            // exit(0);
            cout << "bugggg" << endl;
            for (const int id : net) {
                cout << macro_pins[id].min_x << " " << macro_pins[id].min_y << endl;
            }
        }
        wirelength_ += (max_x - min_x) + (max_y - min_y);
    }
}  // END MODULE

//---------------------------------------------------------------------

void Packer::write(const string& output_path) {
    ofstream outfile;
    outfile.open(output_path, ios::out);
    outfile << "Wirelength " << wirelength_ << "\n";
    outfile << "Blocks\n";
    for (int i = 0; i < b_star_tree_.num_nodes(); i++) {
        const int macro_id = i;
        const string macro_name = database->macros[macro_id]->name();
        auto bounding_box = macro_bounding_box(macro_id);
        const fp::Point& lower_left = bounding_box.first;
        const fp::Point& upper_right = bounding_box.second;

        outfile << macro_name << ' ' << lower_left.x() << ' ' << lower_left.y() << ' '
                << is_macro_rotated_by_id_.at(macro_id) << endl;
    }
}  // END MODULE

//---------------------------------------------------------------------
