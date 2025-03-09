
#pragma once

#include "global.h"
#include "parser/db/Database.h"
#include "parser/gp/GPDatabase.h"

typedef std::map<std::string,
                 std::variant<int,
                              double,
                              float,
                              std::string,
                              torch::Tensor,
                              std::vector<std::tuple<gp::index_type, gp::index_type, std::string>>,
                              unordered_map<string, string>,
                              std::vector<std::string>,
                              std::tuple<int, int>>>
    Dict;

class GlobalParser {
public:
    /* I/O operation */
    void set_single_design_params(const st::Setting& setting);
    void single_ispd2005(const st::Setting& setting);
    void single_IBM(const st::Setting& setting);
    void single_iccad2022(const st::Setting& setting);
    void single_iccad2019(const st::Setting& setting);
    void single_ispd2015_without_fence(const st::Setting& setting);
    void single_openroad(const st::Setting& setting);

    shared_ptr<db::Database> create_database() { return make_shared<db::Database>(); }
    shared_ptr<gp::GPDatabase> create_gpdatabase(shared_ptr<db::Database> db) {
        return std::make_shared<gp::GPDatabase>(db);
    }

    /* Pre_process */
    Dict preprocess_design_info(shared_ptr<gp::GPDatabase> gpdb);
    Dict preprocess_design_info_iccad2022(shared_ptr<gp::GPDatabase> gpdb);
};