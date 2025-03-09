#include "die_partitioner.h"
namespace dp {
vector<int> DiePartitioner::merge(vector<int> partition_result_1, int valid_part_num_1, vector<int> partition_result_2) {
    /*
    vector<int> real_id_to_die_id;
    vector<int> die_id_to_real_id;

    vector<int> net_list;
    vector<int> net_list_end;

    torch::Tensor net_list_real;
    torch::Tensor net_list_real_end;
    int num_nodes;
    int num_nets;
    int num_pins;
    vector<int> partVec;
    */
    // important partition_result
    vector<int> partition_result;
    partition_result.resize(partition_result_1.size());
    /*
    int max_block_id=-1;
    for(int i=0;i<partition_result_1.size();i++)
    {
        partition_result.push_back(partition_result_1[i]);
        max_block_id=max(max_block_id,partition_result_1[i]);
    }
    max_block_id++;
    for(int i=0;i<partition_result_1.size();i++)
    {
        if(partition_result_2[i]>=0)
        {
            int new_id = max_block_id+partition_result_2[i];
            partition_result[i]=new_id;
        }
    }
    */
    for(int i=0;i<partition_result_1.size();i++)
    {
        if(partition_result_1[i]>=0&&partition_result_2[i]>=0)
        {
            logger.info("ERROR! cell %d on top die and bottom die", i);
        }
        if(partition_result_1[i]<0&&partition_result_2[i]<0)
        {
            logger.info("ERROR! cell %d on top die and bottom die", i);
        }
        if(partition_result_1[i]>=0)
        {
            partition_result[i] = partition_result_1[i];
        }else{
            partition_result[i] = partition_result_2[i]+valid_part_num_1;
        }
    }
    return partition_result;
}

vector<int> DiePartitioner::devide_to_pieces(NodeData &data, torch::Tensor node_weight, int numPart) {
    // logger.info("============= Deviding die %d ============", die_id);
    // at::TensorAccessor<int, 1> node_weight = dp_db.node_weight;
    real_id_to_die_id.resize(data.cell_mov_rhs);
    auto node_die = data.node_die;
    int die_node_id = 0;
    for (int i = data.cell_mov_lhs; i < data.cell_mov_rhs; i++) {
        // if(data.node_die[i].item<int>()!=die_id)
        // {
        //     real_id_to_die_id[i] = -1;
        //     continue;
        // }
        if (node_weight[i].item<int>() == 0) {
            continue;
        }
        if (data.macro_mask[i].item<int>() == 1) {
            continue;
        }
        die_id_to_real_id.push_back(i);
        // cout<<"push! "<<i<<endl;
        real_id_to_die_id[i] = die_node_id;
        die_node_id++;
    }

    net_list_real = data.hyperedge_list;
    net_list_real_end = data.hyperedge_list_end;

    for (int i = 0; i < data.num_nets; ++i) {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = net_list_real_end[i - 1].item<int64_t>();
        }
        int64_t end_idx = net_list_real_end[i].item<int64_t>();
        /* add cell pin */
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int pin = net_list_real[idx].item<int>();
            int node_id_real = data.pin_id2node_id[pin].item<int>();
            // if(data.node_die[node_id_real].item<int>()!=die_id)
            // {
            //     continue;
            // }
            if (node_weight[node_id_real].item<int>() == 0) {
                continue;
            }
            if (data.macro_mask[node_id_real].item<int>() == 1) {
                continue;
            }
            int node_id_die = real_id_to_die_id[node_id_real];
            net_list.push_back(node_id_die);
        }
        /* add via */
        net_list_end.push_back(net_list.size());
    }
    //////////////////////////////////////////////////////////////////////////////////////
    auto device = data.device;
    num_nodes = die_id_to_real_id.size();
    num_nets = net_list_end.size();
    num_pins = net_list.size();
    // torch::Tensor net_to_num_nodes = torch::zeros({num_nets}, dtype(torch::kInt));
    // int net_to_num_pins = data.net_to_num_pins.clone();
    // vector<shared_ptr<ptNode>> nodes;
    // vector<shared_ptr<ptNet>> nets;

    //////////////////////////////////////////////////////////////////////////////////////

    int nNode = num_nodes;
    int *cwghts = new int[nNode];
    for (int i = 0; i < nNode; i++) cwghts[i] = 1;

    // int nNets = nets.size();
    int nNets = num_nets;
    int *nwghts = new int[nNets];
    for (int i = 0; i < nNets; i++) {
        nwghts[i] = 1;
        // if (net_to_num_pins[i].item<int>() <= st::setting.cut_net_thres)
        //     nwghts[i] = (int)(st::setting.net_weight_coef * st::setting.net_weight_offset);
        // else
        //     nwghts[i] = st::setting.net_weight_offset;
    }
    // logger.info("Nets larger than %d will be applied with wa force %.3f",
    //             st::setting.cut_net_thres,
    //             st::setting.net_weight_coef);

    int _c = nNode;
    int _n = nNets;
    int _nconst = 1;
    int useFixCells = false;

    int nPin = num_pins;
    // for (auto net : nets) {
    //     nPin += net->Nodes.size();
    // }

    int *xpins = new int[_n + 1];
    xpins[0] = 0;

    // int *pins = net_list;
    int *pins = new int[nPin];
    for (int i = 0; i < nPin; i++) {
        pins[i] = net_list[i];
    }

    for (int i = 1; i <= _n; i++) {
        // auto net = nets[i];
        // for (auto node : net->Nodes) {
        //     pins[p++] = node->id;
        // }
        xpins[i] = net_list_end[i - 1];
    }

    PaToH_Parameters args;
    // PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_QUALITY);
    PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_QUALITY);
    args.seed = 0;
    args._k = numPart;
    // args.final_imbal = 0.05;  // completely ignore balance
    // args.final_imbal = 1;  // completely ignore balance
    args.final_imbal = st::setting.pt_imbl;

    args.MemMul_Pins = 1000;
    args.MemMul_CellNet = 1000;
    // args.MemMul_CellNet = 3;
    // args.MemMul_General = 2;
    PaToH_Check_User_Parameters(&args, true);

    int *partvec = new int[nNode];
    int *partweights = new int[numPart];
    int cut;
    PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, xpins, pins);

    // for (int i = 0; i < nNode; i++) partvec[i] = nodes[i]->group;
    // for (int i = 0; i < nNode; i++) partvec[i] = -1; // FIXME:
    for (int i = 0; i < nNode; i++) partvec[i] = -1;

    logger.info("Partitioner::run_patoh, %s", "finish setting up");

    float *targetweights = new float[numPart];
    float ratio = 1.0 / float(numPart);  // data.tech_ratio.item<double>() + st::setting.top_util_filler;
    for (int i = 0; i < numPart; i++) {
        targetweights[i] = ratio;
    }
    // targetweights[0] = ratio;
    // targetweights[1] = 1 - ratio;  // TODO:
    if(numPart>1)
    {
        PaToH_Part(
        &args, _c, _n, _nconst, useFixCells, cwghts, nwghts, xpins, pins, targetweights, partvec, partweights, &cut);
    }
    else{
        for(int i=0;i<nNode;i++)
        {
            partvec[i]=0;
        }
    }
    
    // PaToH_Partition_with_FixCells(&args, _c, _n, cwghts, nwghts, xpins, pins, partvec, partweights, &cut);

    logger.info("============ Original partition result ============");
    // for (int i = 0; i < nNode; i++) {
    //     node_die[i] = partvec[i];
    // }
    // logger.info("Partition ratio: %d : %d", partweights[0], partweights[1]);
    // rpt_cut_size();
    // logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));

    // mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    for (int i = 0; i < nNode; i++) {
        int group = partvec[i];
        // TODO: force balance
        // if (st::setting.clamp_util) {
        //     if ((mov_cell_areas[group] + nodes[i]->sizes[group] > max_mov_cell_areas[group]).item<bool>())
        //         group = !group;
        // }
        // mov_cell_areas[group] += nodes[i]->sizes[group];
        // nodes[i]->group = group;
        // node_die[i] = group;
    }

    // data.node_die = node_die.clone();  // TODO: construct data from pt
    // data.mov_cell_areas = mov_cell_areas.clone();

    // logger.info("============ Legalized partition result ============");
    // rpt_cut_size();
    // logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));
    // logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    // logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(),
    // (mov_cell_areas[1]).item<long>()); logger.info("Utils for each chip (%.2f, %.2f)", (mov_cell_areas[0] /
    //max_mov_cell_areas[0]).item<double>(), (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());
    vector<int> partition_result;
    partition_result.resize(data.cell_mov_rhs);
    for (int i = 0; i < data.cell_mov_rhs; i++) {
        partition_result[i] = -1;
    }
    for (int i = 0; i < nNode; i++) {
        int real_id = die_id_to_real_id[i];
        if(real_id<0||real_id>=data.cell_mov_rhs)
        {
            logger.info("ERROR!, %d",real_id);
        }
        partition_result[real_id] = partvec[i];
        partVec.push_back(partvec[i]);  // partVec是一个die上重新编号的
    }
    logger.info("============ finish die partition ============");
    free(cwghts);
    free(nwghts);
    free(xpins);
    free(pins);
    free(partweights);
    free(partvec);
    PaToH_Free();
    logger.info("============ free memory finish, exiting ============");
    return partition_result;
}

}  // namespace dp