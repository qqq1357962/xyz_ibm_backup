
#include "partition.h"
#include "patoh.h"

void Partitioner::run_patoh(NodeData &data, bool is_move_macro) {
    logger.info("============= Running PaToH ============");

    int nNode = nodes.size();
    int *cwghts = new int[nNode];
    for (int i = 0; i < nNode; i++) {
        if (is_move_macro) {
            cwghts[i] = static_cast<int>((nodes[i]->sizes[0] + nodes[i]->sizes[1]) / 100 / data.site_width * data.macro_mask[i].item<int>());
        } else {
            cwghts[i] = static_cast<int>((nodes[i]->sizes[0] + nodes[i]->sizes[1]) / 100 / data.site_width * (1 - data.macro_mask[i].item<int>()));
        }
        
    }
    int nNets = nets.size();
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

    int nPin = 0;
    for (auto net : nets) {
        nPin += net->Nodes.size();
    }

    int *xpins = new int[_n + 1];
    int *pins = new int[nPin];

    for (int i = 0, p = 0; i < _n; i++) {
        auto net = nets[i];
        for (auto node : net->Nodes) {
            pins[p++] = node->id;
        }
        xpins[i] = p;
    }
    xpins[_n] = nPin;

    PaToH_Parameters args;
    // PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_QUALITY);
    PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_QUALITY);
    args.seed = 0;
    args._k = numPart;
    // args.final_imbal = 0.05;  // completely ignore balance
    // args.final_imbal = 1;  // completely ignore balance
    args.final_imbal = st::setting.pt_imbl;

    args.MemMul_Pins = 1000;
    // args.MemMul_CellNet = 3;
    // args.MemMul_General = 2;
    PaToH_Check_User_Parameters(&args, true);

    int *partvec = new int[nNode];
    int *partweights = new int[numPart];
    int cut;
    PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, xpins, pins);

    for (int i = 0; i < nNode; i++) partvec[i] = nodes[i]->group;
    // for (int i = 0; i < nNode; i++) partvec[i] = -1; // FIXME:

    logger.info("Partitioner::run_patoh, %s", "finish setting up");

    float *targetweights = new float[2];
    double ratio = data.tech_ratio.item<double>() + st::setting.top_util_filler;
    targetweights[0] = ratio;
    targetweights[1] = 1 - ratio;  // TODO:

    
    // if(!is_move_macro) {
    //     useFixCells = true;
    //     for (int i = 0; i < nNode; i++) {
    //         if(data.macro_mask[i].item<int>() == 1) {
    //             partvec[i] = data.node_die[i].item<int>();
    //             std::cout << i << " " << data.node_die[i].item<int>() << std::endl; 
    //         } else {
    //             partvec[i] = -1;
    //         }
    //     }
    // }

    PaToH_Part(
        &args, _c, _n, _nconst, useFixCells, cwghts, nwghts, xpins, pins, targetweights, partvec, partweights, &cut);
    // PaToH_Partition_with_FixCells(&args, _c, _n, cwghts, nwghts, xpins, pins, partvec, partweights, &cut);

    logger.info("============ Original partition result ============");
    for (int i = 0; i < nNode; i++) {
        node_die[i] = partvec[i];
    }
    logger.info("Partition ratio: %d : %d", partweights[0], partweights[1]);
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));

    if(is_move_macro) {
        for (int i = 0; i < nNode; i++) {
            if(data.macro_mask[i].item<int>() == 1) {
                int group = partvec[i];
                //mov_cell_areas[group] += nodes[i]->sizes[group];
                nodes[i]->group = group;
                node_die[i] = group;
                std::cout << i << " " << group << std::endl; 
            }
        }
    }
    //for(int stage = 0; stage < 2; stage++) {


    if (!is_move_macro) {
        mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
        auto [sorted_tensor, indices] = (data.node_size).index({torch::indexing::Slice(), 0}).sort(-1, true);
        for (int n = 0; n < nNode; n++) {
            int i = indices[n].item<int>();

            int group = partvec[i];
            if (data.macro_mask[i].item<int>() == 1) {
                group = nodes[i]->group;
            }
            // TODO: force balance
            if (st::setting.clamp_util) {
                if ((mov_cell_areas[group] + nodes[i]->sizes[group] > max_mov_cell_areas[group]).item<bool>())
                    group = !group;
            }
            mov_cell_areas[group] += nodes[i]->sizes[group];
            nodes[i]->group = group;
            node_die[i] = group;
            if(data.macro_mask[i].item<int>() == 1) {
                std::cout << i << " " << group << std::endl; 
            }
        }
    }
    

    data.node_die = node_die.clone();  // TODO: construct data from pt
    data.mov_cell_areas = mov_cell_areas.clone();

    logger.info("============ Legalized partition result ============");
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));
    logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)",
                (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());

    free(cwghts);
    free(nwghts);
    free(xpins);
    free(pins);
    free(partweights);
    free(partvec);
    PaToH_Free();
}

void Partitioner::run_patoh_area(NodeData &data) {
    logger.info("============= Running PaToH ============");

    int nNode = nodes.size();
    int _nconst = 2;
    int *cwghts = new int[nNode * _nconst];
    for (int i = 0; i < nNode; i++) {
        cwghts[i * _nconst] = static_cast<int>((nodes[i]->sizes[0] + nodes[i]->sizes[1]) / 100 / data.site_width);
        for (int j = 1; j < _nconst; j++) {
            cwghts[i * _nconst + j] = 1;
        }
    }
    int nNets = nets.size();
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
    int useFixCells = false;

    int nPin = 0;
    for (auto net : nets) {
        nPin += net->Nodes.size();
    }

    int *xpins = new int[_n + 1];
    int *pins = new int[nPin];

    for (int i = 0, p = 0; i < _n; i++) {
        auto net = nets[i];
        for (auto node : net->Nodes) {
            pins[p++] = node->id;
        }
        xpins[i] = p;
    }
    xpins[_n] = nPin;

    PaToH_Parameters args;
    // PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_QUALITY);
    PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_QUALITY);
    args.seed = 0;
    args._k = numPart;
    // args.final_imbal = 0.05;  // completely ignore balance
    // args.final_imbal = 1;  // completely ignore balance
    args.final_imbal = st::setting.pt_imbl;

    args.MemMul_Pins = 1000;
    // args.MemMul_CellNet = 3;
    // args.MemMul_General = 2;
    PaToH_Check_User_Parameters(&args, true);

    int *partvec = new int[nNode];
    int *partweights = new int[numPart * _nconst];
    int cut;
    PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, xpins, pins);

    // for (int i = 0; i < nNode; i++) partvec[i] = nodes[i]->group;
    for (int i = 0; i < nNode; i++) partvec[i] = -1; // FIXME:

    logger.info("Partitioner::run_patoh, %s", "finish setting up");

    float *targetweights = new float[2 * _nconst];
    double ratio = data.tech_ratio.item<double>() + st::setting.top_util_filler;
    for (int i = 0; i < _nconst; i++) {
        targetweights[0 + i * 2] = ratio;
        targetweights[1 + i * 2] = 1 - ratio; 
    }

    PaToH_Part(
        &args, _c, _n, _nconst, useFixCells, cwghts, nwghts, xpins, pins, targetweights, partvec, partweights, &cut);
    // PaToH_Partition_with_FixCells(&args, _c, _n, cwghts, nwghts, xpins, pins, partvec, partweights, &cut);

    logger.info("============ Original partition result ============");
    for (int i = 0; i < nNode; i++) {
        node_die[i] = partvec[i];
    }
    logger.info("Partition ratio: %d : %d", partweights[0], partweights[1]);
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));

    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    for (int i = 0; i < nNode; i++) {
        int group = partvec[i];
        // TODO: force balance
        if (st::setting.clamp_util) {
            if ((mov_cell_areas[group] + nodes[i]->sizes[group] > max_mov_cell_areas[group]).item<bool>())
                group = !group;
        }
        mov_cell_areas[group] += nodes[i]->sizes[group];
        nodes[i]->group = group;
        node_die[i] = group;
    }

    data.node_die = node_die.clone();  // TODO: construct data from pt
    data.mov_cell_areas = mov_cell_areas.clone();

    logger.info("============ Legalized partition result ============");
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));
    logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)",
                (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());

    free(cwghts);
    free(nwghts);
    free(xpins);
    free(pins);
    free(partweights);
    free(partvec);
    PaToH_Free();
}

void Partitioner::run_patoh(NodeData &data, torch::Tensor net_wgt_grad) {
    logger.info("============= Running PaToH ============");

    int nNode = nodes.size();
    int *cwghts = new int[nNode];
    for (int i = 0; i < nNode; i++) cwghts[i] = 1;

    int nNets = nets.size();
    int *nwghts = new int[nNets];
    for (int i = 0; i < nNets; i++) {
        if (net_to_num_pins[i].item<int>() <= st::setting.cut_net_thres)
            if (net_wgt_grad[i + nNode].item<float>() > 1) {
                nwghts[i] = (int)(1 * st::setting.net_weight_offset);
            } else
                nwghts[i] = (int)(0 * st::setting.net_weight_offset);
        else
            nwghts[i] = st::setting.net_weight_offset;
    }
    logger.info("Nets larger than %d will be applied with wa force %.3f",
                st::setting.cut_net_thres,
                st::setting.net_weight_coef);

    int _c = nNode;
    int _n = nNets;
    int _nconst = 1;
    int useFixCells = false;

    int nPin = 0;
    for (auto net : nets) {
        nPin += net->Nodes.size();
    }

    int *xpins = new int[_n + 1];
    int *pins = new int[nPin];

    for (int i = 0, p = 0; i < _n; i++) {
        auto net = nets[i];
        for (auto node : net->Nodes) {
            pins[p++] = node->id;
        }
        xpins[i] = p;
    }
    xpins[_n] = nPin;

    PaToH_Parameters args;
    // PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_QUALITY);
    PaToH_Initialize_Parameters(&args, PATOH_CUTPART, PATOH_SUGPARAM_QUALITY);
    args.seed = 0;
    args._k = numPart;
    // args.final_imbal = 0.05;  // completely ignore balance
    // args.final_imbal = 1;  // completely ignore balance
    args.final_imbal = st::setting.pt_imbl;

    args.MemMul_Pins = 1000;
    // args.MemMul_CellNet = 3;
    // args.MemMul_General = 2;
    PaToH_Check_User_Parameters(&args, true);

    int *partvec = new int[nNode];
    int *partweights = new int[numPart];
    int cut;
    PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts, xpins, pins);

    for (int i = 0; i < nNode; i++) partvec[i] = nodes[i]->group;

    logger.info("Partitioner::run_patoh", "finish setting up");

    float *targetweights = new float[2];
    double ratio = data.tech_ratio.item<double>();
    targetweights[0] = ratio;
    targetweights[1] = 1 - ratio;  // TODO:

    PaToH_Part(
        &args, _c, _n, _nconst, useFixCells, cwghts, nwghts, xpins, pins, targetweights, partvec, partweights, &cut);
    // PaToH_Partition_with_FixCells(&args, _c, _n, cwghts, nwghts, xpins, pins, partvec, partweights, &cut);

    logger.info("============ Original partition result ============");
    for (int i = 0; i < nNode; i++) {
        node_die[i] = partvec[i];
    }
    logger.info("Partition ratio: %d : %d", partweights[0], partweights[1]);
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));

    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    for (int i = 0; i < nNode; i++) {
        int group = partvec[i];
        // TODO: force balance
        if (st::setting.clamp_util) {
            if ((mov_cell_areas[group] + nodes[i]->sizes[group] > max_mov_cell_areas[group]).item<bool>())
                group = !group;
        }
        mov_cell_areas[group] += nodes[i]->sizes[group];
        nodes[i]->group = group;
        node_die[i] = group;
    }

    data.node_die = node_die.clone();  // TODO: construct data from pt
    data.mov_cell_areas = mov_cell_areas.clone();

    logger.info("============ Legalized partition result ============");
    rpt_cut_size();
    logger.info("Weights < %.2f <-- %.2f >", ratio, (1 - node_die).sum().item<float>() / node_die.size(0));
    logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)",
                (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());

    free(cwghts);
    free(nwghts);
    free(xpins);
    free(pins);
    free(partweights);
    free(partvec);
    PaToH_Free();
}
