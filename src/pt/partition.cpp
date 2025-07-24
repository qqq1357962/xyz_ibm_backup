#include "partition.h"

Partitioner::Partitioner(NodeData& data_, states& hpwl_state_) : data(data_), hpwl_state(hpwl_state_) {
    /* general info */
    device = data.device;
    num_nodes = data.num_nodes;
    num_nets = data.hyperedge_list_end.size(0);
    num_pins = data.hyperedge_list.size(0);
    net_to_num_nodes = torch::zeros({num_nets}, dtype(torch::kInt));
    net_to_num_pins = data.net_to_num_pins.clone();

    /* raw net/node graph */
    vector<set<int>> netCells;
    nodes.reserve(num_nodes);
    nets.reserve(num_nets);
    netCells.resize(num_nets);

    /* add cell */
    node_areas = torch::_cast_Long(torch::cat({data.node_area_bot.unsqueeze(0), data.node_area_top.unsqueeze(0)}, 0));
    int max_area=0;
    for (int i = 0; i != num_nodes; i++) {
        shared_ptr<ptNode> node = make_shared<ptNode>(i);
        nodes.push_back(node);
        node->sizes.push_back(data.node_area_bot[i].item<long>());
        node->sizes.push_back(data.node_area_top[i].item<long>());
        if(node->sizes[0]>max_area)
        {
            max_area=node->sizes[0];
        }
        if(node->sizes[1]>max_area)
        {
            max_area=node->sizes[1];
        }
    }
    logger.info("max area,, %d",max_area);
    for (unsigned i = 0; i != num_nets; ++i) {
        int64_t start_idx = 0;
        if (i != 0) start_idx = data.hyperedge_list_end[i - 1].item().toInt();
        int64_t end_idx = data.hyperedge_list_end[i].item().toInt();
        int64_t degree = end_idx - start_idx;  // degree count
        maxDegree = max(maxDegree, degree);
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int64_t pin_id = data.hyperedge_list[idx].item().toInt();
            int64_t node_id = data.pin_id2node_id[pin_id].item().toInt();
            netCells[i].insert(node_id);
        }
    }
    /* add net */
    for (unsigned i = 0; i != num_nets; ++i) {
        shared_ptr<ptNet> net = make_shared<ptNet>(i);
        nets.push_back(net);
        net_to_num_nodes[i] = (int)netCells[i].size();
        for (int nodeID : netCells[i]) {
            auto node = nodes[nodeID];
            net->Nodes.push_back(node);
            node->Nets.push_back(net);
        }
    }
    maxDegree = 0;
    for (unsigned i = 0; i != num_nodes; ++i) {
        int64_t count = 0;
        for (auto net : nodes[i]->Nets) {
            count++;
        }
        maxDegree = max(maxDegree, count);
    }

    /* Max utilizations */
    upper_lower_bound_ratio = data.upper_lower_bound_ratio;
    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    max_mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    max_mov_cell_areas[0] = data.mov_cell_area_bot_bound;
    max_mov_cell_areas[1] = data.mov_cell_area_top_bound;
    node_die = torch::zeros(num_nodes, torch::dtype(torch::kInt));

    max_hpwl = (data.die_info[1] + data.die_info[3]).item<float>();

    /* random partition */
    for (int i = 0; i != num_nodes; i++) {
        double rand_seed = (double)rand() / RAND_MAX;
        if ((rand_seed > data.tech_ratio.item<double>()) &&
            ((mov_cell_areas[1] + nodes[i]->sizes[1]) < max_mov_cell_areas[1]).item<bool>()) {
            node_die[i] = 1;
            nodes[i]->group = 1;
            mov_cell_areas[1] += nodes[i]->sizes[1];
        } else {
            nodes[i]->group = 0;
            mov_cell_areas[0] += nodes[i]->sizes[0];
        }
    }

    /* adjust utilization */
    /*
    at::Tensor die_area = at::prod(data.die_ur - data.die_ll);
    float bot_util_adjust = mov_cell_areas[0].item<float>() / die_area.item<float>();
    float top_util_adjust = mov_cell_areas[1].item<float>() / die_area.item<float>();
    logger.info("Random partition: bot_util %.2f, top_util %.2f", bot_util_adjust, top_util_adjust);
    // logger.info("Max Utilization: bot %.2f, top %.2f", data.maxUtilM[0].item<float>(), data.maxUtilM[1].item<float>());
    
    // auto node_area_0_at = mov_cell_areas[0].accessor<float, 1>();
    // auto node_area_1_at = mov_cell_areas[1].accessor<float, 1>();

    auto mov_cell_areas_at = mov_cell_areas.accessor<int64_t, 1>();
    auto macro_mask_at = data.macro_mask.accessor<float, 1>();
    for (int adjust_iter = 0; adjust_iter < 1000 && 
    (mov_cell_areas_at[1] > data.mov_cell_area_top_bound || 
    mov_cell_areas_at[0] > data.mov_cell_area_bot_bound); adjust_iter++)
    {
        
        for (int i = 0; i != num_nodes 
            && mov_cell_areas_at[0] > data.mov_cell_area_bot_bound; i++)
        {
            if (nodes[i]->group == 0 && macro_mask_at[i] == 0)
            {
                node_die[i] = 1;
                nodes[i]->group = 1;
                mov_cell_areas_at[1] += nodes[i]->sizes[1];
                mov_cell_areas_at[0] -= nodes[i]->sizes[0];
            } 
        }

        for (int i = 0; i != num_nodes 
            && mov_cell_areas_at[1] > data.mov_cell_area_top_bound; i++)
        {
            if (nodes[i]->group == 1 && macro_mask_at[i] == 1)
            {
                node_die[i] = 0;
                nodes[i]->group = 0;
                mov_cell_areas_at[1] -= nodes[i]->sizes[1];
                mov_cell_areas_at[0] += nodes[i]->sizes[0];
            }
        }
    }

    torch::Tensor tensor1 = torch::tensor({5, 7, 10}, torch::kInt32);
    torch::Tensor tensor2 = torch::tensor({2, 3, 4}, torch::kInt32);
    
    torch::Tensor result = tensor1.to(torch::kFloat32) / tensor2;
    
    std::cout << "Result: " << result << std::endl;

    // auto ratio_nodes = data.node_area_bot/data.node_area_top;
    auto ratio_nodes = data.node_area_bot.to(torch::kFloat32)/data.node_area_top.to(torch::kFloat32);
    auto [sorted_tensor, indices] = ratio_nodes.sort();
    int previous_index=10;
    for(int i=0;i<indices.size(0);i++)
    {
        int node_id = indices[i].item<int>();
        bool isEqual = torch::equal(data.node_size_bot[previous_index], data.node_size_bot[node_id]);
        if(isEqual)
        {
            continue;
        }
        previous_index=node_id;
        logger.info("index: %d, node_id: %d, is_macro: %d, ratio: %f, area_bot: %f", i, node_id, 
                                                             data.macro_mask[node_id].item<int>(), ratio_nodes[node_id].item<float>(), 
                                                             data.node_area_bot[node_id].item<float>());
    }


    bot_util_adjust = mov_cell_areas[0].item<float>() / die_area.item<float>();
    top_util_adjust = mov_cell_areas[1].item<float>() / die_area.item<float>();
    logger.info("Adjust after random partition: bot_util %.2f, top_util %.2f", bot_util_adjust, top_util_adjust);
    */
    logger.info("Max Utilization: bot %.2f, top %.2f", data.maxUtilM[0].item<float>(), data.maxUtilM[1].item<float>());

    pin_id2node_id = data.pin_id2node_id.clone();
    hyperedge_list = data.hyperedge_list.clone();
    hyperedge_list_end = data.hyperedge_list_end.clone();
    pin_id2net_id = data.pin_id2net_id.clone();

}  // END MODULE

//-------------------------------------------------------------------------------
void Partitioner::greedy_macro_partition(NodeData& data) {
    for (int i = 0; i < data.node_pos.size(0); i++) {
        if (data.macro_mask[i].item<int>() == 1) {
            node_die[i] = 0;
        }
    }

    node_die[109812] = 1;
    node_die[94575] = 1;
    node_die[8466] = 1;
    node_die[102136] = 1;
    node_die[102139] = 1;

    node_die[32010] = 1;
    node_die[86867] = 1;
    node_die[86866] = 1;
    node_die[94579] = 1;
    node_die[102138] = 1;
    node_die[86869] = 1;
    node_die[16428] = 1;
    node_die[102140] = 1;
    node_die[109811] = 1;
    node_die[39783] = 1;
    node_die[39784] = 1;
    node_die[86865] = 1;
    node_die[55551] = 1;
    node_die[94578] = 1;
    node_die[63403] = 1;
    node_die[71293] = 1;
    node_die[8465] = 1;
    // for (int i = 0; i < data.node_pos.size(0); i++) {
    //     if (data.macro_mask[i].item<int>() == 1) {
    //         node_die[i] = 0;
    //     }
    // }
    // vector<int> top_macros;
    // for (int i = 0; i < data.node_pos.size(0); i++) {
    //     float xl1 = data.node_pos[i][0].item<float>() - data.node_size[i][0].item<float>() / 2.8;  //@@FREEZE
    //     float yl1 = data.node_pos[i][1].item<float>() - data.node_size[i][1].item<float>() / 2.8;

    //     float xh1 = data.node_pos[i][0].item<float>() + data.node_size[i][0].item<float>() / 2.8;
    //     float yh1 = data.node_pos[i][1].item<float>() + data.node_size[i][0].item<float>() / 2.8;

    //     if (data.macro_mask[i].item<int>() == 1) {
    //         int flag = 0;
    //         for (auto macro_id : top_macros) {
    //             float xl2 = data.node_pos[macro_id][0].item<float>() - data.node_size[macro_id][0].item<float>() / 2;
    //             float yl2 = data.node_pos[macro_id][1].item<float>() - data.node_size[macro_id][1].item<float>() / 2;
    //             float xh2 = data.node_pos[macro_id][0].item<float>() + data.node_size[macro_id][0].item<float>() / 2;
    //             float yh2 = data.node_pos[macro_id][1].item<float>() + data.node_size[macro_id][0].item<float>() / 2;

    //             if (std::min(xh1, xh2) > std::max(xl1, xl2) && std::min(yh1, yh2) > std::max(yl1, yl2)) {
    //                 flag = 1;
    //                 break;
    //             }
    //         }
    //         if (flag == 0) {
    //             top_macros.push_back(i);
    //             node_die[i] = 1;
    //             logger.info("put macro %d on top die pos: (%f,%f), (%f,%f)", i, xl1, yl1, xh1, yh1);
    //         }
    //     }
    // }
}

void Partitioner::greedy_partition_by_area(NodeData& data) {
    auto node_area_top = torch::prod(data.node_size_top, 1);
    auto [sorted_tensor, indices] = data.node_area_bot.slice(0,0,data.cell_mov_rhs).sort();
    float bound = data.max_mov_cell_areas[0].item<float>();
    float area0=0;
    auto  area1=0;
    for(int i=0;i<data.cell_mov_rhs;i++){
         node_die[i] = 1;
         area1+=data.node_area_top[i].item<float>();
    }
    for(int i=0;i<data.cell_mov_rhs;i++)
    {
        int index = indices[i].item<int>();
        if(area0<area1)
        {
            node_die[index] = 0;
            area0+=data.node_area_bot[index].item<float>();
            area1-=data.node_area_top[index].item<float>();
        }else{
            break;
        }
    }
    logger.info("after greedy partition, area_top=%f, area_bot=%f",area1, area0);
}

void Partitioner::greedy_partition_by_std_area2(NodeData& data) {
    auto node_area_top = torch::prod(data.node_size_top, 1);
    auto [sorted_tensor, indices] = data.node_area_bot.slice(0,0,data.cell_mov_rhs).sort();
    float bound0 = data.max_mov_cell_areas[0].item<float>();
    float bound1 = data.max_mov_cell_areas[1].item<float>();
    float area0=0;
    auto area1 = data.mov_cell_area[1].item<float>();
    for(int i=0;i<data.cell_mov_rhs;i++){
        if(data.macro_mask[i].item<int>()==1)
        {
            continue;
        }
        else{
            node_die[i] = 1;
            area1+=data.node_area_top[i].item<float>();
        }
    }
    for(int i=0;i<data.cell_mov_rhs;i++)
    {
        if(data.macro_mask[i].item<int>()==1)
        {
            continue;
        }
        else{
            int index = indices[i].item<int>();
            if(area0<area1)
            {
                node_die[index] = 0;
                area0+=data.node_area_bot[index].item<float>();
                area1-=data.node_area_top[index].item<float>();
            }else{
                break;
            }
        }  
    }
    logger.info("after std greedy partition, std_area_top=%f, std_area_bot=%f",area1, area0);
    for(int i=0;i<data.cell_mov_rhs;i++){
        if(data.macro_mask[i].item<int>()==0)
        {
            continue;
        }
        if(area1+data.node_area_top[i].item<float>()<bound1)
        {
            node_die[i] = 1;
            area1+=data.node_area_top[i].item<float>();
        }else{
            node_die[i] = 0;
            area0+=data.node_area_bot[i].item<float>();
        }
    }
    logger.info("after greedy partition, area_top=%f, area_bot=%f",area1, area0);
}

// an area calculation code copy from csdn 
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 111;
#define ls i<<1
#define rs i<<1|1
#define m(i) ((q[i].l + q[i].r)>>1)
struct myEdge
{
    double l,r;//这条线的左右端点的横坐标
    double h;//这条线的纵坐标
    int f;//这条线是矩形的上边还是下边
}e[N<<1];
bool cmp(myEdge a,myEdge b)
{
    return a.h < b.h;
}
struct Node
{
    int l,r;//横坐标的区间，是横坐标数组的下标
    int s;//该节点被覆盖的情况（是否完全覆盖）
    double len;//该区间被覆盖的总长度
}q[N*8];
double x[2*N];//横坐标
void build(int i,int l,int r)
{
    q[i].l = l,q[i].r = r;
    q[i].s = 0;q[i].len = 0;
    if (l == r) return;
    int mid = m(i);
    build(ls,l,mid);
    build(rs,mid+1,r);
}
void pushup(int i)
{
    if (q[i].s) //非零，已经被整段覆盖
    {
        q[i].len = x[q[i].r+1] - x[q[i].l];
    }
    else if (q[i].l == q[i].r) //这是一个点而不是线段
    {
        q[i].len = 0;
    }
    else //是一条没有整个区间被覆盖的线段，合并左右子的信息
    {
        q[i].len = q[ls].len + q[rs].len;
    }
}
void update(int i,int l,int r,int xx)//这里深刻体会为什么令下边为1，上边-1
{                                   //下边插入边，上边删除边
    if (q[i].l == l&&q[i].r == r)
    {
        q[i].s += xx;
        pushup(i);//更新区间被覆盖de总长度
        return;
    }
    int mid = m(i);
    if (r <= mid) update(ls,l,r,xx);
    else if (l > mid) update(rs,l,r,xx);
    else
    {
        update(ls,l,mid,xx);
        update(rs,mid+1,r,xx);
    }
    pushup(i);
}
struct node{
	int lx,ly,rx,ry;
};
double calc_area(vector<node> rectangles)
{
    int n;
    n = rectangles.size();
    int tot = 0;
    for (int i = 0;i < n;++i)
    {
        double x1,x2,y1,y2;
        x1 = rectangles[i].lx;
        y1 = rectangles[i].ly;
        x2 = rectangles[i].rx;
        y2 = rectangles[i].ry;

        myEdge &t1 = e[tot];myEdge &t2 = e[1+tot];
        t1.l = t2.l = x1,t1.r = t2.r = x2;
        t1.h = y1;t1.f = 1;
        t2.h = y2;t2.f = -1;
        x[tot] = x1;x[tot+1] = x2;
        tot += 2;
    }
    sort(e,e+tot,cmp);//边按高度从小到大排序（自下而上扫描）
    sort(x,x+tot);
    //离散化横坐标
    int k = 1;
    for (int i = 1;i < tot;++i)
    {
        if (x[i] != x[i-1]) //去重
        {
            x[k++] = x[i];
        }
    }
    build(1,0,k-1);//离散化后的区间是[0，k-1]
    double ans = 0.0;
    for (int i = 0;i < tot;++i)
    {
        //因为线段树维护的是横坐标们的下标，所以对每条边求出其两个横坐标对应的下标
        int l = lower_bound(x,x+k,e[i].l) - x;//在横坐标数组里找到这条边的位置
        int r = lower_bound(x,x+k,e[i].r) - x - 1;
        update(1,l,r,e[i].f);//每扫到一条边就更新横向的覆盖len
        ans += (e[i+1].h - e[i].h)*q[1].len;//q[1]是整个区间,q[1].k=len是整个区间的有效长度
        //计算面积就是用区间横向的有效长度乘以两条边的高度差（面积是两条边里面的部分）
    }
    return ans;
}
void Partitioner::adjust_macros(NodeData& data) {


    vector<int> macro_list_top;
    vector<int> macro_list_bot;
    for (int i = 0; i < data.cell_mov_rhs; i++) {
        if (data.macro_mask[i].item<int>() == 1) {
            if(node_die[i].item<int>() == 1) {
                macro_list_top.push_back(i);
            }
            else {
                macro_list_bot.push_back(i);
            }
        }
    }
    int perturb_iter = 3000;
    double area_previous=1e31;
    if (macro_list_top.size() && macro_list_bot.size()) {
        for(int i=0;i<perturb_iter;i++)
        {
            int index1 = random()%macro_list_top.size();
            int index0 = random()%macro_list_bot.size();
            int back_up = macro_list_bot[index0];
            macro_list_bot[index0] = macro_list_top[index1];
            macro_list_top[index1] = back_up;
            vector<node> rectangles_bot;
            vector<node> rectangles_top;
            float area0=0;
            for(auto macro_id:macro_list_bot)
            {
                int lx,ly,rx,ry;
                lx = data.node_pos[macro_id][0].item<int>()-data.node_size_bot[macro_id][0].item<int>()/2;
                ly = data.node_pos[macro_id][1].item<int>()-data.node_size_bot[macro_id][1].item<int>()/2;
                rx = data.node_pos[macro_id][0].item<int>()+data.node_size_bot[macro_id][0].item<int>()/2;
                ry = data.node_pos[macro_id][1].item<int>()+data.node_size_bot[macro_id][1].item<int>()/2;
                rectangles_bot.push_back(node({lx,ly,rx,ry}));
                area0+=data.node_size_bot[macro_id][0].item<int>()*data.node_size_bot[macro_id][1].item<int>();
            }
            float area1=0;
            for(auto macro_id:macro_list_top)
            {
                int lx,ly,rx,ry;
                lx = data.node_pos[macro_id][0].item<int>()-data.node_size_top[macro_id][0].item<int>()/2;
                ly = data.node_pos[macro_id][1].item<int>()-data.node_size_top[macro_id][1].item<int>()/2;
                rx = data.node_pos[macro_id][0].item<int>()+data.node_size_top[macro_id][0].item<int>()/2;
                ry = data.node_pos[macro_id][1].item<int>()+data.node_size_top[macro_id][1].item<int>()/2;
                rectangles_top.push_back(node({lx,ly,rx,ry}));
                area1+=data.node_size_bot[macro_id][0].item<int>()*data.node_size_bot[macro_id][1].item<int>();
            }
            double area = calc_area(rectangles_bot)+calc_area(rectangles_top);
            // float area = calc_area(macro_list_bot,data.node_size_bot,data.node_pos)+calc_area(macro_list_top,data.node_size_top,data.node_pos);
            // float area=0;
            double area_overlap = 2*(area0+area1)-area;
            if(area_overlap<area_previous)
            {
                area_previous = area_overlap;
            }else{
                int back_up = macro_list_bot[index0];
                macro_list_bot[index0] = macro_list_top[index1];
                macro_list_top[index1] = back_up;
            }
        }
    }
    
    for(int i=0;i<macro_list_bot.size();i++)
    {
        node_die[macro_list_bot[i]]=0;
    }
    for(int i=0;i<macro_list_top.size();i++)
    {
        node_die[macro_list_top[i]]=1;
    }
    // for(int i=0;i<macro_list_bot.size();i++)
    // {
    //     node_die[macro_list_bot[i]]=1;
    // }




    auto ratio_nodes = data.node_area_bot.to(torch::kFloat32)/data.node_area_top.to(torch::kFloat32);
    auto [sorted_tensor, indices] = ratio_nodes.sort();
    float bound0 = data.max_mov_cell_areas[0].item<float>();
    float bound1 = data.max_mov_cell_areas[1].item<float>();
    float area0=0;
    float area1=0;
    for(int i=0;i<macro_list_bot.size();i++)
    {
        node_die[macro_list_bot[i]]=1;
        area1+=data.node_area_top[macro_list_bot[i]].item<float>();
    }
    for(int i=0;i<macro_list_top.size();i++)
    {
        node_die[macro_list_top[i]]=1;
        area1+=data.node_area_top[macro_list_top[i]].item<float>();
    }
    for (int i = 0; i < data.cell_mov_rhs; i++) {
        int index = indices[i].item<int>();
        if (data.macro_mask[index].item<int>() == 0) {
            area1+=data.node_area_top[index].item<float>();
            node_die[index] = 1;
        }
    }
    logger.info("@putting std cells to top area_top=%f/%f, area_bot=%f/%f",area1,bound1,area0,bound0);

    float balance;
    
    balance = abs(area1-area0);
    int i_std=0;
    for(i_std=0;i_std<data.cell_mov_rhs;i_std++){
        int index = indices[i_std].item<int>();
        if(data.macro_mask[index].item<int>()==1)
        {
            continue;
        }
        float new_balance = abs((area1-data.node_area_top[index].item<float>())-(area0+data.node_area_bot[index].item<float>()));
        if(new_balance<balance)
        {
            balance=new_balance;
            node_die[index] = 0;
            area1-=data.node_area_top[index].item<double>();
            area0+=data.node_area_bot[index].item<double>();
        }else{
            break;
        }
    }
    logger.info("after checking, area_top=%f/%f, area_bot=%f/%f",area1,bound1,area0,bound0);
}

void Partitioner::greedy_partition_by_std_area(NodeData& data) {
    // auto node_area_top = torch::prod(data.node_size_top, 1);
    auto ratio_nodes = data.node_area_bot.to(torch::kFloat32)/data.node_area_top.to(torch::kFloat32);
    auto [sorted_tensor, indices] = ratio_nodes.sort();
    // auto [sorted_tensor, indices] = data.node_area_bot.slice(0,0,data.cell_mov_rhs).sort();
    float bound0 = data.max_mov_cell_areas[0].item<float>();
    float bound1 = data.max_mov_cell_areas[1].item<float>();
    float area0=0;
    float area1=0;
    vector<int> macro_list;
    for (int i = 0; i < data.cell_mov_rhs; i++) {
        int index = indices[i].item<int>();
        if (data.macro_mask[index].item<int>() == 1) {
            macro_list.push_back(index);
            area1+=data.node_area_top[index].item<float>();
            node_die[index] = 1;
        }
    }
    //macro_list: ratio从小到大放置的macro id 数组
    int index_macro=0;
    float balance = abs(area1-area0);
    for(index_macro=0;index_macro<macro_list.size();index_macro++)
    {
        int index = macro_list[index_macro];
        if (data.macro_mask[index].item<int>() == 0) {
            continue;
        }
        float new_balance = abs((area1-data.node_area_top[index].item<double>())-(area0+data.node_area_bot[index].item<double>()));
        // if(area0<area1)
        if(new_balance<balance)
        {
            balance=new_balance;
            node_die[index] = 0;
            area0+=data.node_area_bot[index].item<double>();
            area1-=data.node_area_top[index].item<double>();
        }else{
            break;
        }   
    }
    logger.info("after macro greedy partition, macro_area_top=%f, macro_area_bot=%f",area1, area0);//correct
    set<int> check1;
    /////////////////////////////////////////////////////////////////////
    // logger.info("rechecking0...");
    // float area0_check=0;
    // float area1_check=0;
    // for(int i=0;i<data.cell_mov_rhs;i++){
    //     if (data.macro_mask[i].item<int>() == 0) {
    //         continue;
    //     }
    //     if(node_die[i].item<int>() == 1)
    //     {
    //         area1_check+=data.node_area_top[i].item<double>();
    //     }else{
    //         area0_check+=data.node_area_bot[i].item<double>();
    //     }
    // }
    // logger.info("after checking, area_macro_top=%f/%f, area_macro_bot=%f/%f",area1_check,bound1,area0_check,bound0);//correct
    /////////////////////////////////////////////////////////////////////
    for(int i=0;i<data.cell_mov_rhs;i++){
        if(data.macro_mask[i].item<int>()==1)
        {
            continue;
        }
        node_die[i] = 1;
        area1+=data.node_area_top[i].item<double>();
    }
    logger.info("after puttig all std cells on the top, total_area_top=%f, total_area_bot=%f",area1,area0);
    /////////////////////////////////////////////////////////////////////
    // logger.info("rechecking1...");
    // area0_check=0;
    // area1_check=0;
    // for(int i=0;i<data.cell_mov_rhs;i++){
    //     if(node_die[i].item<int>() == 1)
    //     {
    //         area1_check+=data.node_area_top[i].item<double>();
    //     }else{
    //         area0_check+=data.node_area_bot[i].item<double>();
    //     }
    // }
    // logger.info("after checking, area_top=%f/%f, area_bot=%f/%f",area1_check,bound1,area0_check,bound0);
    /////////////////////////////////////////////////////////////////////
    balance = abs(area1-area0);
    int i_std=0;
    for(i_std=0;i_std<data.cell_mov_rhs;i_std++){
        int index = indices[i_std].item<int>();
        if(data.macro_mask[index].item<int>()==1)
        {
            continue;
        }
        float new_balance = abs((area1-data.node_area_top[index].item<float>())-(area0+data.node_area_bot[index].item<float>()));
        if(new_balance<balance)
        {
            balance=new_balance;
            node_die[index] = 0;
            area1-=data.node_area_top[index].item<double>();
            area0+=data.node_area_bot[index].item<double>();
        }else{
            break;
        }
    }
    logger.info("after greedy partition, area_top=%f/%f, area_bot=%f/%f",area1,bound1,area0,bound0);
    logger.info("rechecking2...");
    float area0_check=0;
    float area1_check=0;
    for(int i=0;i<data.cell_mov_rhs;i++){
        if(node_die[i].item<int>() == 1)
        {
            area1_check+=data.node_area_top[i].item<double>();
        }else{
            area0_check+=data.node_area_bot[i].item<double>();
        }
    }
    logger.info("after checking, area_top=%f/%f, area_bot=%f/%f",area1_check,bound1,area0_check,bound0);
    // here init partition finished, start check legality
    if(area1>bound1||area0>bound0)
    {
        logger.info("not legal, try to adjust!");
        // compare ratio of std cell and macro
        float std_ratio = ratio_nodes[i_std].item<float>();
        float macro_ratio = ratio_nodes[macro_list[index_macro]].item<float>();
        if(macro_ratio>std_ratio)
        {
            //inverse
            for(int j=index_macro;j>=0;j--)
            {
                int solved=0;
                int index = macro_list[j];
                if(node_die[index].item<int>()==1)
                {
                    continue;
                }
                node_die[index] = 1;
                area0-=data.node_area_bot[index].item<double>();
                area1+=data.node_area_top[index].item<double>();
                logger.info("after moving macro %d to top, area_top=%f/%f, area_bot=%f/%f",index,area1,bound1,area0,bound0);
                balance = abs(area1-area0);
                for(int ii=i_std;ii<data.cell_mov_rhs;ii++){
                    if(data.macro_mask[ii].item<int>()==1)
                    {
                        continue;
                    }
                    float new_balance = abs((area1-data.node_area_top[ii].item<float>())-(area0+data.node_area_bot[ii].item<float>()));
                    if(new_balance<balance)
                    {
                        if(node_die[ii].item<int>()==1)
                        {
                            balance=new_balance;
                            node_die[ii] = 0;
                            area1-=data.node_area_top[ii].item<double>();
                            area0+=data.node_area_bot[ii].item<double>();
                        }
                    }else{
                        break;
                    }
                }
                logger.info("after moving stds, area_top=%f/%f, area_bot=%f/%f",area1,bound1,area0,bound0);
                if(area0<=bound0 && area1<=bound1)
                {
                    break;
                }
            }
        }else{
            if(index_macro+1<macro_list.size())
            for(int j=index_macro+1;j<macro_list.size();j++)
            {
                int solved=0;
                int index = macro_list[j];
                if(node_die[index].item<int>()==0)
                {
                    continue;
                }
                node_die[index] = 0;
                area0+=data.node_area_bot[index].item<double>();
                area1-=data.node_area_top[index].item<double>();
                logger.info("after moving macro %d to bot, area_top=%f/%f, area_bot=%f/%f",index, area1,bound1,area0,bound0);
                balance = abs(area1-area0);
                for(int ii=i_std;ii>=0;ii--){
                    if(data.macro_mask[ii].item<int>()==1)
                    {
                        continue;
                    }
                    float new_balance = abs((area1+data.node_area_top[ii].item<float>())-(area0-data.node_area_bot[ii].item<float>()));
                    if(new_balance<balance)
                    {
                        if(node_die[ii].item<int>()==0)
                        {
                            balance=new_balance;
                            node_die[ii] = 1;
                            area1+=data.node_area_top[ii].item<double>();
                            area0-=data.node_area_bot[ii].item<double>();
                        }
                    }else{
                        break;
                    }
                }
                logger.info("after moving stds, area_top=%f/%f, area_bot=%f/%f",area1,bound1,area0,bound0);
                if(area0<=bound0 && area1<=bound1)
                {
                    break;
                }
            }
        }
    }
    logger.info("rechecking...");
    area0=0;
    area1=0;
    for(int i=0;i<data.cell_mov_rhs;i++){
        if(node_die[i].item<int>() == 1)
        {
            area1+=data.node_area_top[i].item<double>();
        }else{
            area0+=data.node_area_bot[i].item<double>();
        }
    }
    logger.info("after checking, area_top=%f/%f, area_bot=%f/%f",area1,bound1,area0,bound0);
}

torch::Tensor Partitioner::remove_macro_margin(NodeData& data, torch::Tensor node_pos, torch::Tensor node_size, torch::Tensor node_size_backup) {
    auto node_pos_tmp = node_pos.detach().clone();
    auto mov_node_sideline_ll = data.mov_node_sideline_ll;
    auto mov_node_sideline_ur = data.mov_node_sideline_ur;
    for(int i=0;i<data.cell_mov_rhs;i++)
    {
        if(data.macro_mask[i].item<int>()==0)
        {
            continue;
        }
        float w = node_size[i][0].item<float>();
        float h = node_size[i][1].item<float>();
        if(node_pos_tmp[i][0].item<float>()-w/2<=mov_node_sideline_ll[i][0].item<float>()+40)
        {
            logger.info("detect macro %d on the left", i);
            node_pos_tmp[i][0]=node_size_backup[i][0].item<float>()/2;
        }
        if(node_pos_tmp[i][1].item<float>()-h/2<=mov_node_sideline_ll[i][1].item<float>()+40)
        {
            logger.info("detect macro %d on the bottom", i);
            node_pos_tmp[i][1]=node_size_backup[i][1].item<float>()/2;
        }
        if(node_pos_tmp[i][0].item<float>()+w/2>=mov_node_sideline_ur[i][0].item<float>()-40)
        {
            logger.info("detect macro %d on the right", i);
            node_pos_tmp[i][0] = data.die_ur[0]-node_size_backup[i][0].item<float>()/2;
        }
        if(node_pos_tmp[i][1].item<float>()+h/2>=mov_node_sideline_ur[i][1].item<float>()-40)
        {
            logger.info("detect macro %d on the top", i);
            node_pos_tmp[i][1] = data.die_ur[1]-node_size_backup[i][1].item<float>()/2;
        }
    }
        return node_pos_tmp;
}


void Partitioner::greedy_partition_by_maximize_cuts(NodeData& data) {
    vector<int> node_die_tmp;
    node_die_tmp.resize(data.cell_mov_rhs);
    for(int i=0;i<node_die_tmp.size();i++)
    {
        node_die_tmp[i]=-1;
    }
    float bound0 = data.max_mov_cell_areas[0].item<float>();
    float bound1 = data.max_mov_cell_areas[1].item<float>();
    float area0=0;
    float area1=0;
    for(int i=0;i<data.num_nets;i++)
    {
        int64_t start_idx = 0;
        if (i != 0) {
            start_idx = data.hyperedge_list_end[i - 1].item<int64_t>();
        }
        int64_t end_idx = data.hyperedge_list_end[i].item<int64_t>();
        int cnt_node_die0=0;
        int cnt_node_die1=0;
        // calculate number on each die first
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int pin = data.hyperedge_list[idx].item<int>();
            int node_id = data.pin_id2node_id[pin].item<int>();
            int node_die_i = node_die_tmp[node_id];
            if(node_die_i==0)
            {
                cnt_node_die0++;
            }
            else if(node_die_i==1)
            {
                cnt_node_die1++;
            }
        }
        // greedy partition to another die
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int pin = data.hyperedge_list[idx].item<int>();
            int node_id = data.pin_id2node_id[pin].item<int>();
            if(node_die_tmp[node_id]>=0)
            {
                continue;
            }
            if(area0>=bound0)
            {
                node_die_tmp[node_id]=1;
                area1+=data.node_area_top[node_id].item<float>();
                cnt_node_die1++;
                continue;
            }else if(area1>=bound1){
                node_die_tmp[node_id]=0;
                area0+=data.node_area_bot[node_id].item<float>();
                cnt_node_die0++;
                continue;
            }
            if(node_id<=data.cell_mov_rhs)
            {
                int flag=2;
                if(cnt_node_die1<cnt_node_die0) flag=1;
                else if (cnt_node_die1>cnt_node_die0) flag=0;
                else if(cnt_node_die1==cnt_node_die0)
                {
                    // int die_id = random()%2;
                    // flag=die_id;
                    flag = area1<area0;
                }
                if(flag==1)
                {
                    node_die_tmp[node_id]=1;
                    area1+=data.node_area_top[node_id].item<float>();
                    cnt_node_die1++;
                }
                else if(flag==0){
                    node_die_tmp[node_id]=0;
                    area0+=data.node_area_bot[node_id].item<float>();
                    cnt_node_die0++;
                }
            }
        }
    }
    //write back to node_die
    for(int i=0;i<node_die_tmp.size();i++)
    {
        node_die[i] = node_die_tmp[i];
    }
    logger.info("after greedy partition, area_top=%f, area_bot=%f",area1, area0);
}

void Partitioner::pass() {
    int num_free = num_nodes;
    int GAIN_ITER = 0, GAIN_MAX = 0, maxGAINIndex = -1;

    // cout << mov_cell_areas[0].item<float>() << " " << mov_cell_areas[1].item<float>() << endl;

    for (int i = 0; i < num_nodes; i++) {
        // int cell_mov_idx = pop_max();
        int cell_mov_idx = i;

        // if (node_die[cell_mov_idx].item<int>() == 1) continue;
        if (cell_mov_idx == -1) break;
        auto cell_mov = nodes[cell_mov_idx];

        int gain = cell_mov->gain;
        GAIN_ITER += gain;

        freecells[cell_mov_idx] = 0;
        num_free--;
        update_gain(cell_mov);
        update_area(cell_mov);

        cell_mov->group = !cell_mov->group;
        tracker.push_back(cell_mov_idx);
        if (GAIN_ITER > GAIN_MAX) {
            GAIN_MAX = GAIN_ITER;
            maxGAINIndex = i;
        }

        /* early break */
        cutsize -= gain;
        if (st::setting.num_cuts > 0 && cutsize < st::setting.num_cuts) {
            running = false;
            break;
        }
    }

    // cout << mov_cell_areas[0].item<float>() << " " << mov_cell_areas[1].item<float>() << endl;
    // maxGAINIndex = tracker.size() - 1;

    GAIN += GAIN_MAX;
    shared_ptr<ptNode> cell;
    for (int i = maxGAINIndex + 1; i < tracker.size(); i++) {
        cell = nodes[tracker[i]];
        update_area(cell);
        cell->group = !cell->group;
    }

    logger.info("%d cells moved", maxGAINIndex + 1);
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::run() {
    st::setting.net_weight_coef = 0;
    run_patoh(data);
    logger.info("=================== Running fm-partition ===================");
    GAIN = 0;
    int64_t GAIN_dif = INT_MAX;
    GAINS.push_back(0);
    int iteration = 0;
    running = true;
    rpt_cut_size();
    for (int iteration = 0; iteration < 30 && running; iteration++) {
        initList(iteration != 0);
        pass();
        GAIN_dif = GAIN - GAINS.back();
        GAINS.push_back(GAIN);

        tracker.clear();
        logger.info("iter: %d | GAIN: %d", iteration, GAIN);
        rpt_cut_size();

        if (GAIN_dif < num_nodes / 1e4) break;
    }
    rpt_cut_size();

    /* dump to partition results */
    mov_cell_areas = torch::zeros(2, torch::dtype(torch::kLong));
    for (int i = 0; i < num_nodes; i++) {
        int group = nodes[i]->group;
        if ((mov_cell_areas[group] + nodes[i]->sizes[group] > max_mov_cell_areas[group]).item<bool>()) group = !group;
        mov_cell_areas[group] += nodes[i]->sizes[group];
        nodes[i]->group = group;
        node_die[i] = group;
    }

    logger.info("#Cells for each chip (%d, %d)", (1 - node_die).sum().item<int>(), node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)",
                (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());

    exit(1);
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::initList(bool update) {
    via_gainlist = torch::zeros(num_nodes, dtype(torch::kFloat));
    if (!update) {
        freecells = torch::ones(num_nodes, dtype(torch::kBool));
        // gainlist = torch::zeros(num_nodes, dtype(torch::kInt));
        BucketList.resize(2);
        maxGainIndex.resize(2);
        maxGainIndex = {0, 0};
        for (int i = 0; i < BucketList.size(); i++) {
            for (int j = -maxDegree; j <= maxDegree; j++) {
                BucketList[i][j] = make_shared<Vertex>("head");
            }
        }
        vertexList.resize(num_nodes);
    } else {
        freecells = torch::ones(num_nodes, dtype(torch::kBool));
        for (int i = 0; i < BucketList.size(); i++) {
            for (int j = -maxDegree; j <= maxDegree; j++) {
                BucketList[i][j]->next = NULL;  // FIXME:
            }
        }
    }
    for (auto cell : nodes) {
        int FS = 0, TE = 0;
        for (auto net : cell->Nets) {
            int fss = -1, tes = 0;
            for (auto cell_conn : net->Nodes) {
                int grouper = (cell->group == cell_conn->group);
                fss += grouper;
                tes += !grouper;
            }
            TE += (tes == 0);
            FS += (fss == 0);
        }
        cell->gain = FS - TE;
        via_gainlist[cell->id] = static_cast<float>(cell->gain);

        if (std::abs(cell->gain) > maxDegree) {
            cout << "========================== Cell ========================" << endl;
            cout << cell->Nets.size() << " connected nets\n";
            for (auto net : cell->Nets) {
                cout << "Net " << net->id << endl;
                for (auto cell_conn : net->Nodes) {
                    cout << cell_conn->id << " ";
                }
                cout << "\n\n";
            }
        }
        if (!update)
            addVertex(cell);
        else
            renewVertex(cell);
    }
}  // END MODULE

//-------------------------------------------------------------------------------

bool Partitioner::check_balance(int idx) {
    if (idx == -1) return false;
    shared_ptr<ptNode> cell_mov = nodes[idx];
    int group = cell_mov->group;
    if (((mov_cell_areas[!group] + cell_mov->sizes[!group]) < st::setting.soft_margin * max_mov_cell_areas[!group])
            .item<bool>()) {
        return true;
    } else
        return false;
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::update_area(shared_ptr<ptNode> cell_mov) {
    int group = cell_mov->group;

    mov_cell_areas[group] -= cell_mov->sizes[group];
    mov_cell_areas[!group] += cell_mov->sizes[!group];
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::swap(shared_ptr<ptNode> cell, int gain_offset) {
    int gain_index = cell->gain;
    cell->gain += gain_offset;
    via_gainlist[cell->id] = via_gainlist[cell->id] + static_cast<float>(gain_offset);
    vertexList[cell->id]->gain += gain_offset;
    // gainlist[cell->id] += gain_offset;
    update_vertex(cell->id, gain_index);
}  // END MODULE

//-------------------------------------------------------------------------------

int Partitioner::pop_max() {
    int idx_0, idx_1;
    idx_0 = (BucketList[0][maxGainIndex[0]]->next != NULL) ? BucketList[0][maxGainIndex[0]]->next->id : -1;
    idx_1 = (BucketList[1][maxGainIndex[1]]->next != NULL) ? BucketList[1][maxGainIndex[1]]->next->id : -1;

    bool check0 = check_balance(idx_0);
    bool check1 = check_balance(idx_1);

    if (!check0 && !check1)
        return -1;
    else if (check0) {
        if (check1) {
            auto cell_0 = nodes[idx_0];
            auto cell_1 = nodes[idx_1];
            return (cell_0->gain > cell_1->gain ? idx_0 : idx_1);
        } else
            return idx_0;
    } else
        return idx_1;
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::update_gain(shared_ptr<ptNode> cell_mov) {
    /* remove moved cell */
    rmVertex(cell_mov->id, cell_mov->gain);
    for (shared_ptr<ptNet> net : cell_mov->Nets) {
        int fss = -1, tes = 0;
        shared_ptr<ptNode> critical_cell_1;
        shared_ptr<ptNode> critical_cell_2;
        for (shared_ptr<ptNode> cell : net->Nodes) {
            int grouper = (cell->group == cell_mov->group);
            fss += grouper;
            tes += !grouper;
            if (grouper && (cell->id != cell_mov->id)) critical_cell_1 = cell;
            if (!grouper) critical_cell_2 = cell;
        }
        if ((fss == 1) && (critical_cell_1->id == -1)) {
            for (shared_ptr<ptNode> cell : net->Nodes) {
                int grouper = (cell->group == cell_mov->group);
                if (grouper && (cell->id != cell_mov->id)) critical_cell_1 = cell;
                if (!grouper) critical_cell_2 = cell;
            }
        }
        if (fss == 0) {
            for (shared_ptr<ptNode> cell : net->Nodes) {
                if (freecells[cell->id].item<bool>() && cell->id != cell_mov->id) swap(cell, -1);
            }
        } else if (fss == 1) {
            if (freecells[critical_cell_1->id].item<bool>()) swap(critical_cell_1, 1);
        }
        if (tes == 0) {  // FIXME: two-pin nets
            for (shared_ptr<ptNode> cell : net->Nodes) {
                if (freecells[cell->id].item<bool>()) swap(cell, 1);
            }
        } else if (tes == 1) {
            if (freecells[critical_cell_2->id].item<bool>()) swap(critical_cell_2, -1);
        }
    }
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::rpt_cut_size() {
    if (st::setting.partitioner == "fm" || st::setting.partitioner == "fm_grid") {
        // if (true) {
        cutsize = 0;
        for (auto net : nets) {
            int partition[2] = {0, 0};
            for (auto cell : net->Nodes) {
                partition[cell->group]++;
            }
            cutsize += ((partition[0] * partition[1]) > 0);
        }
    } else {
        // torch::Tensor net_cut_info = torch::zeros({num_nets, 2}, torch::dtype(torch::kInt));
        // for (int i = 0; i != num_nodes; i++) {
        //     auto node = nodes[i];
        //     int num_nets_node = node->Nets.size();
        //     for (unsigned n = 0; n != num_nets_node; ++n) {
        //         int net_id = node->Nets[n]->id;
        //         net_cut_info[net_id][node_die[i].item<int>()] += 1;
        //     }
        // }
        // cutsize = (torch::prod(net_cut_info, 1) != 0).sum().item<int>();

        torch::Tensor net_cut_info = torch::zeros({num_nets, 2}, torch::dtype(torch::kInt));
        const torch::TensorAccessor<int64_t, 1> pin_id2node_id_at = pin_id2node_id.accessor<int64_t, 1>();
        const torch::TensorAccessor<int64_t, 1> hyperedge_list_at = hyperedge_list.accessor<int64_t, 1>();
        const torch::TensorAccessor<int64_t, 1> hyperedge_list_end_at = hyperedge_list_end.accessor<int64_t, 1>();
        const torch::TensorAccessor<int, 1> cell_die_at = node_die.accessor<int, 1>();

        for (int i = 0; i != num_nets; i++) {
            int64_t start_idx = 0;
            if (i != 0) {
                start_idx = hyperedge_list_end_at[i - 1];
            }
            int64_t end_idx = hyperedge_list_end_at[i];
            if (end_idx != start_idx) {
                for (int64_t idx = start_idx; idx < end_idx; idx++) {
                    int64_t pin_id = hyperedge_list_at[idx];
                    int64_t node_id = pin_id2node_id_at[pin_id];

                    int c_id = cell_die_at[node_id];

                    net_cut_info[i][c_id] += 1;
                }
            }
        }
        cutsize = torch::_cast_Int(torch::prod(net_cut_info, 1) != 0).sum().item<int>();
    }

    logger.info("[Cutsize: %d]", cutsize);
}  // END MODULE

//-------------------------------------------------------------------------------

Box Partitioner::check_net_weight(NodeData& data, int i, torch::Tensor node_pos) {
    // if (data.net_to_num_pins[i].item<int>() > st::setting.cut_net_thres) return 1;

    Box box(std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max());
    int64_t start_idx = 0;
    if (i != 0) {
        start_idx = data.hyperedge_list_end[i - 1].item<int64_t>();
    }
    int64_t end_idx = data.hyperedge_list_end[i].item<int64_t>();
    if (end_idx != start_idx) {
        for (int64_t idx = start_idx; idx < end_idx; idx++) {
            int64_t pin_id = data.hyperedge_list[idx].item<int64_t>();
            int64_t node_id = data.pin_id2node_id[pin_id].item<int64_t>();

            box.xl = std::min(box.xl, (node_pos[node_id][0] + data.pin_rel_cpos[pin_id][0]).item<float>());
            box.xh = std::max(box.xh, (node_pos[node_id][0] + data.pin_rel_cpos[pin_id][0]).item<float>());
            box.yl = std::min(box.yl, (node_pos[node_id][1] + data.pin_rel_cpos[pin_id][1]).item<float>());
            box.yh = std::max(box.yh, (node_pos[node_id][1] + data.pin_rel_cpos[pin_id][1]).item<float>());
        }
    }

    return box;
}


struct MacroInfo {
    int id;
    int die;
    float x, y;
    float w[2], h[2];
    int counter;
};

float macro_overlap(MacroInfo &a, MacroInfo &b, bool swap = false) {
    int a_die = swap? 1 - a.die: a.die;
    if (a.x > b.x + b.w[b.die] || a.x + a.w[a_die] < b.x)
        return 0;
    if (a.y > b.y + b.h[b.die] || a.y + a.h[a_die] < b.y)
        return 0;
    float width = min(abs(a.x - (b.x + b.w[b.die])), abs((a.x + a.w[a_die]) - b.x));
    float height = min(abs(a.y - (b.y + b.h[b.die])), abs((a.y + a.h[a_die]) - b.y));
    return width * height;
}


void Partitioner::set_macro2d(NodeData& data) {
    //is_move_macro = false;
    std::vector<MacroInfo> m;
    m.reserve(torch::sum(data.macro_mask).item<int>());
    int count = 0;
    int cur_group = 0;
    float area[2] = {0, 0};

    
    for (int i = 0; i < num_nodes; i++) {
        if(data.macro_mask[i].item<int>() == 1) {
            MacroInfo tmp;
            tmp.id = i;
            count++;
            tmp.counter = 0;
            tmp.w[0] = data.node_size[i][0].item<float>();
            tmp.w[1] = data.node_size[i][0].item<float>();
            tmp.h[0] = data.node_size[i][1].item<float>();
            tmp.h[1] = data.node_size[i][1].item<float>();
            tmp.x = data.node_pos[i][0].item<float>() - data.node_size[i][0].item<float>() / 2;
            tmp.y = data.node_pos[i][1].item<float>() - data.node_size[i][1].item<float>() / 2;
            m.emplace_back(tmp);
            cur_group = 1 - cur_group;
        }
    }
    
    std::sort(m.begin(), m.end(), [](const MacroInfo &a, const MacroInfo &b)
    { 
        if (a.x < b.x) return true;
        if (a.x == b.x && a.y < b.y) return true;
        return false;
    });

    cur_group = 0;
    for(int i = 0; i < m.size(); i++) {
        m[i].die = cur_group;
        area[cur_group] += m[i].w[cur_group] * m[i].h[cur_group];
        if(area[cur_group] > area[1 - cur_group]) cur_group = 1 - cur_group;
        std::cout << m[i].id << "\n";
    }

    std::cout << "init" << "\n";

    cur_group = 0;
    for(int iter = 0; iter < m.size() * 6; iter++) {
        cur_group = area[0] > area[1] ? 1 : 0;
        int bestID = -1;
        float best_cost = 0;
        for(int i = 0; i < m.size(); i++) {
            float cost = 0;
            if (m[i].die != cur_group || m[i].counter > 4) {
                continue;
            }
            for(int j = 0; j < m.size(); j++) {
                if(i==j) continue;
                if(m[j].x > m[i].x + m[i].w[m[i].die]) break;
                if(m[i].x > m[j].x + m[j].w[m[j].die]) continue;
                if(m[i].die == m[j].die) {
                    cost += macro_overlap(m[i], m[j]);
                }
                else {
                    cost -= macro_overlap(m[i], m[j]);
                }
                if((m[i].id == 410300 && m[j].id == 399225)|| (m[i].id == 399225 && m[j].id ==  410300)) {
                    std::cout << m[i].id << " " << m[i].die <<" "<< m[j].id << " " << m[j].die  << " " << macro_overlap(m[i], m[j]) << " " << cost << "\n";
                }
            }
            if(cost > best_cost) {
                best_cost = cost;
                bestID = i;
            }
        }
        cur_group = 1 - cur_group;
        if(bestID >= 0){
            area[m[bestID].die] -= m[bestID].w[m[bestID].die] * m[bestID].h[m[bestID].die];
            m[bestID].die = 1 - m[bestID].die;
            m[bestID].counter++;
            area[m[bestID].die] += m[bestID].w[m[bestID].die] * m[bestID].h[m[bestID].die];
            
        } else {
            break;
        }
    }
    
    for(int i = 0; i < m.size(); i++) {

        //std::cout << m[i].id << " "<< m[i].die << " " << " " << "\n";
        data.node_die[m[i].id] = m[i].die;
        nodes[m[i].id]->group = m[i].die;
        node_die[m[i].id] = m[i].die;
    }
    m.clear();
}


void Partitioner::set_macro3d(NodeData& data) {
    //is_move_macro = false;
    std::vector<MacroInfo> m;
    m.reserve(torch::sum(data.macro_mask).item<int>());
    float area[2] = {0, 0};
    area[0] = mov_cell_areas[0].item<float>();
    area[1] = mov_cell_areas[1].item<float>();


    for (int i = 0; i < num_nodes; i++) {
        if(data.macro_mask[i].item<int>() == 1) {
            MacroInfo tmp;
            tmp.id = i;
            tmp.die = data.node_die[i].item<int>();
            tmp.counter = 0;
            tmp.w[0] = data.node_size_bot[i][0].item<float>();
            tmp.w[1] = data.node_size_top[i][0].item<float>();
            tmp.h[0] = data.node_size_bot[i][1].item<float>();
            tmp.h[1] = data.node_size_top[i][1].item<float>();
            if(tmp.die == 1) {
                tmp.x = data.node_pos[i][0].item<float>() - data.node_size_top[i][0].item<float>() / 2;
                tmp.y = data.node_pos[i][1].item<float>() - data.node_size_top[i][1].item<float>() / 2;
            } else {
                tmp.x = data.node_pos[i][0].item<float>() - data.node_size_bot[i][0].item<float>() / 2;
                tmp.y = data.node_pos[i][1].item<float>() - data.node_size_bot[i][1].item<float>() / 2;
            }
            area[0] += tmp.w[0] * tmp.h[0];
            area[1] += tmp.w[1] * tmp.h[1];
            m.emplace_back(tmp);
        }
    }

    std::sort(m.begin(), m.end(), [](const MacroInfo &a, const MacroInfo &b)
    { 
        if (a.x < b.x) return true;
        if (a.x == b.x && a.y < b.y) return true;
        return false;
    });

    std::cout << "init" << "\n";

    for(int iter = 0; iter < m.size() * 4; iter++) {
        if(iter % 1000 == 0) std::cout << iter << "\n";
        int cur_group = area[0] / max_mov_cell_areas[0].item<double>() > area[1]/ max_mov_cell_areas[1].item<double>()? 0: 1;
        cur_group = area[0] / max_mov_cell_areas[0].item<double>() > 1.0 ? 0: cur_group;
        cur_group = area[1] / max_mov_cell_areas[1].item<double>() > 1.0 ? 1: cur_group;
        int bestID = -1;
        float best_cost = 0;
        for(int i = 0; i < m.size(); i++) {
            float cost = 0;
            if (m[i].die != cur_group || m[i].counter > 4) {
                continue;
            }
            for(int j = 0; j < m.size(); j++) {
                if(i==j) continue;
                if(m[j].x > m[i].x + m[i].w[m[i].die]) break;
                if(m[i].die == m[j].die) {
                    cost += macro_overlap(m[i], m[j], false);
                }
                else {
                    cost -= macro_overlap(m[i], m[j], true);
                }
            }
            if(cost > best_cost) {
                best_cost = cost;
                bestID = i;
            }
        }
        if(bestID > 0){
            //std::cout << m[bestID].id << " "<< best_cost << "\n";
            area[m[bestID].die] -= m[bestID].w[m[bestID].die] * m[bestID].h[m[bestID].die];
            m[bestID].die = 1 - cur_group;
            area[m[bestID].die] += m[bestID].w[m[bestID].die] * m[bestID].h[m[bestID].die];
            m[bestID].counter++;
            
        } else {
            break;
        }
    }
    std::cout << area[0] << " "<< area[1] << " " << " " << "\n";
    for(int i = 0; i < m.size(); i++) {

        //std::cout << m[i].id << " "<< m[i].die << " " << " " << "\n";
        data.node_die[m[i].id] = m[i].die;
        nodes[m[i].id]->group = m[i].die;
    }
    m.clear();
    mov_cell_areas[0] = area[0];
    mov_cell_areas[1] = area[1];
    node_die = data.node_die.clone();

    logger.info("#Cells for each chip (%d, %d)", (1 - data.node_die).sum().item<int>(), data.node_die.sum().item<int>());
    logger.info("Areas for each chip (%ld, %ld)", (mov_cell_areas[0]).item<long>(), (mov_cell_areas[1]).item<long>());
    logger.info("Utils for each chip (%.2f, %.2f)", (mov_cell_areas[0] / max_mov_cell_areas[0]).item<double>(),
                (mov_cell_areas[1] / max_mov_cell_areas[1]).item<double>());

}

