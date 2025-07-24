#include "partition.h"

//-------------------------------------------------------------------------------

// void Partitioner::clear() {
//     for (int i = 0; i < BucketList.size(); i++) {
//         for (int j = -maxDegree; j <= maxDegree; j++) {
//             BucketList[i][j]->next = NULL;
//         }
//     }
//     maxGainIndex = {0, 0};
// }  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::addVertex(shared_ptr<ptNode> cell) {
    int gain = cell->gain;
    int group = cell->group;
    // gain index
    maxGainIndex[group] = max(gain, maxGainIndex[group]);
    // head vertex
    VertexPtr header = BucketList[group][gain];
    VertexPtr tmp = header->next;
    // create vertex
    VertexPtr vertex = make_shared<Vertex>();
    vertex->id = cell->id;
    vertex->gain = gain;
    // insert after head
    header->next = vertex;
    vertex->next = tmp;
    // pre
    vertex->pre = header;
    if (tmp != NULL) tmp->pre = vertex;
    // add to graph
    vertexList[cell->id] = vertex;
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::renewVertex(shared_ptr<ptNode> cell) {
    int gain = cell->gain;
    int group = cell->group;
    // gain index
    maxGainIndex[group] = max(gain, maxGainIndex[group]);

    // head vertex
    VertexPtr header = BucketList[group][gain];
    VertexPtr tmp = header->next;

    // maintain vertex
    VertexPtr vertex = vertexList[cell->id];
    vertex->id = cell->id;
    vertex->gain = gain;

    // insert after head
    header->next = vertex;
    vertex->next = tmp;

    // pre
    vertex->pre = header;
    if (tmp != NULL) tmp->pre = vertex;
}  // END MODULE

//-------------------------------------------------------------------------------

void Partitioner::rmVertex(int& cell, int& gain_index) {
    int group = nodes[cell]->group;
    via_gainlist[cell] = -std::numeric_limits<float>::max();
    VertexPtr vertex = vertexList[cell];

    // remove
    VertexPtr previous = vertex->pre;
    previous->next = vertex->next;
    if (vertex->next != NULL) vertex->next->pre = previous;

    // BucketList[group][gain_index]->next = vertex->next;
    // if (vertex->next != NULL) vertex->next->pre = BucketList[group][gain_index];

    // set null
    vertex->next = NULL;
    vertex->pre = NULL;

    // refresh max index
    int maxIndexOld = maxGainIndex[group];
    while ((BucketList[group][maxIndexOld]->next == NULL) && maxIndexOld != -maxDegree) {
        maxIndexOld--;
    }
    maxGainIndex[group] = maxIndexOld;
}

//-------------------------------------------------------------------------------

void Partitioner::update_vertex(int& cell, int& gain_index) {
    // int new_gain = gainlist[cell].item<int>();
    int new_gain = nodes[cell]->gain;
    int group = nodes[cell]->group;

    // new max index
    int maxIndex = maxGainIndex[group];
    maxGainIndex[group] = max(maxIndex, new_gain);

    VertexPtr vertex = vertexList[cell];
    // remove
    VertexPtr previous = vertex->pre;
    previous->next = vertex->next;
    if (vertex->next != NULL) vertex->next->pre = previous;

    // insert
    VertexPtr tmp = BucketList[group][new_gain]->next;
    BucketList[group][new_gain]->next = vertex;
    vertex->next = tmp;

    // previous
    vertex->pre = BucketList[group][new_gain];
    if (tmp != NULL) tmp->pre = vertex;

}  // END MODULE
