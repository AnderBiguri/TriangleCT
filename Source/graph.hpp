

#ifndef GRAPH
#define GRAPH

    
    
// A finite element, a tetrahedron
struct Element {
    unsigned int nNeighbour;
    unsigned int* neighbour;
    unsigned int* nodeID;
    float value; // TODO: we want this here or separated?
};

// A point, part of some elements
struct Node {
    unsigned int nAdjacent;
    unsigned int* adjacent_element;
    float* position;
    
};

struct Graph{
     unsigned int nElement;
     unsigned int nNode;
     Element * element;
     Node* node;
     unsigned int nBoundary;
     unsigned int* boundary;
};
#endif