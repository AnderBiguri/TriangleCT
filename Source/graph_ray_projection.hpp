


#ifndef GRAPH_RAY_PROJECTION
#define GRAPH_RAY_PROJECTION

#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "graph.hpp"
#include "types_TIGRE.hpp"
#include "mex.h"


void graphForwardRay(float const * const  image,  Geometry geo, 
                         const double * angles,const unsigned int nangles, 
                         const float* nodes,const unsigned long nnodes,
                         const unsigned long* elements,const unsigned long nelements,
                         const long* neighbours,const unsigned long nneighbours,
                         const unsigned long* boundary,const unsigned long nboundary,
                         float ** result);

void graphForwardRay_CPU(float const * const  image,  Geometry geo, 
                         const double * angles,const unsigned int nangles, 
                         const double* nodes,const unsigned long nnodes,
                         const unsigned long* elements,const unsigned long nelements,
                         const long* neighbours,const unsigned long nneighbours,
                         const unsigned long* boundary,const unsigned long nboundary,
                         float ** result);
void bruteForwardRay(float const * const  image,  Geometry geo, const double * angles,const unsigned int nangles, Graph graph, float ** result);


void reduceNodes(float *d_nodes, unsigned long nnodes, float* max,float* min);
// Memory related
void cudaGraphMalloc(const Graph* inGraph, Graph **outGraph, Graph** outGraphHost, Element ** outElementHost, Node** outNodeHost);
void cudaGraphFree(Graph** tempHostGraph, Element** tempHostElement, Node** tempHostNode );

void computeGeometricParams(const Geometry geo,float3 * source, float3* deltaU, float3* deltaV, float3* originUV,unsigned int idxAngle);
void eulerZYZ(Geometry geo,  float3* point);
#endif