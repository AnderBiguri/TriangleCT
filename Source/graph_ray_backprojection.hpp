


#ifndef GRAPH_RAY_BACKPROJECTION
#define GRAPH_RAY_BACKPROJECTION

#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "graph.hpp"
#include "types_TIGRE.hpp"
#include "mex.h"


void graphBackwardRay(float const * const  projections,  Geometry geo,
                    const double * angles,const unsigned int nangles,
                    const float* nodes,const unsigned long nnodes,
                    const unsigned long* elements,const unsigned long nelements,
                    const long* neighbours,const unsigned long nneighbours,
                    const unsigned long* boundary,const unsigned long nboundary,
                    const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree, const long tree_depth,
                    float * result);
// Memory related
void reduceNodes(float *d_nodes, unsigned long nnodes, float* max,float* min);

void computeGeomtricParams(const Geometry geo,float3 * source, float3* deltaU, float3* deltaV, float3* originUV,unsigned int idxAngle);
void eulerZYZ(Geometry geo,  float3* point);
#endif