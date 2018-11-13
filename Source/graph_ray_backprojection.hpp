


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
                    float * result);
// Memory related
void reduceNodes(float *d_nodes, unsigned long nnodes, float* max,float* min);

void computeGeomtricParams(const Geometry geo,vec3 * source, vec3* deltaU, vec3* deltaV, vec3* originUV,unsigned int idxAngle);
void eulerZYZ(Geometry geo,  vec3* point);
#endif