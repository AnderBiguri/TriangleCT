


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
        const double* nodes,const unsigned long nnodes,
        const unsigned long* elements,const unsigned long nelements,
        const long* neighbours,const unsigned long nneighbours,
        const unsigned long* boundary,const unsigned long nboundary,
        float * result);
// Memory related
void computeGeomtricParams(const Geometry geo,vec3d * source, vec3d* deltaU, vec3d* deltaV, vec3d* originUV,unsigned int idxAngle);
void eulerZYZ(Geometry geo,  vec3d* point);
#endif