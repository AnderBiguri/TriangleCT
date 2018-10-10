


#ifndef IN_MESH_GPU
#define IN_MESH_GPU

#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "mex.h"
#include "types_TIGRE.hpp"

void inMesh_gpu(unsigned long const* faces,  unsigned long const nfaces, 
                float const*      vertices,  unsigned long const nvertices,
                float const*        points,  unsigned long const npoints,
                char* isIn);
#endif