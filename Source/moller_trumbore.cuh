#include "types_TIGRE.hpp"

#ifndef MOLLER_TRUMBORE
#define MOLLER_TRUMBORE
 __device__  float moller_trumbore(const vec3 ray1, const vec3 ray2,
        const vec3 trip1,const vec3 trip2,const vec3 trip3,const float safety);
__device__ __inline__ vec3 cross(const vec3 a,const vec3 b);
__device__ __inline__ float dot(const vec3 a, const vec3 b);
#endif