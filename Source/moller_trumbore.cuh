#include "types_TIGRE.hpp"

#ifndef MOLLER_TRUMBORE
#define MOLLER_TRUMBORE
 __device__  float moller_trumbore(const vec3 ray1, const vec3 ray2,
        const vec3 trip1,const vec3 trip2,const vec3 trip3);
#endif