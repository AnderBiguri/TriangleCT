
#include "moller_trumbore.cuh"
__device__ __inline__ vec3 cross(const vec3 a,const vec3 b)
{
    vec3 c = {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
    return c;
}
__device__ __inline__ float dot(const vec3 a, const vec3 b)
{
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
__device__ __inline__ float moller_trumbore(const vec3 ray1, const vec3 ray2,
        const vec3 trip1,const vec3 trip2,const vec3 trip3,const float safety){
    
    float epsilon=0.000001; //DEFINE?
    
    vec3 direction,e1,e2;
    direction.x=ray2.x-ray1.x;     direction.y=ray2.y-ray1.y;     direction.z=ray2.z-ray1.z;
    e1.x       =trip2.x-trip1.x;   e1.y       =trip2.y-trip1.y;   e1.z       =trip2.z-trip1.z;
    e2.x       =trip3.x-trip1.x;   e2.y       =trip3.y-trip1.y;   e2.z       =trip3.z-trip1.z;
    
    
    vec3 q=cross(direction,e2);
    float a=dot(e1,q);
    
    if (a>-epsilon-safety  && a<epsilon+safety){
        // the vector is parallel to the plane (the intersection is at infinity)
        return -1;
    }
    
    float f=1/a;
    vec3 s;
    s.x=ray1.x-trip1.x;     s.y=ray1.y-trip1.y;     s.z=ray1.z-trip1.z;
    float u=f*dot(s,q);
    
    if (u<-safety){
        // the intersection is outside of the triangle
        return -1;
    }
    
    vec3 r=cross(s,e1);
    float v= f*dot(s,q);
    
    if (v<-safety || u+v>1.0+safety){
        // the intersection is outside of the triangle
        return -1;
    }
    return f*dot(e2,r);   
}




