
#include "graph_ray_backprojection.hpp"


// This flag activates timing of the code
#define DEBUG_TIME 0
#define MAXTHREADS 1024

#define EPSILON 0.000001

// Cuda error checking fucntion.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        mexPrintf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort){
            cudaDeviceReset();
            mexErrMsgIdAndTxt("MEX:graph_ray_projections", ".");
        }
    }
}
__device__ __inline__ float maxf_cuda(float a,float b){
    return (a>b)?a:b;
}
__device__ __inline__ float minf_cuda(float a,float b){
    return (a<b)?a:b;
}
__device__ void warpMaxReduce(volatile float *sdata,unsigned int tid) {
    sdata[tid] = maxf_cuda(sdata[tid + 32],sdata[tid]);
    sdata[tid] = maxf_cuda(sdata[tid + 16],sdata[tid]);
    sdata[tid] = maxf_cuda(sdata[tid + 8],sdata[tid]);
    sdata[tid] = maxf_cuda(sdata[tid + 4],sdata[tid]);
    sdata[tid] = maxf_cuda(sdata[tid + 2],sdata[tid]);
    sdata[tid] = maxf_cuda(sdata[tid + 1],sdata[tid]);
    
}
__device__ void warpMinReduce(volatile float *sdata,unsigned int tid) {
    sdata[tid] = minf_cuda(sdata[tid + 32],sdata[tid]);
    sdata[tid] = minf_cuda(sdata[tid + 16],sdata[tid]);
    sdata[tid] = minf_cuda(sdata[tid + 8],sdata[tid]);
    sdata[tid] = minf_cuda(sdata[tid + 4],sdata[tid]);
    sdata[tid] = minf_cuda(sdata[tid + 2],sdata[tid]);
    sdata[tid] = minf_cuda(sdata[tid + 1],sdata[tid]);
    
}
__global__ void  maxReduceOffset(float *g_idata, float *g_odata, unsigned long n,unsigned int offset){
    extern __shared__ volatile float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + tid;
    unsigned int gridSize = blockDim.x*gridDim.x;
    float myMax = 0;
    while (i < n) {
        myMax = maxf_cuda(myMax,g_idata[i*3+offset]);
        i += gridSize;
    }
    sdata[tid] = myMax;
    __syncthreads();
    
    if (tid < 512)
        sdata[tid] = maxf_cuda(sdata[tid],sdata[tid + 512]);
    __syncthreads();
    if (tid < 256)
        sdata[tid] = maxf_cuda(sdata[tid],sdata[tid + 256]);
    __syncthreads();
    
    if (tid < 128)
        sdata[tid] = maxf_cuda(sdata[tid],sdata[tid + 128]);
    __syncthreads();
    
    if (tid <  64)
        sdata[tid] = maxf_cuda(sdata[tid],sdata[tid + 64]);
    __syncthreads();
    if (tid <  32){
        warpMaxReduce(sdata,tid);
        myMax = sdata[0];
    }
    if (tid == 0) g_odata[blockIdx.x] = myMax;
}
__global__ void  minReduceOffset(float *g_idata, float *g_odata, unsigned long n,unsigned int offset){
    extern __shared__ volatile float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + tid;
    unsigned int gridSize = blockDim.x*gridDim.x;
    float myMin = 0;
    while (i < n) {
        myMin = minf_cuda(myMin,g_idata[i*3+offset]);
        i += gridSize;
    }
    sdata[tid] = myMin;
    __syncthreads();
    
    if (tid < 512)
        sdata[tid] = minf_cuda(sdata[tid],sdata[tid + 512]);
    __syncthreads();
    if (tid < 256)
        sdata[tid] = minf_cuda(sdata[tid],sdata[tid + 256]);
    __syncthreads();
    
    if (tid < 128)
        sdata[tid] = minf_cuda(sdata[tid],sdata[tid + 128]);
    __syncthreads();
    
    if (tid <  64)
        sdata[tid] = minf_cuda(sdata[tid],sdata[tid + 64]);
    __syncthreads();
    if (tid <  32){
        warpMinReduce(sdata,tid);
        myMin = sdata[0];
    }
    if (tid == 0) g_odata[blockIdx.x] = myMin;
}    
/**************************************************************************
 *********************** vector addition ****************************
 *************************************************************************/
__global__ void addVectorsInPlace(float* a,float* b,unsigned long n){
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
     for(; idx<n; idx+=gridDim.x*blockDim.x) {
            a[idx]+=b[idx];
            
     }
}
    
/**************************************************************************
 *********************** cross product in CUDA ****************************
 *************************************************************************/
__device__ __inline__ vec3d cross(const vec3d a,const vec3d b)
{
    vec3d c;
    c.x= a.y*b.z - a.z*b.y;
    c.y= a.z*b.x - a.x*b.z;
    c.z= a.x*b.y - a.y*b.x;
    return c;
}
/**************************************************************************
 *********************** Dot product in CUDA ******************************
 *************************************************************************/
__device__ __inline__ double dot(const vec3d a, const vec3d b)
{
    
    return a.x*b.x+a.y*b.y+a.z*b.z;
}


/**************************************************************************
 *********************** maximum value in a 4 valued array of floats*******
 *************************************************************************/
__device__ __inline__ float max4(float *t,int* indM){
    float max=0;
    *indM=-1;
    for(int i=0;i<4;i++){
        if (t[i]>max){
            max=t[i];
            *indM=i;
        }
    }
    return max;
}
/**************************************************************************
 ********* minimum nonzero value in a 4 valued array of float *************
 *************************************************************************/
__device__ __inline__ float min4nz(float *t){
    float min=1;
    for(int i=0;i<4;i++)
        min=(t[i]<min && t[i]!=0)?t[i]:min;
        return min;
}

/**************************************************************************
 ********* number of non zeroes in a 4 legth float array **** *************
 *************************************************************************/
__device__ __inline__ int nnz(float *t){
    int nz=0;
    for(int i=0;i<4;i++){
        if(t[i]>0){
            nz++;
        }
    }
    return nz;
    
}


/**************************************************************************
 *********************** Moller trumbore **********************************
 **************************************************************************/
__device__ __inline__ float moller_trumbore(const vec3 ray1, const vec3 ray2,
        const vec3d trip1,const vec3d trip2,const vec3d trip3, const float safetyEpsilon){
    
    
    
    
    vec3d direction,e1,e2;
    
    direction.x=ray2.x-ray1.x;     direction.y=ray2.y-ray1.y;     direction.z=ray2.z-ray1.z;
    e1.x       =trip2.x-trip1.x;   e1.y       =trip2.y-trip1.y;   e1.z       =trip2.z-trip1.z;
    e2.x       =trip3.x-trip1.x;   e2.y       =trip3.y-trip1.y;   e2.z       =trip3.z-trip1.z;
    
    
    vec3d q=cross(direction,e2);
    double a=dot(e1,q);
    if ((a>-EPSILON) & (a<EPSILON)){
        // the vector is parallel to the plane (the intersection is at infinity)
        return 0.0f;
    }
    
    double f=1/a;
    vec3d s;
    
    s.x=ray1.x-trip1.x;     s.y=ray1.y-trip1.y;     s.z=ray1.z-trip1.z;
    double u=f*dot(s,q);
    
    if (u<0.0-safetyEpsilon){
        // the intersection is outside of the triangle
        return 0.0f;
    }
    
    vec3d r=cross(s,e1);
    double v= f*dot(direction,r);
    
    if (v<0.0-safetyEpsilon || (u+v)>1.0+safetyEpsilon){
        // the intersection is outside of the triangle
        return 0.0;
    }
    return f*dot(e2,r);
    
    
    
}

/**************************************************************************
 ***************************Tetra-line intersection************************
 *************************************************************************/

__device__ __inline__ bool tetraLineIntersect(const unsigned long *elements,const float *vertices,
        const vec3 ray1, const vec3 ray2,
        const unsigned long elementId,float *t,bool computelenght,const float safetyEpsilon){
    
    unsigned long auxNodeId[4];
    auxNodeId[0]=elements[elementId*4+0];
    auxNodeId[1]=elements[elementId*4+1];
    auxNodeId[2]=elements[elementId*4+2];
    auxNodeId[3]=elements[elementId*4+3];
    
    
    vec3d triN1,triN2,triN3;
    
    float l1,l2,l3,l4;
    
    ///////////////////////////////////////////////////////////////////////
    // As modular arithmetic is bad on GPUs (flop-wise), I manually unroll the loop
    //for (int i=0;i<4;i++)
    ///////////////////////////////////////////////////////////////////////
    // Triangle
    triN1.x=vertices[auxNodeId[0]*3+0];    triN1.y=vertices[auxNodeId[0]*3+1];    triN1.z=vertices[auxNodeId[0]*3+2];
    triN2.x=vertices[auxNodeId[1]*3+0];    triN2.y=vertices[auxNodeId[1]*3+1];    triN2.z=vertices[auxNodeId[1]*3+2];
    triN3.x=vertices[auxNodeId[2]*3+0];    triN3.y=vertices[auxNodeId[2]*3+1];    triN3.z=vertices[auxNodeId[2]*3+2];
    //compute
    l1=moller_trumbore(ray1,ray2,triN1,triN2,triN3,safetyEpsilon);
    // Triangle
    triN1.x=vertices[auxNodeId[0]*3+0];    triN1.y=vertices[auxNodeId[0]*3+1];    triN1.z=vertices[auxNodeId[0]*3+2];
    triN2.x=vertices[auxNodeId[1]*3+0];    triN2.y=vertices[auxNodeId[1]*3+1];    triN2.z=vertices[auxNodeId[1]*3+2];
    triN3.x=vertices[auxNodeId[3]*3+0];    triN3.y=vertices[auxNodeId[3]*3+1];    triN3.z=vertices[auxNodeId[3]*3+2];
    //compute
    l2=moller_trumbore(ray1,ray2,triN1,triN2,triN3,safetyEpsilon);
    // Triangle
    triN1.x=vertices[auxNodeId[0]*3+0];    triN1.y=vertices[auxNodeId[0]*3+1];    triN1.z=vertices[auxNodeId[0]*3+2];
    triN2.x=vertices[auxNodeId[2]*3+0];    triN2.y=vertices[auxNodeId[2]*3+1];    triN2.z=vertices[auxNodeId[2]*3+2];
    triN3.x=vertices[auxNodeId[3]*3+0];    triN3.y=vertices[auxNodeId[3]*3+1];    triN3.z=vertices[auxNodeId[3]*3+2];
    //compute
    l3=moller_trumbore(ray1,ray2,triN1,triN2,triN3,safetyEpsilon);
    // Triangle
    triN1.x=vertices[auxNodeId[1]*3+0];    triN1.y=vertices[auxNodeId[1]*3+1];    triN1.z=vertices[auxNodeId[1]*3+2];
    triN2.x=vertices[auxNodeId[2]*3+0];    triN2.y=vertices[auxNodeId[2]*3+1];    triN2.z=vertices[auxNodeId[2]*3+2];
    triN3.x=vertices[auxNodeId[3]*3+0];    triN3.y=vertices[auxNodeId[3]*3+1];    triN3.z=vertices[auxNodeId[3]*3+2];
    //compute
    l4=moller_trumbore(ray1,ray2,triN1,triN2,triN3,safetyEpsilon);
    
    //dump
    
    //fuck branches, but what can I do ....
    if ((l1==0.0)&&(l2==0.0)&&(l3==0.0)&&(l4==0.0)){
        t[0]=0.0;t[1]=0.0;t[2]=0.0;t[3]=0.0;
        return false;
    }else{
        t[0]=l1;t[1]=l2;t[2]=l3;t[3]=l4;
        // find which one is the intersection
        return true;
    }
}

/**************************************************************************
 ***************************Intersection between line-box******************
 *************************************************************************/

__device__ bool rayBoxIntersect(const vec3 ray1, const vec3 ray2,const vec3 nodemin, const vec3 nodemax){
    vec3 direction;
    direction.x=ray2.x-ray1.x;
    direction.y=ray2.y-ray1.y;
    direction.z=ray2.z-ray1.z;
    
    float tmin,tymin,tzmin;
    float tmax,tymax,tzmax;
    if (direction.x >= 0){
        tmin = (nodemin.x - ray1.x) / direction.x;
        tmax = (nodemax.x - ray1.x) / direction.x;
        
    }else{
        tmin = (nodemax.x - ray1.x) / direction.x;
        tmax = (nodemin.x - ray1.x) / direction.x;
    }
    
    if (direction.y >= 0){
        tymin = (nodemin.y - ray1.y) / direction.y;
        tymax = (nodemax.y - ray1.y) / direction.y;
    }else{
        tymin = (nodemax.y - ray1.y) / direction.y;
        tymax = (nodemin.y - ray1.y) / direction.y;
    }
    
    if ( (tmin > tymax) || (tymin > tmax) ){
        return false;
    }
    
    if (tymin > tmin){
        tmin = tymin;
    }
    
    if (tymax < tmax){
        tmax = tymax;
    }
    
    if (direction.z >= 0){
        tzmin = (nodemin.z - ray1.z) / direction.z;
        tzmax = (nodemax.z - ray1.z) / direction.z;
    }else{
        tzmin = (nodemax.z - ray1.z) / direction.z;
        tzmax = (nodemin.z - ray1.z) / direction.z;
    }
    
    
    if ((tmin > tzmax) || (tzmin > tmax)){
        return false;
    }
    // If we wanted the ts as output
////
// if (tzmin > tmin){
//     tmin = tzmin;
// }
//
// if (tzmax < tmax){
//     tmax = tzmax;
// }
////
    return true;
}
/**************************************************************************
 ******Fucntion to detect the first triangle to expand the graph***********
 *************************************************************************/

__global__ void initXrays(const unsigned long* elements, const float* vertices,
        const unsigned long *boundary,const unsigned long nboundary,
        float * d_aux, Geometry geo,
        const vec3 source,const vec3 deltaU,const vec3 deltaV,const vec3 uvOrigin,const vec3 nodemin,const vec3 nodemax)
{
    
    
    unsigned long  y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long  x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= geo.nDetecU) || (y>= geo.nDetecV))
        return;
    
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    
    
    // Compute detector position
    vec3 det;
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    
    // Should we even try? if the ray does not cross the boundary, dont try
    bool crossBound=rayBoxIntersect(source, det, nodemin,nodemax);
    if (!crossBound){
        d_aux[idx]=-1.0f;
        return;
    }
    
    
    
    // Check intersection with all elements in the boudnary
    unsigned long notintersect=nboundary;
    float t[4];
    float t1,tinter=10000.0f;
    float safetyEpsilon=0.0000001f;
    long crossingID=-1;
    //Check with all elements, and keep the one that gives lowest parameter
    while(notintersect==nboundary){
        notintersect=0;
        
        for(unsigned long i=0 ;i<nboundary;i++){
            tetraLineIntersect(elements,vertices,source,det,boundary[i],t,true,safetyEpsilon);
            if (nnz(t)==0){
                notintersect++;
            }else{
                t1=min4nz(t);
                if (t1<tinter){
                    tinter=t1;
                    crossingID=i;
                }
            }
        }
        safetyEpsilon=safetyEpsilon*10;
    }
    
    d_aux[idx]=(float)crossingID;
    return;
    
}
/**************************************************************************
 ******************The mein projection fucntion ***************************
 *************************************************************************/

__global__ void graphBackproject(const unsigned long *elements, const float *vertices,const unsigned long *boundary,const long *neighbours, const float * d_proj, const float * d_auxInit, float * d_image, Geometry geo,
        vec3 source, vec3 deltaU, vec3 deltaV, vec3 uvOrigin){
    
    unsigned long  y = blockIdx.y * blockDim.y + threadIdx.y;
    //unsigned long  y = threadIdx.y * gridDim.y + blockIdx.y;
    
    unsigned long  x = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned long  x = threadIdx.x * gridDim.x + blockIdx.x;
    
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= geo.nDetecU) || (y>= geo.nDetecV))
        return;
    
    
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    
    
    // Read initial position. Generate auxiliar variables for element tracking
    long current_element=(long)d_auxInit[idx];
    long previous_element;
    long aux_element;
    
    // for speed. Minimize reads
    float pixel_value=d_proj[idx];
    //  Get the coordinates of the detector for this kernel
    vec3 det;
    
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    
    
    // If the current element is "none", then we are done, we are not itnersecting the mesh
    if (current_element==-1){
        //no need to do stuff
        return;
    }
    // initialize variables for the lengths and resutl
    float length,t1,t2;
    float t[4];
    int indM;
    bool isIntersect;
    
    
    // Lets compute the first intersection outside the main loop.
    // The structure of this loop has to be identical to the one in InitXrays() or
    // there is risk of not getting the same floating point value bit by bit.
    float safeEpsilon=0.00001f;
    isIntersect=tetraLineIntersect(elements,vertices,source,det,boundary[current_element],t,true,0.0f);
    while(!isIntersect){
        isIntersect=tetraLineIntersect(elements,vertices,source,det,boundary[current_element],t,true,safeEpsilon);
        if (nnz(t)<=1){
            isIntersect=false;
            safeEpsilon*=10;
        }
    }
    // Reset the safety variable
    safeEpsilon=0.00001f;
    
    // Find the maximum and minimum non-zero intersection parameters
    t2=max4(t,&indM);
    t1=min4nz(t);
    
    // Lets get the ray (direction) and the current intersection length.
    vec3 direction,p1,p2;
    direction.x=det.x-source.x;     direction.y=det.y-source.y;     direction.z=det.z-source.z;
    p2.x=direction.x* (t2);  p2.y=direction.y* (t2); p2.z=direction.z* (t2);
    p1.x=direction.x* (t1);  p1.y=direction.y* (t1); p1.z=direction.z* (t1);
    
    length=sqrt((p2.x-p1.x)*(p2.x-p1.x)+(p2.y-p1.y)*(p2.y-p1.y)+(p2.z-p1.z)*(p2.z-p1.z));
    
    
    // Start accumulating the result
    atomicAdd(&d_image[boundary[current_element]],length*pixel_value);
    // If t1 and t2 are the same, we need to make sure that the one we choose as
    // t2 (the one that will lead us to the next element) is the correct one.
    // Otherwise we will go out of the image, and the code will end.
    // This piece of code makes sure that is checked and swaps them otherwise.
    if(t1==t2){
        aux_element=neighbours[boundary[current_element]*4+indM];
        if(aux_element==-1){
            int auxind;
            for(int i=0;i<4;i++){
                if(indM!=i && t[i]==t1){
                    auxind=i;
                }
            }
            indM=auxind;
        }
    }
    
    // Grab the index of the next elements and save the current one for further checking
    previous_element=boundary[current_element];
    current_element=neighbours[boundary[current_element]*4+indM];
    // if its "none" then thats it, we are done.
    if (current_element==-1){
        return;
    }
    
    float sumt;
    unsigned long c=0;
    bool noNeighbours=false;
    while(!noNeighbours && c<5000){ // RANDOM safe distance, change to something sensible
        // c is a counter to avoid infinite loops
        c++;
        // Check intersections we now this one is intersected )because it shares a face with the previosu one that was intersected)
        isIntersect=tetraLineIntersect(elements,vertices,source,det,(unsigned int)current_element,t,true,0.0f);
        while(!isIntersect){
            // If intersection failed, then lets slightly increase the size of the triangle
            // (not really, we increase the bounds of acceptable intersection values)
            // We can do it without safety becasue we already know it must happen.
            isIntersect=tetraLineIntersect(elements,vertices,source,det,(unsigned int)current_element,t,true,safeEpsilon);
            if (nnz(t)<=1){
                isIntersect=false;
                safeEpsilon*=10;
            }
        }
        safeEpsilon=0.00001f;
        
        // Find the maximum and minimum non-zero intersection parameters
        t2=max4(t,&indM);
        
        t1=min4nz(t);
// if they are very similar just treat them as if they were the same
// NOTE This was necesary in a previosu version, Its left here just in case its neeed again.
//////
//         if (fabsf(t2-t1)<0.00000001){
//             t2=t1;
//             t[indM]=t1;
//         }
//////
        
        // Are they all zero?
        sumt=t[0]+t[1]+t[2]+t[3];
        if (sumt!=0.0){
            // compute intersection length and update result integral
            p2.x=direction.x* (t2);  p2.y=direction.y* (t2); p2.z=direction.z* (t2);
            p1.x=direction.x* (t1);  p1.y=direction.y* (t1); p1.z=direction.z* (t1);
            length=sqrt((p2.x-p1.x)*(p2.x-p1.x)+(p2.y-p1.y)*(p2.y-p1.y)+(p2.z-p1.z)*(p2.z-p1.z));
            atomicAdd(&d_image[current_element],length*pixel_value);
            
            // Now lets make sure we can find the next element correctly
            
            // If t1 and t2 are the same, we need to make sure that the one we choose as
            // t2 (the one that will lead us to the next element) is the correct one.
            // Otherwise we will go backwards and get trapped in an infinite loop
            // This piece of code makes sure this does not happen.
            if(t1==t2){
                
                aux_element=neighbours[current_element*4+indM];
                if(aux_element==previous_element){
                    int auxind;
                    for(int i=0;i<4;i++){
                        if(indM!=i && t[i]==t1){
                            auxind=i;
                        }
                    }
                    indM=auxind;
                }
            }
            // Update the elements
            previous_element=current_element;
            current_element=neighbours[current_element*4+indM];
            
            // if we are out then thats it, we are done.
            if (current_element==-1){
                
                return;
            }
            continue;
        }
        // If there was no intrsection, then we are out. Can this even happen?
        noNeighbours=true;
    }//endwhile
    
    // It should never get here, ever.
    return;
}
/**************************************************************************
 *********************** Main fucntion ************************************
 *************************************************************************/
void graphBackwardRay(float const * const  projections,  Geometry geo,
        const double * angles,const unsigned int nangles,
        const float* nodes,const unsigned long nnodes,
        const unsigned long* elements,const unsigned long nelements,
        const long* neighbours,const unsigned long nneighbours,
        const unsigned long* boundary,const unsigned long nboundary,
        float * result)
{
    
    
    // Prepare for MultiGPU
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    mexPrintf("Devices: %d",deviceCount);
    if (deviceCount == 0) {
        mexErrMsgIdAndTxt("TriangleCT:graphBackward:GPUselect","There are no available device(s) that support CUDA\n");
    }
    //
    // CODE assumes
    // 1.-All available devices are usable by this code
    // 2.-All available devices are equal, they are the same machine (warning trhown)
    int dev;
    char * devicenames;
    cudaDeviceProp deviceProp;
    
    for (dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev>0){
            if (strcmp(devicenames,deviceProp.name)!=0){
                mexWarnMsgIdAndTxt("TriangleCT:graphBackward:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed. \n graph_ray_projection.cu line 526.");
                break;
            }
        }
        devicenames=deviceProp.name;
    }
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned long long mem_GPU_global=(unsigned long long)(deviceProp.totalGlobalMem*0.9);
    // This is the mandatory mem that we need to broadcast to all GPUs
    size_t num_bytes_nodes = nnodes*3*sizeof(float);
    size_t num_bytes_elements = nelements*4*sizeof(unsigned long);
    size_t num_bytes_neighbours = nneighbours*4*sizeof(long);
    size_t num_bytes_boundary = nboundary*sizeof(unsigned long);
    
    unsigned long long mem_needed_graph=num_bytes_nodes+num_bytes_elements+num_bytes_neighbours+num_bytes_boundary;
    unsigned long long mem_free_GPU=mem_GPU_global-mem_needed_graph;
    
    size_t num_bytes_proj = geo.nDetecU*geo.nDetecV * sizeof(float);
    size_t num_bytes_img  = nelements*sizeof(float);
    
    // The kenrels need graph+image+projection+auxiliar projection
    if (mem_needed_graph>mem_GPU_global)
        mexErrMsgIdAndTxt("TriangleCT:graphForward:Memory","The entire mesh does not fit on the GPU \n");
    if ((num_bytes_proj*2+num_bytes_img)>mem_free_GPU)
        mexErrMsgIdAndTxt("TriangleCT:graphForward:Memory","The entire mesh + attenuation values + 2 projections do not fit on a GPU.\n Dividig the projections is not supported \n");
    

    
    
    
    
    
    
    
    float**         d_proj          =   (float **)          malloc(deviceCount*sizeof(float*));
    float**         d_auxInit       =   (float **)          malloc(deviceCount*sizeof(float*));
    float**         d_image         =   (float **)          malloc(deviceCount*sizeof(float*));
    float**         d_nodes         =   (float **)          malloc(deviceCount*sizeof(float*));
    unsigned long** d_elements      =   (unsigned long **)  malloc(deviceCount*sizeof(unsigned long*));
    long **         d_neighbours    =   ( long **)          malloc(deviceCount*sizeof(long*));
    unsigned long** d_boundary      =   (unsigned long **)  malloc(deviceCount*sizeof(unsigned long*));
    
    for (dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        
        // First send all the relevant data to CUDA, and allocate enough memory for the result
        
        gpuErrchk(cudaMalloc((void **)&d_proj[dev],num_bytes_proj));
        gpuErrchk(cudaMalloc((void **)&d_auxInit[dev],num_bytes_proj));
        
        
        gpuErrchk(cudaMalloc((void **)&d_image[dev],num_bytes_img));
        gpuErrchk(cudaMemset(d_image[dev],0,num_bytes_img));
        
        
        gpuErrchk(cudaMalloc((void **)&d_nodes[dev],num_bytes_nodes));
        gpuErrchk(cudaMemcpyAsync(d_nodes[dev],nodes,num_bytes_nodes,cudaMemcpyHostToDevice));
        
        
        gpuErrchk(cudaMalloc((void **)&d_elements[dev],num_bytes_elements));
        gpuErrchk(cudaMemcpyAsync(d_elements[dev],elements,num_bytes_elements,cudaMemcpyHostToDevice));
        
        
        gpuErrchk(cudaMalloc((void **)&d_neighbours[dev],num_bytes_neighbours));
        gpuErrchk(cudaMemcpyAsync(d_neighbours[dev],neighbours,num_bytes_neighbours,cudaMemcpyHostToDevice));
        
        
        gpuErrchk(cudaMalloc((void **)&d_boundary[dev],num_bytes_boundary));
        gpuErrchk(cudaMemcpyAsync(d_boundary[dev],boundary,num_bytes_boundary,cudaMemcpyHostToDevice));
        
    }
    
    
  
    // Replace by a reduction (?)
     // Replace by a reduction (?)
    vec3 nodemin, nodemax;
    float max[3],min[3];
    cudaSetDevice(0);
    reduceNodes(d_nodes[0], nnodes, max, min);
    nodemax.x=max[0];
    nodemax.y=max[1];
    nodemax.z=max[2];
    nodemin.x=min[0];
    nodemin.y=min[1];
    nodemin.z=min[2];
    
    // KERNEL TIME!
    int divU,divV;
    divU=8;
    divV=8;
    dim3 grid((geo.nDetecU+divU-1)/divU,(geo.nDetecV+divV-1)/divV,1);
    dim3 block(divU,divV,1);
    
    vec3  deltaU, deltaV, uvOrigin;
    vec3 source;
    for (unsigned int i=0;i<nangles;i+=(unsigned int)deviceCount){
        for (dev = 0; dev < deviceCount; dev++){
             cudaSetDevice(dev);
             gpuErrchk(cudaMemcpyAsync(d_proj[dev],&projections[geo.nDetecU*geo.nDetecV*(i+dev)],num_bytes_proj,cudaMemcpyHostToDevice));
        }
        cudaDeviceSynchronize(); 
        for (dev = 0; dev < deviceCount; dev++){
            geo.alpha=angles[(i+dev)*3];
            geo.theta=angles[(i+dev)*3+1];
            geo.psi  =angles[(i+dev)*3+2];
            computeGeomtricParams(geo, &source,&deltaU, &deltaV,&uvOrigin,i+dev);
         
            cudaSetDevice(dev);

            initXrays << <grid,block >> >(d_elements[dev],d_nodes[dev],d_boundary[dev],nboundary,d_auxInit[dev], geo, source,deltaU, deltaV,uvOrigin,nodemin,nodemax);
            graphBackproject<< <grid,block >> >(d_elements[dev],d_nodes[dev],d_boundary[dev],d_neighbours[dev],d_proj[dev],d_auxInit[dev],d_image[dev], geo,source,deltaU,deltaV,uvOrigin);
        }
    }
    cudaDeviceSynchronize();
    for (dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaFree(d_proj[dev]);
        cudaFree(d_auxInit[dev]);
        cudaFree(d_nodes[dev]);
        cudaFree(d_neighbours[dev]);
        cudaFree(d_elements[dev]);
        cudaFree(d_boundary[dev]);

    }
    
    
    if(deviceCount>1){
        mexPrintf("Vector adding triggered\n");
        cudaSetDevice(0);
        float* d_imageaux;
        gpuErrchk(cudaMalloc((void **)&d_imageaux,num_bytes_img));
        for (dev = 1; dev < deviceCount; dev++){
            
            gpuErrchk(cudaMemcpyPeer(d_imageaux,0,d_image[dev],dev,num_bytes_img)); 
            cudaDeviceSynchronize();
            addVectorsInPlace<<<60,1024>>>(d_image[0],d_imageaux,nelements);    
        
        }
        cudaFree(d_imageaux);
    }
    
    gpuErrchk(cudaMemcpy(result, d_image[0], num_bytes_img, cudaMemcpyDeviceToHost)); 
    
    
    for (dev = 0; dev < deviceCount; dev++) { 
        cudaSetDevice(dev);
        cudaFree(d_image[dev]);
    }

    return;
    
    
}
void reduceNodes(float *d_nodes, unsigned long nnodes, float* max, float* min){
    
    int divU;
    divU=MAXTHREADS;
    dim3 grid((nnodes+divU-1)/divU,1,1);
    dim3 block(divU,1,1);
    
    
    //auxiliary for reduction
    float* d_auxmax,*d_auxmin;
    gpuErrchk(cudaMalloc((void **)&d_auxmax, sizeof(float)*(nnodes + MAXTHREADS - 1) / MAXTHREADS));
    gpuErrchk(cudaMalloc((void **)&d_auxmin, sizeof(float)*(nnodes + MAXTHREADS - 1) / MAXTHREADS));

    //gpuErrchk(cudaMalloc((void **)&debugreduce,MAXTHREADS*sizeof(float)));
    
    float** getFinalreducesmin=(float**)malloc(3*sizeof(float*));
    float** getFinalreducesmax=(float**)malloc(3*sizeof(float*));

    for (unsigned int i=0; i<3; i++){
        getFinalreducesmin[i]=(float*)malloc(MAXTHREADS*sizeof(float));
        getFinalreducesmax[i]=(float*)malloc(MAXTHREADS*sizeof(float));
    }
    
    
    // for X,Y,Z
    for (unsigned int i=0; i<3; i++){
        maxReduceOffset<<<grid, block, MAXTHREADS*sizeof(float)>>>(d_nodes, d_auxmax, nnodes,i);
        minReduceOffset<<<grid, block, MAXTHREADS*sizeof(float)>>>(d_nodes, d_auxmin, nnodes,i);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        if (grid.x > 1) {
            // There shoudl be another reduce here, but, as in the reduce code we have we are accessing every 3 values
            // that means that we can not reuse it. The most efficient way of doing it is doing the final reduce (<1024) on cpu
            // therefore avoiding a deep copy. We coudl also rewrite the reduce twice, but its not worth my time now (D:).
            gpuErrchk(cudaMemcpy( getFinalreducesmin[i], d_auxmin,grid.x*sizeof(float), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy( getFinalreducesmax[i], d_auxmax,grid.x*sizeof(float), cudaMemcpyDeviceToHost));
            max[i]=getFinalreducesmax[i][0];
            max[i]=getFinalreducesmin[i][0];
            for (unsigned int j=1;j<grid.x;j++){
                max[i]=( getFinalreducesmax[i][j]>max[i])? getFinalreducesmax[i][j]:max[i];
                min[i]=( getFinalreducesmin[i][j]<min[i])? getFinalreducesmin[i][j]:min[i];
            }
        } else {
            gpuErrchk(cudaMemcpy(&max[i], d_auxmax, sizeof(float), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(&min[i], d_auxmin, sizeof(float), cudaMemcpyDeviceToHost));

        }
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    cudaFree(d_auxmax);
    cudaFree(d_auxmin);
}


// TODO: quite a lot of geometric transforms.
void computeGeomtricParams(const Geometry geo,vec3 * source, vec3* deltaU, vec3* deltaV, vec3* originUV,unsigned int idxAngle){
    
    vec3 auxOriginUV;
    vec3 auxDeltaU;
    vec3 auxDeltaV;
    auxOriginUV.x=-(geo.DSD[idxAngle]-geo.DSO[idxAngle]);
    // top left
    auxOriginUV.y=-geo.sDetecU/2+/*half a pixel*/geo.dDetecU/2;
    auxOriginUV.z=geo.sDetecV/2-/*half a pixel*/geo.dDetecV/2;
    
    //Offset of the detector
    auxOriginUV.y=auxOriginUV.y+geo.offDetecU[idxAngle];
    auxOriginUV.z=auxOriginUV.z+geo.offDetecV[idxAngle];
    
    // Change in U
    auxDeltaU.x=auxOriginUV.x;
    auxDeltaU.y=auxOriginUV.y+geo.dDetecU;
    auxDeltaU.z=auxOriginUV.z;
    //Change in V
    auxDeltaV.x=auxOriginUV.x;
    auxDeltaV.y=auxOriginUV.y;
    auxDeltaV.z=auxOriginUV.z-geo.dDetecV;
    
    vec3 auxSource;
    auxSource.x=geo.DSO[idxAngle];
    auxSource.y=0;
    auxSource.z=0;
    
    // rotate around axis.
    eulerZYZ(geo,&auxOriginUV);
    eulerZYZ(geo,&auxDeltaU);
    eulerZYZ(geo,&auxDeltaV);
    eulerZYZ(geo,&auxSource);
    
    // Offset image (instead of offseting image, -offset everything else)
    auxOriginUV.x  =auxOriginUV.x-geo.offOrigX[idxAngle];     auxOriginUV.y  =auxOriginUV.y-geo.offOrigY[idxAngle];     auxOriginUV.z  =auxOriginUV.z-geo.offOrigZ[idxAngle];
    auxDeltaU.x=auxDeltaU.x-geo.offOrigX[idxAngle];           auxDeltaU.y=auxDeltaU.y-geo.offOrigY[idxAngle];           auxDeltaU.z=auxDeltaU.z-geo.offOrigZ[idxAngle];
    auxDeltaV.x=auxDeltaV.x-geo.offOrigX[idxAngle];           auxDeltaV.y=auxDeltaV.y-geo.offOrigY[idxAngle];           auxDeltaV.z=auxDeltaV.z-geo.offOrigZ[idxAngle];
    auxSource.x=auxSource.x-geo.offOrigX[idxAngle];           auxSource.y=auxSource.y-geo.offOrigY[idxAngle];           auxSource.z=auxSource.z-geo.offOrigZ[idxAngle];
    
    auxDeltaU.x=auxDeltaU.x-auxOriginUV.x;  auxDeltaU.y=auxDeltaU.y-auxOriginUV.y; auxDeltaU.z=auxDeltaU.z-auxOriginUV.z;
    auxDeltaV.x=auxDeltaV.x-auxOriginUV.x;  auxDeltaV.y=auxDeltaV.y-auxOriginUV.y; auxDeltaV.z=auxDeltaV.z-auxOriginUV.z;
    
    *originUV=auxOriginUV;
    *deltaU=auxDeltaU;
    *deltaV=auxDeltaV;
    *source=auxSource;
    
    return;
}

void eulerZYZ(Geometry geo,  vec3* point){
    vec3 auxPoint;
    auxPoint.x=point->x;
    auxPoint.y=point->y;
    auxPoint.z=point->z;
    
    point->x=(+cos(geo.alpha)*cos(geo.theta)*cos(geo.psi)-sin(geo.alpha)*sin(geo.psi))*auxPoint.x+
            (-cos(geo.alpha)*cos(geo.theta)*sin(geo.psi)-sin(geo.alpha)*cos(geo.psi))*auxPoint.y+
            cos(geo.alpha)*sin(geo.theta)*auxPoint.z;
    
    point->y=(+sin(geo.alpha)*cos(geo.theta)*cos(geo.psi)+cos(geo.alpha)*sin(geo.psi))*auxPoint.x+
            (-sin(geo.alpha)*cos(geo.theta)*sin(geo.psi)+cos(geo.alpha)*cos(geo.psi))*auxPoint.y+
            sin(geo.alpha)*sin(geo.theta)*auxPoint.z;
    
    point->z=-sin(geo.theta)*cos(geo.psi)*auxPoint.x+
            sin(geo.theta)*sin(geo.psi)*auxPoint.y+
            cos(geo.theta)*auxPoint.z;
    
    
    
    
}