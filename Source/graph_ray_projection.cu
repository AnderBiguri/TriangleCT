
#include "graph_ray_projection.hpp"


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
__device__ __inline__ float moller_trumbore(const float3 ray1, const float3 ray2,
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
        const float3 ray1, const float3 ray2,
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

__device__ bool rayBoxIntersect(const float3 ray1, const float3 ray2,const float3 nodemin, const float3 nodemax){
    float3 direction;
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
template <int tree_depth>
__global__ void initXrays(const unsigned long* elements, const float* vertices,
        const unsigned long *boundary,const unsigned long nboundary,
        float * d_res, Geometry geo,
        const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,
        const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree)
{
    
    // Depth first R-tree search
    
    unsigned long  y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long  x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= geo.nDetecU) || (y>= geo.nDetecV))
        return;
    
    
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    
    
    
    long crossingID=-1;
//     int* path=(int*)malloc(tree_depth*sizeof(int));
//     int* n_checked=(int*)malloc(tree_depth*sizeof(int));
    int path[tree_depth];
    int n_checked[tree_depth];
    #pragma unroll
    for (int i=0;i<tree_depth;i++)
        n_checked[i]=0;
    
    float safetyEpsilon=0.0000001f;
    float t[4];
    float taux, tmin=1;
    int depth=0;
    // Compute detector position
    float3 det;
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    

    float3 nodemin,nodemax; 
    nodemax.x=(float)bin_box[root*6+0]; nodemax.y=(float)bin_box[root*6+1]; nodemax.z=(float)bin_box[root*6+2];
    nodemin.x=(float)bin_box[root*6+3]; nodemin.y=(float)bin_box[root*6+4]; nodemin.z=(float)bin_box[root*6+5];
    bool isinbox=rayBoxIntersect(source, det, nodemin,nodemax);
    if (!isinbox){
        
        d_res[idx]=-1.0f;
        return;
    }
   
    
    bool finished=false;
    int next_node;
    
    // we know it intersecst, lets start from teh first one.
   
    depth=0;   
    path[0]=root;
    n_checked[0]=0;
//     path[depth]=(int)bin_elements[root*(M+1)+0];
   
    crossingID=-1;
    int iter=0;
    
    
    while (~finished){
        iter++;
        // if the next one to check in the current node is the last one, then we have checked everything,
        // go up one node
        while((long)n_checked[depth]>=(long)bin_n_elements[path[depth]]){
            depth--;
        }
        if (depth<0){
            finished=true;
            d_res[idx]=(float)crossingID;
            return;
        }
        
        next_node=bin_elements[path[depth]*(M+1)+n_checked[depth]];
        
        //get bounding box
        nodemax.x=bin_box[next_node*6+0]; nodemax.y=bin_box[next_node*6+1]; nodemax.z=bin_box[next_node*6+2];
        nodemin.x=bin_box[next_node*6+3]; nodemin.y=bin_box[next_node*6+4]; nodemin.z=bin_box[next_node*6+5];
        isinbox=rayBoxIntersect(source, det, nodemin,nodemax);
        // count that we checked it already
        n_checked[depth]++;
        
        if (isinbox){
            if(!isleaf[next_node]){
                // if its not a leaf, then we just go deeper
                depth++;
                n_checked[depth]=0;//lets make sure prior values do not interfere now. If we go deeper, it means its the first time on this node.
                path[depth]=next_node;
            }else{
                

                // if its a leaf, we shoudl check the triangles
                for(unsigned int i=0;i<bin_n_elements[next_node];i++){
                    // check all triangles, obtain smallest t
                    tetraLineIntersect(elements,vertices,source,det,boundary[bin_elements[next_node*(M+1)+i]],t,true,safetyEpsilon);
                    // if there is an intersection
                    if ((t[0]+t[1]+t[2]+t[3])!=0){
                        taux=min4nz(t);
                        if (taux<tmin){
                            tmin=taux;
                            crossingID=bin_elements[next_node*(M+1)+i];
                        }
                    }
                }//endfor
            }
        }//end isinbox
        // If its not inside, then we just loop again and check the next one.
    }
}

template __global__ void initXrays<2>(const unsigned long* elements, const float* vertices,const unsigned long *boundary,const unsigned long nboundary,float * d_res, Geometry geo,const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree);
template __global__ void initXrays<4>(const unsigned long* elements, const float* vertices,const unsigned long *boundary,const unsigned long nboundary,float * d_res, Geometry geo,const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree);
template __global__ void initXrays<6>(const unsigned long* elements, const float* vertices,const unsigned long *boundary,const unsigned long nboundary,float * d_res, Geometry geo,const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree);
template __global__ void initXrays<8>(const unsigned long* elements, const float* vertices,const unsigned long *boundary,const unsigned long nboundary,float * d_res, Geometry geo,const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree);
template __global__ void initXrays<10>(const unsigned long* elements, const float* vertices,const unsigned long *boundary,const unsigned long nboundary,float * d_res, Geometry geo,const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree);
template __global__ void initXrays<12>(const unsigned long* elements, const float* vertices,const unsigned long *boundary,const unsigned long nboundary,float * d_res, Geometry geo,const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree);
template __global__ void initXrays<14>(const unsigned long* elements, const float* vertices,const unsigned long *boundary,const unsigned long nboundary,float * d_res, Geometry geo,const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree);


/**************************************************************************
 ******Fucntion to detect the first triangle to expand the graph***********
 *************************************************************************/
/////////////////////////////////////////////////////////////////////////// 
/////////////////////////         NOT USED       //////////////////////////
///////////////////////////////////////////////////////////////////////////
__global__ void initXraysBrute(const unsigned long* elements, const float* vertices,
        const unsigned long *boundary,const unsigned long nboundary,
        float * d_res, Geometry geo,
        const float3 source,const float3 deltaU,const float3 deltaV,const float3 uvOrigin,const float3 nodemin,const float3 nodemax)
{
    
    unsigned long  y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long  x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= geo.nDetecU) || (y>= geo.nDetecV))
        return;
    
    
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    
    
    // Compute detector position
    float3 det;
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    
    // Should we even try? if the ray does not cross the boundary, dont try
    bool crossBound=rayBoxIntersect(source, det, nodemin,nodemax);
    if (!crossBound){
        d_res[idx]=-1.0f;
        return;
    }
    
    
    // Check intersection with all elements in the boudnary
    unsigned long notintersect=nboundary;
    float t[4];
    float t1,tinter=10000.0f;
    float safetyEpsilon=0.0000001f;
    unsigned long crossingID=0;
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
    d_res[idx]=(float)crossingID;
    
    
}
/**************************************************************************
 ******************The mein projection fucntion ***************************
 *************************************************************************/

__global__ void graphProject(const unsigned long *elements, const float *vertices,const unsigned long *boundary,const long *neighbours, const float * d_image, float * d_res, Geometry geo,
        float3 source, float3 deltaU, float3 deltaV, float3 uvOrigin){
    
    unsigned long  y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long  x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= geo.nDetecU) || (y>= geo.nDetecV))
        return;
    
    
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    
    
    // Read initial position. Generate auxiliar variables for element tracking
    long current_element=(long)d_res[idx];
    long previous_element;
    long aux_element;
    
    
    //  Get the coordinates of the detector for this kernel
    float3 det;
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    
    

    // If the current element is "none", then we are done, we are not itnersecting the mesh
    if (current_element<0){
        //no need to do stuff
        d_res[idx]=0.0f;

        return;
    }
    
    // initialize variables for the lengths and resutl
    float result=0.0f;
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
    float3 direction,p1,p2;
    direction.x=det.x-source.x;     direction.y=det.y-source.y;     direction.z=det.z-source.z;
    p2.x=direction.x* (t2);  p2.y=direction.y* (t2); p2.z=direction.z* (t2);
    p1.x=direction.x* (t1);  p1.y=direction.y* (t1); p1.z=direction.z* (t1);
    
    length=sqrt((p2.x-p1.x)*(p2.x-p1.x)+(p2.y-p1.y)*(p2.y-p1.y)+(p2.z-p1.z)*(p2.z-p1.z));
    
    
    // Start accumulating the result
    result=d_image[boundary[current_element]]*length;
    
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
        d_res[idx]=result;
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
        // This was necesary in a previosu version, Its left here just in case its neeed again.
        
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
            result+=d_image[current_element]*length;
            
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
                d_res[idx]=result;
                return;
            }
            continue;
        }
        // If there was no intrsection, then we are out. Can this even happen?
        noNeighbours=true;
    }//endwhile
    
    // It should never get here, ever.
    d_res[idx]=-1.0;
    return;
}
/**************************************************************************
 *********************** Main fucntion ************************************
 *************************************************************************/
void graphForwardRay(float const * const  image,  Geometry geo,
        const double * angles,const unsigned int nangles,
        const float* nodes,const unsigned long nnodes,
        const unsigned long* elements,const unsigned long nelements,
        const long* neighbours,const unsigned long nneighbours,
        const unsigned long* boundary,const unsigned long nboundary,
        const int* bin_n_elements,const long* bin_elements,const double* bin_box,const int M,const int m,const double* MBR,const bool* isleaf,const long root,const long length_tree,const long tree_depth,
        float ** result)
{
    // Prepare for MultiGPU
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        mexErrMsgIdAndTxt("TriangleCT:graphForward:GPUselect","There are no available device(s) that support CUDA\n");
    }
    //
    // CODE assumes
    // 1.-All available devices are usable by this code
    // 2.-All available devices are equal, they are the same machine (warning trhown)
    unsigned int dev;
    char * devicenames;
    cudaDeviceProp deviceProp;
    
    for (dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev>0){
            if (strcmp(devicenames,deviceProp.name)!=0){
                mexWarnMsgIdAndTxt("TriangleCT:graphForward:GPUselect","Detected one (or more) different GPUs.\n This code is not smart enough to separate the memory GPU wise if they have different computational times or memory limits.\n First GPU parameters used. If the code errors you might need to change the way GPU selection is performed. \n graph_ray_projection.cu line 526.");
                break;
            }
        }
        devicenames=deviceProp.name;
    }
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    unsigned long long mem_GPU_global=(unsigned long long)(deviceProp.totalGlobalMem*0.9);
    
    // This is the mandatory mem that we need to broadcast to all GPUs
    size_t num_bytes_img  = nelements*sizeof(float);
    size_t num_bytes_nodes = nnodes*3*sizeof(float);
    size_t num_bytes_elements = nelements*4*sizeof(unsigned long);
    size_t num_bytes_neighbours = nneighbours*4*sizeof(long);
    size_t num_bytes_boundary = nboundary*sizeof(unsigned long);
    // R-tree
    size_t num_bytes_bin_n_elements = length_tree*sizeof(int);
    size_t num_bytes_bin_elements =  length_tree*(M+1)*sizeof(long);
    size_t num_bytes_bin_box = 6*length_tree*sizeof(double);
    size_t num_bytes_MBR = 6*nboundary*sizeof(double);
    size_t num_bytes_isleaf=length_tree*sizeof(bool);
    
    unsigned long long mem_needed_graph=num_bytes_img+num_bytes_nodes+num_bytes_elements+num_bytes_neighbours+num_bytes_boundary;
    unsigned long long mem_free_GPU=mem_GPU_global-mem_needed_graph;
    
//     mexPrintf(" num_bytes_img %llu \n", num_bytes_img );
//     mexPrintf("num_bytes_nodes  %llu \n", num_bytes_nodes );
//     mexPrintf("num_bytes_elements  %llu \n",  num_bytes_elements);
//     mexPrintf("num_bytes_neighbours  %llu \n", num_bytes_neighbours );
//     mexPrintf("num_bytes_boundary  %llu \n",  num_bytes_boundary);
//     mexPrintf("num_bytes_needed  %llu \n", mem_needed_graph);
//     mexPrintf("num_bytes_GPU %llu \n", mem_GPU_global );
    
    size_t num_bytes_proj = geo.nDetecU*geo.nDetecV * sizeof(float);
    if (mem_needed_graph>mem_GPU_global)
        mexErrMsgIdAndTxt("TriangleCT:graphForward:Memory","The entire mesh does not fit on the GPU \n");
    if (num_bytes_proj>mem_free_GPU)
        mexErrMsgIdAndTxt("TriangleCT:graphForward:Memory","The entire mesh + attenuation values + 2 projection do not fit on a GPU.\n Dividig the projections is not supported \n");
    
    
    
    
    float time;
    float timecopy=0, timekernel=0,timeaux;
    cudaEvent_t start, stop;
    if (DEBUG_TIME){
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }
    
    //result
    float ** d_res= (float **)malloc(deviceCount*sizeof(float*));
    // FE structured graph
    float** d_image=(float **)malloc(deviceCount*sizeof(float*));
    float** d_nodes=(float **)malloc(deviceCount*sizeof(float*));
    unsigned long** d_elements=(unsigned long **)malloc(deviceCount*sizeof(unsigned long*));
    long ** d_neighbours=( long **)malloc(deviceCount*sizeof(long*));
    unsigned long** d_boundary=(unsigned long **)malloc(deviceCount*sizeof(unsigned long*));
    // R-tree vars
    int**  d_bin_n_elements=(int **)malloc(deviceCount*sizeof(int*));
    long** d_bin_elements=  (long**)malloc(deviceCount*sizeof(long*));
    double** d_bin_box=(double**)malloc(deviceCount*sizeof(double*));
    double** d_MBR=(double**)malloc(deviceCount*sizeof(double*));
    bool** d_isleaf=(bool**)malloc(deviceCount*sizeof(bool*));
   
    
    //start allocation
    for (dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        
        // First send all the relevant data to CUDA, and allocate enough memory for the result
        
        gpuErrchk(cudaMalloc((void **)&d_res[dev],num_bytes_proj));
        
        
        gpuErrchk(cudaMalloc((void **)&d_image[dev],num_bytes_img));
        gpuErrchk(cudaMemcpyAsync(d_image[dev],image,num_bytes_img,cudaMemcpyHostToDevice));
        
        gpuErrchk(cudaMalloc((void **)&d_nodes[dev],num_bytes_nodes));
        gpuErrchk(cudaMemcpyAsync(d_nodes[dev],nodes,num_bytes_nodes,cudaMemcpyHostToDevice));
        
        gpuErrchk(cudaMalloc((void **)&d_elements[dev],num_bytes_elements));
        gpuErrchk(cudaMemcpyAsync(d_elements[dev],elements,num_bytes_elements,cudaMemcpyHostToDevice));
        
        gpuErrchk(cudaMalloc((void **)&d_neighbours[dev],num_bytes_neighbours));
        gpuErrchk(cudaMemcpyAsync(d_neighbours[dev],neighbours,num_bytes_neighbours,cudaMemcpyHostToDevice));
        
        gpuErrchk(cudaMalloc((void **)&d_boundary[dev],num_bytes_boundary));
        gpuErrchk(cudaMemcpyAsync(d_boundary[dev],boundary,num_bytes_boundary,cudaMemcpyHostToDevice));
        
        // Now all the R-tree stuff
        gpuErrchk(cudaMalloc((void **)&d_bin_n_elements[dev],num_bytes_bin_n_elements));
        gpuErrchk(cudaMemcpyAsync(d_bin_n_elements[dev],bin_n_elements,num_bytes_bin_n_elements,cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void **)&d_bin_elements[dev],num_bytes_bin_elements));
        gpuErrchk(cudaMemcpyAsync(d_bin_elements[dev],bin_elements,num_bytes_bin_elements,cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc((void **)&d_bin_box[dev],num_bytes_bin_box));
        gpuErrchk(cudaMemcpyAsync(d_bin_box[dev],bin_box,num_bytes_bin_box,cudaMemcpyHostToDevice));
        
        gpuErrchk(cudaMalloc((void **)&d_MBR[dev],num_bytes_MBR));
        gpuErrchk(cudaMemcpyAsync(d_MBR[dev],MBR,num_bytes_MBR,cudaMemcpyHostToDevice));
  
        gpuErrchk(cudaMalloc((void **)&d_isleaf[dev],num_bytes_isleaf));
        gpuErrchk(cudaMemcpyAsync(d_isleaf[dev],isleaf,num_bytes_isleaf,cudaMemcpyHostToDevice));

        
    }
    
    
    if (DEBUG_TIME){
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        mexPrintf("Time to memcpy:  %3.1f ms \n", time);
    }
    


    gpuErrchk(cudaDeviceSynchronize());

    
    // KERNEL TIME!
    int divU,divV;
    divU=8;
    divV=8;
    dim3 grid((geo.nDetecU+divU-1)/divU,(geo.nDetecV+divV-1)/divV,1);
    dim3 block(divU,divV,1);
    
    float3  deltaU, deltaV, uvOrigin;
    float3 source;
    
    
    for (unsigned int i=0;i<nangles;i+=(unsigned int)deviceCount){
        for (dev = 0; dev < deviceCount; dev++){
            geo.alpha=angles[(i+dev)*3];
            geo.theta=angles[(i+dev)*3+1];
            geo.psi  =angles[(i+dev)*3+2];
            
            //dev=i%deviceCount;
            //dev=0;
            
            
            computeGeometricParams(geo, &source,&deltaU, &deltaV,&uvOrigin,i+dev);
            
            //gpuErrchk(cudaDeviceSynchronize());
            
            cudaSetDevice(dev);
            if (DEBUG_TIME){
                
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
            }
            switch ((int)((tree_depth + 2 - 1) / 2) * 2){
                case 2:
                    initXrays<2><<<grid,block >>>(d_elements[dev],d_nodes[dev],d_boundary[dev],nboundary, d_res[dev], geo, source,deltaU, deltaV,uvOrigin,
                            d_bin_n_elements[dev],d_bin_elements[dev],d_bin_box[dev],M,m,d_MBR[dev],d_isleaf[dev],root,length_tree);
                    break;
                case 4:
                    initXrays<4><<<grid,block >>>(d_elements[dev],d_nodes[dev],d_boundary[dev],nboundary, d_res[dev], geo, source,deltaU, deltaV,uvOrigin,
                            d_bin_n_elements[dev],d_bin_elements[dev],d_bin_box[dev],M,m,d_MBR[dev],d_isleaf[dev],root,length_tree);
                    break;
                case 6:
                    initXrays<6><<<grid,block >>>(d_elements[dev],d_nodes[dev],d_boundary[dev],nboundary, d_res[dev], geo, source,deltaU, deltaV,uvOrigin,
                            d_bin_n_elements[dev],d_bin_elements[dev],d_bin_box[dev],M,m,d_MBR[dev],d_isleaf[dev],root,length_tree);
                    break;
                case 8:
                    initXrays<8><<<grid,block >>>(d_elements[dev],d_nodes[dev],d_boundary[dev],nboundary, d_res[dev], geo, source,deltaU, deltaV,uvOrigin,
                            d_bin_n_elements[dev],d_bin_elements[dev],d_bin_box[dev],M,m,d_MBR[dev],d_isleaf[dev],root,length_tree);
                    break;
                case 10:
                    initXrays<10><<<grid,block >>>(d_elements[dev],d_nodes[dev],d_boundary[dev],nboundary, d_res[dev], geo, source,deltaU, deltaV,uvOrigin,
                            d_bin_n_elements[dev],d_bin_elements[dev],d_bin_box[dev],M,m,d_MBR[dev],d_isleaf[dev],root,length_tree);
                    break;
                case 12:
                    initXrays<12><<<grid,block >>>(d_elements[dev],d_nodes[dev],d_boundary[dev],nboundary, d_res[dev], geo, source,deltaU, deltaV,uvOrigin,
                            d_bin_n_elements[dev],d_bin_elements[dev],d_bin_box[dev],M,m,d_MBR[dev],d_isleaf[dev],root,length_tree);
                    break;
                case 14:
                    initXrays<14><<<grid,block >>>(d_elements[dev],d_nodes[dev],d_boundary[dev],nboundary, d_res[dev], geo, source,deltaU, deltaV,uvOrigin,
                            d_bin_n_elements[dev],d_bin_elements[dev],d_bin_box[dev],M,m,d_MBR[dev],d_isleaf[dev],root,length_tree);
                    break;
                default:
                    mexErrMsgIdAndTxt("MEX:graph_ray_projections","R*-Tree is to deep (more than 14)");
                    break;
                    
            }
            if (DEBUG_TIME){
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                
                mexPrintf("Time to Init Kernel:  %3.1f ms \n", time);
            }
            if (DEBUG_TIME){
                
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
            }

            gpuErrchk(cudaDeviceSynchronize())
            graphProject<< <grid,block >> >(d_elements[dev],d_nodes[dev],d_boundary[dev],d_neighbours[dev],d_image[dev],d_res[dev], geo,source,deltaU,deltaV,uvOrigin);
            gpuErrchk(cudaDeviceSynchronize())
            if (DEBUG_TIME){
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                
                mexPrintf("Time to proj Kernel:  %3.1f ms \n", time);
            }
        }

        
        for (dev = 0; dev < deviceCount; dev++){
            //gpuErrchk(cudaDeviceSynchronize());
            cudaSetDevice(dev);
            gpuErrchk(cudaMemcpyAsync(result[i+dev], d_res[dev], num_bytes_proj, cudaMemcpyDeviceToHost));
        }
        
    }


    
    
//     if (DEBUG_TIME){
//         mexPrintf("Time of Kenrel:  %3.1f ms \n", timekernel);
//         mexPrintf("Time of memcpy to Host:  %3.1f ms \n", timecopy);
//     }
    
    
    if (DEBUG_TIME){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }
//     cudaGraphFree(&tempHostGraph,&tempHostElement,&tempHostNode);
    for (dev = 0; dev < deviceCount; dev++) {
        cudaSetDevice(dev);
        cudaFree(d_res[dev]);
        cudaFree(d_image[dev]);
        cudaFree(d_nodes[dev]);
        cudaFree(d_neighbours[dev]);
        cudaFree(d_elements[dev]);
        cudaFree(d_boundary[dev]);
        //R tree stuff
        cudaFree(d_bin_n_elements[dev]);
        cudaFree(d_bin_elements[dev]);
        cudaFree(d_bin_box[dev]);
        cudaFree(d_MBR[dev]);
        cudaFree(d_isleaf[dev]);
    }
    if (DEBUG_TIME){
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        mexPrintf("Time to free:  %3.1f ms \n", time);
    }
    
    cudaDeviceReset();
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
void computeGeometricParams(const Geometry geo,float3 * source, float3* deltaU, float3* deltaV, float3* originUV,unsigned int idxAngle){
    
    float3 auxOriginUV;
    float3 auxDeltaU;
    float3 auxDeltaV;
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
    
    float3 auxSource;
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

void eulerZYZ(Geometry geo,  float3* point){
    float3 auxPoint;
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

 