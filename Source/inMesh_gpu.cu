


#include "inMesh_gpu.hpp"

#define DEBUG_TIME 1
#define MAXTHREADS 1024
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        mexPrintf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort){
            cudaDeviceReset();
            mexErrMsgIdAndTxt("MEX:InMesh", ".");
        }
    }
}


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
    
    float epsilon=0.00001; //DEFINE?
    
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
    float v= f*dot(direction,r);
    if (v<-safety || (u+v)>(1.0+safety)){
        // the intersection is outside of the triangle
        return -1;
    }
    return f*dot(e2,r);   
}




__device__ __inline__ float maxf_cuda(float a,float b){
    return (a>b)?a:b;
}
__device__ void warpReduce(volatile float *sdata,unsigned int tid) {
        sdata[tid] = maxf_cuda(sdata[tid + 32],sdata[tid]);
        sdata[tid] = maxf_cuda(sdata[tid + 16],sdata[tid]);
        sdata[tid] = maxf_cuda(sdata[tid + 8],sdata[tid]);
        sdata[tid] = maxf_cuda(sdata[tid + 4],sdata[tid]);
        sdata[tid] = maxf_cuda(sdata[tid + 2],sdata[tid]);
        sdata[tid] = maxf_cuda(sdata[tid + 1],sdata[tid]);

}
__global__ void  reduce_z(float *g_idata, float *g_odata, unsigned int n){
    extern __shared__ volatile float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + tid;
    unsigned int gridSize = blockDim.x*gridDim.x;
    float myMax = 0;
    while (i < n) {
        myMax = maxf_cuda(myMax,g_idata[i*3+2]);
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
// CC<3.0
    if (tid <  32){
        warpReduce(sdata,tid);
        myMax = sdata[0];
    }
    // Only for CC>3.0
//     if ( tid < 32 )
//     {
//         myMax = maxf_cuda(sdata[tid],sdata[tid + 32]);
//         
//         for (int offset = warpSize/2; offset > 0; offset /= 2) {
//             myMax = maxf_cuda( __shfl_down_sync(0xFFFFFFFF,myMax, offset,32) ,myMax);
//         }
//     }
    
    if (tid == 0) g_odata[blockIdx.x] = myMax;
}

__global__ void inMesh(unsigned long const* faces,  unsigned long const nfaces,
        float const*      vertices,  unsigned long const nvertices,
        float const*        points,  unsigned long const npoints,
        float maxZ,
        char* isIn, float* debug){
    unsigned long  idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>npoints)
        return;
    
    // define ray.
    // Choose random direction (Z)
    vec3 ray1,ray2;
    ray1.x=points[idx*3+0];
    ray1.y=points[idx*3+1];
    ray1.z=points[idx*3+2];
    
    ray2.x=ray1.x;
    ray2.y=ray1.y;
    ray2.z=maxZ+1;
    // Iterate over all faces
    char intersection=0;
    vec3 p1,p2,p3;
    
    for(unsigned long i=0;i<nfaces;i++){
//        for(unsigned long i=619;i<620;i++){
        // extraxt points
        p1.x=vertices[faces[i*3+0]*3+0];
        p1.y=vertices[faces[i*3+0]*3+1];
        p1.z=vertices[faces[i*3+0]*3+2];
        
        p2.x=vertices[faces[i*3+1]*3+0];
        p2.y=vertices[faces[i*3+1]*3+1];
        p2.z=vertices[faces[i*3+1]*3+2];
        
        p3.x=vertices[faces[i*3+2]*3+0];
        p3.y=vertices[faces[i*3+2]*3+1];
        p3.z=vertices[faces[i*3+2]*3+2];
//         discard the ones that 100% are not being crossed (is this slower on GPU???)
        if((p1.z<ray1.z && p2.z<ray1.z && p3.z <ray1.z)|| 
           (p1.x<ray1.x && p2.x<ray1.x && p3.x <ray1.x)||
           (p1.x>ray2.x && p2.x>ray2.x && p3.x >ray2.x)||
           (p1.y<ray1.y && p2.y<ray1.y && p3.y <ray1.y)||
           (p1.y>ray2.y && p2.y>ray2.y && p3.y >ray2.y))
            continue;
//        
        // check intersection
        char aux=(char)(moller_trumbore(ray1,ray2,p1,p2,p3,0.00)>=0.000f);
       
        intersection+=aux;
//         if (idx==2&&i==631){
// //             intersection=i;
//            debug[0]=moller_trumbore(ray1,ray2,p1,p2,p3,0.0f);
// //              debug[0]=aux;
//             
//         }
//         if (aux&&idx==2){
//             debug[0]=i;
//             break;
//         }

        
    }
    isIn[idx]=(char)(intersection%2);
//     isIn[idx]=intersection;
    
}


void inMesh_gpu(unsigned long const* faces,  unsigned long const nfaces,
        float const*      vertices,  unsigned long const nvertices,
        float const*        points,  unsigned long const npoints,
        char* isIn)
{
    
    float time;
    float timecopy, timekernel;
    cudaEvent_t start, stop;     
   if (DEBUG_TIME){
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }
    // Copy inputs to device
    size_t num_bytes_faces    = nfaces   *3* sizeof(unsigned long);
    size_t num_bytes_vertices = nvertices*3* sizeof(float);
    size_t num_bytes_points   = npoints  *3* sizeof(float);
    
    unsigned long *d_faces;
    float         *d_vertices, *d_points;
    
    gpuErrchk(cudaMalloc((void **)&d_faces,num_bytes_faces));
    gpuErrchk(cudaMemcpy(d_faces,faces,num_bytes_faces,cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMalloc((void **)&d_vertices,num_bytes_vertices));
    gpuErrchk(cudaMemcpy(d_vertices,vertices,num_bytes_vertices,cudaMemcpyHostToDevice));
    
    gpuErrchk(cudaMalloc((void **)&d_points,num_bytes_points));
    gpuErrchk(cudaMemcpy(d_points,points,num_bytes_points,cudaMemcpyHostToDevice));
    
    // Allocate output in device
    size_t num_bytes_isIn  = npoints * sizeof(char);
    char *d_isIn;
    gpuErrchk(cudaMalloc((void **)&d_isIn,num_bytes_isIn));
    
    if (DEBUG_TIME){
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        mexPrintf("Time to memcpy:  %3.1f ms \n", time);
    }
    // Create grid sizes
    int divU;
    divU=MAXTHREADS;
    dim3 grid((nvertices+divU-1)/divU,1,1);
    dim3 block(divU,1,1);
    
    ////
    // Get maxZ
    ////
     if (DEBUG_TIME){
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }
    //auxiliary for reduction
    float* d_aux,*d_aux2;
    gpuErrchk(cudaMalloc((void **)&d_aux, sizeof(float)*(nvertices + MAXTHREADS - 1) / MAXTHREADS));
    gpuErrchk(cudaMalloc((void **)&d_aux2,sizeof(float)));

     //gpuErrchk(cudaMalloc((void **)&debugreduce,MAXTHREADS*sizeof(float)));
    float maxZ;
    reduce_z<<<grid, block, MAXTHREADS*sizeof(float)>>>(d_vertices, d_aux, nvertices);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    float* getZs;
    getZs=(float*)malloc(MAXTHREADS*sizeof(float));
    if (grid.x > 1) {
        // There shoudl be another reduce here, but, as in the reduce code we have we are accessing every 3 values (for z)
        // that means that we can not reuse it. The most efficient way of doing it is doing the final reduce (<1024) on cpu
        // therefore avoiding a deep copy. We coudl also rewrite the reduce twice, but its not worth it.
        gpuErrchk(cudaMemcpy(getZs, d_aux,grid.x*sizeof(float), cudaMemcpyDeviceToHost));
        maxZ=getZs[0];
        for (int i=1;i<grid.x;i++)
            maxZ=(getZs[i]>maxZ)?getZs[i]:maxZ;
            
    } else {
        gpuErrchk(cudaMemcpy(&maxZ, d_aux, sizeof(float), cudaMemcpyDeviceToHost));
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
     if (DEBUG_TIME){
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        mexPrintf("Time to reduce maxZ:  %3.1f ms \n", time);
    }
    gpuErrchk(cudaFree(d_aux));
    gpuErrchk(cudaFree(d_aux2));
    
    
    ////
    // Call kernel inMehs
    ////
    if (DEBUG_TIME){
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }
    divU=MAXTHREADS;
    dim3 grid2((npoints+divU-1)/divU,1,1);
    dim3 block2(divU,1,1);
    
    float* d_debug,*debug;
    debug=(float*)malloc(sizeof(float));
    cudaMalloc((void**)&d_debug,sizeof(float));
    inMesh<<<grid2, block2>>>(d_faces,nfaces,d_vertices,nvertices,d_points, npoints,maxZ,d_isIn,d_debug);
    gpuErrchk(cudaMemcpy(debug, d_debug, sizeof(float), cudaMemcpyDeviceToHost));
    
   // mexPrintf("%f\n",debug[0]);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    if (DEBUG_TIME){
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        mexPrintf("Time to inMesh:  %3.1f ms \n", time);
    }
    // Return result
    gpuErrchk(cudaMemcpy(isIn, d_isIn, num_bytes_isIn, cudaMemcpyDeviceToHost));
    
    // Free memory
    
    gpuErrchk(cudaFree(d_faces));
    gpuErrchk(cudaFree(d_vertices));
    gpuErrchk(cudaFree(d_points));
    gpuErrchk(cudaFree(d_isIn));

    return;
    
}
