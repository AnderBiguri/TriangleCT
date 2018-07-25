
#include "graph_ray_projection.hpp"


// This flag activates timing of the code
#define DEBUG_TIME 1


// Cuda error checking.
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

__global__ void testKernel(const Graph * graph,float * d_res){
    d_res[0]=(float)graph->node[1].position[0];
    
};
/*********************************************************************
 *********************** Cross product in CUDA ************************
 ********************************************************************/
__device__ __inline__ vec3 cross(const vec3 a,const vec3 b)
{
    vec3 c;
    c.x=a.y*b.z - a.z*b.y;
    c.y= a.z*b.x - a.x*b.z;
    c.z= a.x*b.y - a.y*b.x;
    return c;
}
/*********************************************************************
 *********************** Dot product in CUDA ************************
 ********************************************************************/
__device__ __inline__ float dot(const vec3 a, const vec3 b)
{
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

/*********************************************************************
 *********************** Moller trumbore ************************
 ********************************************************************/
__device__ __inline__ float moller_trumbore(const vec3 ray1, const vec3 ray2,
        const vec3 trip1,const vec3 trip2,const vec3 trip3){
    float epsilon=0.000001; //DEFINE?
    
    vec3 direction,e1,e2;
    
    direction.x=ray2.x-ray1.x;     direction.y=ray2.y-ray1.y;     direction.z=ray2.z-ray1.z;
    e1.x       =trip2.x-trip1.x;   e1.y       =trip2.y-trip1.y;   e1.z       =trip2.z-trip1.z;
    e2.x       =trip3.x-trip1.x;   e2.y       =trip3.y-trip1.y;   e2.z       =trip3.z-trip1.z;
    
    
    vec3 q=cross(direction,e2);
    float a=dot(e1,q);
    
    if (a>-epsilon & a<epsilon){
        // the vector is parallel to the plane (the intersection is at infinity)
        return -1;
    }
    
    float f=1/a;
    vec3 s;
    s.x=ray1.x-trip1.x;     s.y=ray1.y-trip1.y;     s.z=ray1.z-trip1.z;
    float u=f*dot(s,q);
    
    if (u<0.0){
        // the intersection is outside of the triangle
        return -1;
    }
    
    vec3 r=cross(s,e1);
    float v= f*dot(direction,r);
    
    if (v<0.0 || (u+v)>1.0){
        // the intersection is outside of the triangle
        return -1;
    }
    return f*dot(e2,r);
    
    
    
}

/*********************************************************************
 **********************Tetra-line intersection************************
 ********************************************************************/

// TODO: check if adding if-clauses after each moller trumbore is better of worse.
template<bool computelenght>
        __device__ __inline__ bool tetraLineIntersect(const Graph * graph,
        const vec3 ray1, const vec3 ray2,
        const unsigned int elementId, float * length){
    
    unsigned int* auxNodeId=graph->element[elementId].nodeID;
    vec3 triN1,triN2,triN3;
    
    //Maybe make graph vec3 ? TODO measure time
    float* auxpoint;
    float l1,l2,l3,l4;
    ///////////////////////////////////////////////////////////////////////
    // As modular arithmetic is bad on GPUs (flop-wise), I manually unroll the loop
    //for (int i=0;i<4;i++)
    ///////////////////////////////////////////////////////////////////////
    // Triangle
    auxpoint=graph->node[auxNodeId[0]].position;
    triN1.x=auxpoint[0];triN1.y=auxpoint[1];triN1.z=auxpoint[2];
    auxpoint=graph->node[auxNodeId[1]].position;
    triN2.x=auxpoint[0];triN2.y=auxpoint[1];triN2.z=auxpoint[2];
    auxpoint=graph->node[auxNodeId[2]].position;
    triN3.x=auxpoint[0];triN3.y=auxpoint[1];triN3.z=auxpoint[2];
    //compute
    l1=moller_trumbore(ray1,ray2,triN1,triN2,triN3);
    // Triangle
    auxpoint=graph->node[auxNodeId[1]].position;
    triN1.x=auxpoint[0];triN1.y=auxpoint[1];triN1.z=auxpoint[2];
    auxpoint=graph->node[auxNodeId[2]].position;
    triN2.x=auxpoint[0];triN2.y=auxpoint[1];triN2.z=auxpoint[2];
    auxpoint=graph->node[auxNodeId[3]].position;
    triN3.x=auxpoint[0];triN3.y=auxpoint[1];triN3.z=auxpoint[2];
    //compute
    l2=moller_trumbore(ray1,ray2,triN1,triN2,triN3);
    // Triangle
    auxpoint=graph->node[auxNodeId[2]].position;
    triN1.x=auxpoint[0];triN1.y=auxpoint[1];triN1.z=auxpoint[2];
    auxpoint=graph->node[auxNodeId[3]].position;
    triN2.x=auxpoint[0];triN2.y=auxpoint[1];triN2.z=auxpoint[2];
    auxpoint=graph->node[auxNodeId[0]].position;
    triN3.x=auxpoint[0];triN3.y=auxpoint[1];triN3.z=auxpoint[2];
    //compute
    l3=moller_trumbore(ray1,ray2,triN1,triN2,triN3);
    // Triangle
    auxpoint=graph->node[auxNodeId[3]].position;
    triN1.x=auxpoint[0];triN1.y=auxpoint[1];triN1.z=auxpoint[2];
    auxpoint=graph->node[auxNodeId[0]].position;
    triN2.x=auxpoint[0];triN2.y=auxpoint[1];triN2.z=auxpoint[2];
    auxpoint=graph->node[auxNodeId[1]].position;
    triN3.x=auxpoint[0];triN3.y=auxpoint[1];triN3.z=auxpoint[2];
    //compute
    l4=moller_trumbore(ray1,ray2,triN1,triN2,triN3);
    
    if(!computelenght){
        *length=(l1>0)+(l2>0)*2+(l3>0)*4+(l4>0)*8;
        return (l1!=-1)||(l2!=-1)||(l3!=-1)||(l4!=-1);
    }else{
        
    }
}

/*********************************************************************
 ******Fucntion to detect the first triangle to expand the graph******
 ********************************************************************/

__global__ void initXrays(const Graph * graph,float * d_res, Geometry geo,
        vec3 source, vec3 deltaU, vec3 deltaV, vec3 uvOrigin){
    
    unsigned long  y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long  x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= geo.nDetecU) || (y>= geo.nDetecV))
        return;
    
    // Create ray
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    vec3 det;
    
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    
    // Check intersection with boundary
    float l=-1;
    for(unsigned int i=0 ;i<graph->nBoundary;i++){
        //obtain triangle
        unsigned int* auxNodeId=graph->element[graph->boundary[i]].nodeID;
        vec3 triN1,triN2,triN3;
        
        //Maybe make graph vec3 ? TODO measure time
        float* auxpoint;
        float l1,l2,l3,l4;

        if (tetraLineIntersect<false>(graph,source,det,graph->boundary[i],&l))
        {
            //return id of intersected boundary node per ray
            d_res[idx]=(float)graph->boundary[i];
            return;
        }
        
        
        
    }
    
    d_res[idx]=-1.0;
    
    // Should I put the kernels together, or separate? TODO
    
}

__global__ void graphProject(const Graph * graph,float * d_res, Geometry geo,
        vec3 source, vec3 deltaU, vec3 deltaV, vec3 uvOrigin){
        unsigned long  y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned long  x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= geo.nDetecU) || (y>= geo.nDetecV))
        return;
    
    // Read initial position. 
    // Is having this here going to speed up because the bolew maths can be done while waiting to read?
    int init_element=(int)d_res[idx];
    
    // Create ray
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    vec3 det;
    
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    
    if (init_element<0){
        //no need to do stuff
        d_res[idx]=0.0f;
        return;
    }
    // Check intersection with boundary

}
/*********************************************************************
 *********************** Main fucntion ************************
 ********************************************************************/
void graphForwardRay(float const * const  image,  Geometry geo, const double * angles,const unsigned int nangles, Graph graph, float ** result){
    
    float time;
    float timecopy, timekernel;
    cudaEvent_t start, stop;
    
    size_t num_bytes = geo.nDetecU*geo.nDetecV * sizeof(float);
    
    float * d_res;
    gpuErrchk(cudaMalloc((void **)&d_res,num_bytes));
    
    
    
    
    if (DEBUG_TIME){
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }
    Graph* cudaGraph;
    
    // these three only exist to be able to Free the memory, they refer to the same device arrays than cudaGraph, we just can not directly derreference them
    Graph* tempHostGraph;
    Node * tempHostNode;
    Element* tempHostElement;
    cudaGraphMalloc(&graph, &cudaGraph, &tempHostGraph,&tempHostElement,&tempHostNode);
    
    if (DEBUG_TIME){
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        mexPrintf("Time to memcpy Graph:  %3.1f ms \n", time);
    }
    
    
    // KERNEL TIME!
    int divU,divV;
    divU=8;
    divV=8;
    dim3 grid((geo.nDetecU+divU-1)/divU,(geo.nDetecV+divV-1)/divV,1);
    dim3 block(divU,divV,1);
    
    vec3 source, deltaU, deltaV, uvOrigin;
    
    for (unsigned int i=0;i<nangles;i++){
        geo.alpha=angles[i*3];
        geo.theta=angles[i*3+1];
        geo.psi  =angles[i*3+2];
        
        computeGeomtricParams(geo, &source,&deltaU, &deltaV,&uvOrigin,i);
        if (DEBUG_TIME){
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }
        initXrays << <grid,block >> >(cudaGraph, d_res, geo, source,deltaU, deltaV,uvOrigin);
//         testKernel<<<1,1>>>(cudaGraph,d_res);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        if (DEBUG_TIME){
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&timekernel, start, stop);
            
            
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }
        
        gpuErrchk(cudaMemcpy(result[i], d_res, num_bytes, cudaMemcpyDeviceToHost));
        
        if (DEBUG_TIME){
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&timecopy, start, stop);
        }
    }
    
    
    if (DEBUG_TIME){
        mexPrintf("Time of Kenrel:  %3.1f ms \n", timekernel*nangles);
        mexPrintf("Time of memcpy to Host:  %3.1f ms \n", timecopy*nangles);
        
    }
    
    
    if (DEBUG_TIME){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }
    cudaGraphFree(&tempHostGraph,&tempHostElement,&tempHostNode);
    if (DEBUG_TIME){
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        
        mexPrintf("Time to free:  %3.1f ms \n", time);
    }
    return;
    
    
}


// this is fucking slow......... Copying an image of the same size in bytes is x1000 faster. (measured)
void cudaGraphMalloc(const Graph* inGraph, Graph **outGraph, Graph** outGraphHost, Element ** outElementHost, Node** outNodeHost){
    
    Graph* tempHostGraph;
    tempHostGraph = (Graph*)malloc(sizeof(Graph));
    
    //copy constants
    tempHostGraph->nNode = inGraph->nNode;
    tempHostGraph->nElement = inGraph->nElement;
    tempHostGraph->nBoundary = inGraph->nBoundary;
    
    
    
    // copy boundary
    gpuErrchk(cudaMalloc((void**)&(tempHostGraph->boundary), inGraph->nBoundary * sizeof(unsigned int)));
    gpuErrchk(cudaMemcpy(tempHostGraph->boundary, inGraph->boundary, inGraph->nBoundary * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    //Create nodes
    gpuErrchk(cudaMalloc((void**)&(tempHostGraph->node), tempHostGraph->nNode * sizeof(Node)));
    // Auxiliary host nodes
    Node* auxNodeHost = (Node *)malloc(tempHostGraph->nNode * sizeof(Node));
    for (int i = 0; i < tempHostGraph->nNode; i++)
    {
        auxNodeHost[i].nAdjacent = inGraph->node[i].nAdjacent;
        
        //Allocate device memory to position member of auxillary node
        gpuErrchk(cudaMalloc((void**)&(auxNodeHost[i].adjacent_element), inGraph->node[i].nAdjacent*sizeof(unsigned int)));
        gpuErrchk(cudaMemcpy(auxNodeHost[i].adjacent_element, inGraph->node[i].adjacent_element, inGraph->node[i].nAdjacent*sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        //Allocate device memory to position member of auxillary node
        gpuErrchk(cudaMalloc((void**)&(auxNodeHost[i].position), 3 * sizeof(float)));
        gpuErrchk(cudaMemcpy(auxNodeHost[i].position, inGraph->node[i].position, 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        //Copy auxillary host node to device
        gpuErrchk(cudaMemcpy(tempHostGraph->node + i, &auxNodeHost[i], sizeof(Node), cudaMemcpyHostToDevice));
    }
    
    
    //Create elements
    gpuErrchk(cudaMalloc((void**)&(tempHostGraph->element), tempHostGraph->nElement * sizeof(Element)));
    // Auxiliary host nodes
    Element* auxElementHost = (Element *)malloc(tempHostGraph->nElement * sizeof(Element));
    
    for (int i = 0; i < tempHostGraph->nElement; i++)
    {
        auxElementHost[i].nNeighbour = inGraph->element[i].nNeighbour;
        
        //Allocate device memory to position member of auxillary node
        gpuErrchk(cudaMalloc((void**)&(auxElementHost[i].neighbour), inGraph->element[i].nNeighbour*sizeof(unsigned int)));
        gpuErrchk(cudaMemcpy(auxElementHost[i].neighbour, inGraph->element[i].neighbour, inGraph->element[i].nNeighbour*sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        //Allocate device memory to position member of auxillary node
        gpuErrchk(cudaMalloc((void**)&(auxElementHost[i].nodeID), 4 * sizeof(unsigned int)));
        gpuErrchk(cudaMemcpy(auxElementHost[i].nodeID, inGraph->element[i].nodeID, 4 * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        //Copy auxillary host node to device
        gpuErrchk(cudaMemcpy(tempHostGraph->element + i, &auxElementHost[i], sizeof(Element), cudaMemcpyHostToDevice));
    }
    // Copy the host auxiliary Graph to device.
    // Now we have no host access to this structure, so if you want to free its memory, we need to do it with the axiliary host variables.
    gpuErrchk(cudaMalloc((void**)outGraph, sizeof(Graph)));
    gpuErrchk(cudaMemcpy(*outGraph, tempHostGraph, sizeof(Graph), cudaMemcpyHostToDevice));
    
    *outGraphHost = tempHostGraph;
    *outNodeHost = auxNodeHost;
    *outElementHost=auxElementHost;
    return;
}

void cudaGraphFree(Graph** tempHostGraph, Element** tempHostElement, Node** tempHostNode){
    Graph * freeGraph = *tempHostGraph;
    Node * freeNode = *tempHostNode;
    Element * freeElement = *tempHostElement;
    
    for (int i = 0; i < freeGraph->nNode; i++){
        gpuErrchk(cudaFree(freeNode[i].adjacent_element));
        gpuErrchk(cudaFree(freeNode[i].position));
    }
    gpuErrchk(cudaFree(freeGraph->node));
    
    for (int i = 0; i < freeGraph->nElement; i++){
        gpuErrchk(cudaFree(freeElement[i].neighbour));
        gpuErrchk(cudaFree(freeElement[i].nodeID));
    }
    gpuErrchk(cudaFree(freeGraph->element));
    
    gpuErrchk(cudaFree(freeGraph->boundary));
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
