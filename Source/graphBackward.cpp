/*-------------------------------------------------------------------------
 *
 * MATLAB MEX gateway for projection
 *
 * This file gets the data from MATLAB, checks it for errors and then
 * parses it to C and calls the relevant C/CUDA fucntions.
 *
 * CODE by       Ander Biguri
 *
 *
 *
 *
 *
 *
 */

#include "tmwtypes.h"
#include "mex.h"
#include "matrix.h"
#include "types_TIGRE.hpp"
#include "graph.hpp"
#include "graph_ray_backprojection.hpp"
#include <string.h>

Geometry initGeo(mxArray* geometryMex, const unsigned int nangles);
Graph initGraph(mxArray* graphMex);
void unfuckGraphTemp(Graph* graph);


/*********************************************************************
 *** Mex gateway to CUDA. We assume inputs have been checked and cleaned ***
 ********************************************************************/
void mexFunction(int  nlhs , mxArray *plhs[],
        int nrhs, mxArray const *prhs[])
{
    
    // Check number of inputs
    if (nrhs!=7) {
        mexErrMsgIdAndTxt("MEX:graphForward:InvalidInput", "Invalid number of inputs to MEX file.");
    }
    
    /*********************************************************************
     *********************** First Input ********************************
     ********************************************************************/
    // projetions
    mxArray const * const projMex = prhs[0];
    float const * const proj = static_cast<float const *>(mxGetData(projMex));
    // TODO: do we need to get the size now or are we cool?
    
    
    /*********************************************************************
     *********************** Third Input ********************************
     ********************************************************************/
    // Angles
    
    mxArray const * const anglesMex=prhs[2];
    const mwSize*  angles_size=mxGetDimensions(anglesMex);
    unsigned int nangles=(unsigned int)angles_size[1];
    double const * const angles= static_cast<double const *>(mxGetData(anglesMex));
    
    /*********************************************************************
     *********************** Second Input ********************************
     ********************************************************************/
    // Copy Geometry to C-type struct.
    mxArray * geometryMex=(mxArray*)prhs[1];
    Geometry geo = initGeo(geometryMex,nangles);
    
    
        /*********************************************************************
     *********************** Fourth Input: elements***************************
     ********************************************************************/
    mxArray const * const elementsMex = prhs[3];
    unsigned long const * const elements = static_cast<unsigned long const *>(mxGetData(elementsMex));
    const mwSize*  mwSizeelements=mxGetDimensions(elementsMex);
    if (mwSizeelements[0]>ULONG_MAX)
        mexErrMsgIdAndTxt("MEX:graphForward:TooManelements", "Too Many elements!");
    unsigned long nelements=mwSizeelements[0]/4;
    
    
     /*********************************************************************
     *********************** Fifth Input: vertices***************************
     ********************************************************************/
    mxArray const * const verticesMex = prhs[4];
    float const * const vertices = static_cast<float const *>(mxGetData(verticesMex));
    const mwSize*  mwSizevertices=mxGetDimensions(verticesMex);
    unsigned long nvertices=mwSizevertices[0]/3;
     /*********************************************************************
     *********************** Sixth Input:neighbours***************************
     ********************************************************************/
    mxArray const * const neighboursMex = prhs[5];
    long const * const neighbours = static_cast<long const *>(mxGetData(neighboursMex));
    const mwSize*  mwSizeneighbours=mxGetDimensions(neighboursMex);
    unsigned long nneighbours=mwSizeneighbours[0]/4;
    
    /*********************************************************************
     *********************** seventh Input:boundary***************************
     ********************************************************************/
    mxArray const * const boundaryMex = prhs[6];
    unsigned long const * const boundary = static_cast<unsigned long const *>(mxGetData(boundaryMex));
    const mwSize*  mwSizeboundary=mxGetDimensions(boundaryMex);
    unsigned long nboundary=mwSizeboundary[0];
    
    
    
    /*********************************************************************
     *********************** Create output   ********************************
     ********************************************************************/
    mwSize outsize[1];
    outsize[0]=nelements;

    plhs[0] = mxCreateNumericArray(1, outsize, mxSINGLE_CLASS, mxREAL);
    float *outImage = (float*)mxGetPr(plhs[0]);  // WE will NOT be freeing this pointer!
    float* result = (float*)malloc(nelements * sizeof(float)); // This only allocates memory for pointers
    
    result=outImage;

    /*********************************************************************
     *********************** Run code ********************************
     ********************************************************************/
  
    graphBackwardRay(proj,geo,
                        angles,nangles,
                        vertices,nvertices,
                        elements,nelements,
                        neighbours,nneighbours,
                        boundary,nboundary,
                        result);
//     bruteForwardRay(img,geo,angles,nangles,graph,result);

    
    // we changed the indices to 0-based, change then to 1-based again (as its the same data)
}




Geometry initGeo(mxArray* geometryMex, const unsigned int nangles){
    // Note: most of these are not implemented yet.
    
    const char *fieldnames[14];
    fieldnames[0] = "nVoxel";
    fieldnames[1] = "sVoxel";
    fieldnames[2] = "dVoxel";
    fieldnames[3] = "nDetector";
    fieldnames[4] = "sDetector";
    fieldnames[5] = "dDetector";
    fieldnames[6] = "DSD";
    fieldnames[7] = "DSO";
    fieldnames[8] = "offOrigin";
    fieldnames[9] = "offDetector";
    fieldnames[10]= "accuracy";
    fieldnames[11]= "mode";
    fieldnames[12]= "COR";
    fieldnames[13]= "rotDetector";
    
    // Now we know that all the input struct is good! Parse it from mxArrays to
    // C structures that MEX can understand.
    double * nVoxel, *nDetec; //we need to cast these to int
    double * sVoxel, *dVoxel,*sDetec,*dDetec, *DSO, *DSD;
    double *offOrig,*offDetec,*rotDetector;
    double *  acc, *COR;
    const char* mode;
    int c;
    mxArray    *tmp;
    Geometry geo;
    geo.unitX=1;geo.unitY=1;geo.unitZ=1;
    bool coneBeam=true;
//     mexPrintf("%d \n",nfields);
    for(int ifield=0; ifield<14; ifield++) {
        tmp=mxGetField(geometryMex,0,fieldnames[ifield]);
        if(tmp==NULL){
            //tofix
            continue;
        }
        switch(ifield){
            case 0:
                nVoxel=(double *)mxGetData(tmp);
                // copy data to MEX memory
                geo.nVoxelX=(int)nVoxel[0];
                geo.nVoxelY=(int)nVoxel[1];
                geo.nVoxelZ=(int)nVoxel[2];
                break;
            case 1:
                sVoxel=(double *)mxGetData(tmp);
                geo.sVoxelX=(float)sVoxel[0];
                geo.sVoxelY=(float)sVoxel[1];
                geo.sVoxelZ=(float)sVoxel[2];
                break;
            case 2:
                dVoxel=(double *)mxGetData(tmp);
                geo.dVoxelX=(float)dVoxel[0];
                geo.dVoxelY=(float)dVoxel[1];
                geo.dVoxelZ=(float)dVoxel[2];
                break;
            case 3:
                nDetec=(double *)mxGetData(tmp);
                geo.nDetecU=(int)nDetec[0];
                geo.nDetecV=(int)nDetec[1];
                break;
            case 4:
                sDetec=(double *)mxGetData(tmp);
                geo.sDetecU=(float)sDetec[0];
                geo.sDetecV=(float)sDetec[1];
                break;
            case 5:
                dDetec=(double *)mxGetData(tmp);
                geo.dDetecU=(float)dDetec[0];
                geo.dDetecV=(float)dDetec[1];
                break;
            case 6:
                geo.DSD=(float*)malloc(nangles * sizeof(float));
                DSD=(double *)mxGetData(tmp);
                for (unsigned int i=0;i<nangles;i++){
                    geo.DSD[i]=(float)DSD[i];
                }
                break;
            case 7:
                geo.DSO=(float*)malloc(nangles * sizeof(float));
                DSO=(double *)mxGetData(tmp);
                for (unsigned int i=0;i<nangles;i++){
                    geo.DSO[i]=(float)DSO[i];
                }
                break;
            case 8:
                
                geo.offOrigX=(float*)malloc(nangles * sizeof(float));
                geo.offOrigY=(float*)malloc(nangles * sizeof(float));
                geo.offOrigZ=(float*)malloc(nangles * sizeof(float));
                
                offOrig=(double *)mxGetData(tmp);
                
                for (unsigned int i=0;i<nangles;i++){
                    c=i;
                    geo.offOrigX[i]=(float)offOrig[0+3*c];
                    geo.offOrigY[i]=(float)offOrig[1+3*c];
                    geo.offOrigZ[i]=(float)offOrig[2+3*c];
                }
                break;
            case 9:
                geo.offDetecU=(float*)malloc(nangles * sizeof(float));
                geo.offDetecV=(float*)malloc(nangles * sizeof(float));
                
                offDetec=(double *)mxGetData(tmp);
                for (unsigned int i=0;i<nangles;i++){
                    c=i;
                    geo.offDetecU[i]=(float)offDetec[0+2*c];
                    geo.offDetecV[i]=(float)offDetec[1+2*c];
                }
                break;
            case 10:
                acc=(double*)mxGetData(tmp);
                if (acc[0]<0.001)
                    mexErrMsgIdAndTxt( "MEX:graphForward:unknown","Accuracy should be bigger than 0.001");
                
                geo.accuracy=(float)acc[0];
                break;
            case 11:
                mode="";
                mode=mxArrayToString(tmp);
                if (!strcmp(mode,"parallel"))
                    coneBeam=false;
                break;
            case 12:
                COR=(double*)mxGetData(tmp);
                geo.COR=(float*)malloc(nangles * sizeof(float));
                for (unsigned int i=0;i<nangles;i++){
                    
                    c=i;
                    geo.COR[i]  = (float)COR[0+c];
                }
                break;
                
            case 13:
                geo.dRoll= (float*)malloc(nangles * sizeof(float));
                geo.dPitch=(float*)malloc(nangles * sizeof(float));
                geo.dYaw=  (float*)malloc(nangles * sizeof(float));
                
                rotDetector=(double *)mxGetData(tmp);
                
                for (unsigned int i=0;i<nangles;i++){
                    
                    c=i;
                    geo.dYaw[i]  = (float)rotDetector[0+3*c];
                    geo.dPitch[i]= (float)rotDetector[1+3*c];
                    geo.dRoll[i] = (float)rotDetector[2+3*c];
                    
                }
                break;
            default:
                mexErrMsgIdAndTxt( "MEX:graphForward:unknown","This shoudl not happen. Weird");
                break;
                
        }
    }
    return geo;
}

Graph initGraph(mxArray* graphMex){
    
    Graph graph;
    mxArray* tmp;
    
    // Grab Nodes
    tmp=mxGetField(graphMex,0,"nodes");
    const mwSize* node_size=mxGetDimensions(tmp);
    graph.node=(Node *)malloc(node_size[1] *sizeof(Node));
    graph.nNode=static_cast<unsigned int>(node_size[1]);
    
    mxArray* tmpNeigh;
    mxArray* tmpPositions;
    const mwSize* auxSize;
    for (unsigned int i=0;i<node_size[1];i++){
        tmpNeigh=mxGetField(tmp,i,"neighbour_elems");
        tmpPositions=mxGetField(tmp,i,"positions");
        auxSize=mxGetDimensions(tmpNeigh);
        
        graph.node[i].nAdjacent=auxSize[1];
        graph.node[i].adjacent_element=static_cast<unsigned int *>(malloc(auxSize[1]*sizeof(unsigned int)));
        graph.node[i].adjacent_element=static_cast<unsigned int *>(mxGetData(tmpNeigh));
        for (unsigned int j=0;j<auxSize[1];j++)
            graph.node[i].adjacent_element[j]--;
        
        graph.node[i].position=(float *)malloc(3*sizeof(float));
        graph.node[i].position=static_cast<float *>(mxGetData(tmpPositions));
    }
    // Grab Elements
    tmp=mxGetField(graphMex,0,"elements");
    const mwSize* element_size=mxGetDimensions(tmp);
    graph.element=(Element *)malloc(element_size[1] *sizeof(Element));
    graph.nElement=static_cast<unsigned int>(element_size[1]);
    
    
    mxArray* tmpNodeID;
    for (unsigned int i=0;i<element_size[1];i++){
        tmpNeigh=mxGetField(tmp,i,"neighbours");
        tmpNodeID=mxGetField(tmp,i,"nodeId");
        

        graph.element[i].neighbour=(int*)malloc(4*sizeof(int));
        graph.element[i].neighbour=static_cast<int *>(mxGetData(tmpNeigh));
        
        graph.element[i].nNeighbour=4;        
        for (unsigned int j=0;j< 4;j++)
            graph.element[i].neighbour[j]--;
        graph.element[i].nodeID=(unsigned int*)malloc(4*sizeof(unsigned int));
        graph.element[i].nodeID=static_cast<unsigned int *>(mxGetData(tmpNodeID));
        for (unsigned int j=0;j< 4 ;j++)
            graph.element[i].nodeID[j]--;
    }
    
    // Grab boundary
    tmp=mxGetField(graphMex,0,"boundary_elems");
    const mwSize* boundary_size=mxGetDimensions(tmp);
    
    graph.boundary=(int *)malloc(boundary_size[1] * sizeof(int));
    graph.boundary=static_cast<int *>(mxGetData(tmp));
    
    graph.nBoundary=static_cast<unsigned int>(boundary_size[1]);
    for (unsigned int j=0;j< graph.nBoundary ;j++)
        graph.boundary[j]--;
    
    return graph;
}

// I hope this is temporary. It sets the indices to MATLAB version again. For now, it seems like better idea than
// a deep copy of Graph, we will see later TODO.
void unfuckGraphTemp(Graph* graph){
    
    for (unsigned int i=0;i<graph->nNode;i++){
        for (unsigned int j=0;j<graph->node[i].nAdjacent;j++)
            graph->node[i].adjacent_element[j]++;
    }
    for (unsigned int i=0;i<graph->nElement;i++){
        for (unsigned int j=0;j< 4;j++)
            graph->element[i].neighbour[j]++;
        for (unsigned int j=0;j< 4 ;j++)
            graph->element[i].nodeID[j]++;
    }
    for (unsigned int j=0;j< graph->nBoundary ;j++)
            graph->boundary[j]++;
}



