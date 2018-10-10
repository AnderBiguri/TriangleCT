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
#include "inMesh_gpu.hpp"
#include <string.h>
#include <climits> 



/*********************************************************************
 *** Mex gateway to CUDA. We assume inputs have been checked and cleaned ***
 ********************************************************************/
void mexFunction(int  nlhs, mxArray *plhs[],
        int nrhs, mxArray const *prhs[])
{
    
    // Check number of inputs
    if (nrhs!=3) {
        mexErrMsgIdAndTxt("MEX:graphForward:InvalidInput", "Invalid number of inputs to MEX file.");
    }
    
    /*********************************************************************
     *********************** First Input: Faces***************************
     ********************************************************************/
    mxArray const * const facesMex = prhs[0];
    unsigned long const * const faces = static_cast<unsigned long const *>(mxGetData(facesMex));
    const mwSize*  mwSizefaces=mxGetDimensions(facesMex);
    if (mwSizefaces[0]>ULONG_MAX)
        mexErrMsgIdAndTxt("MEX:graphForward:TooManyNodes", "Too Many nodes!.");
    unsigned long facessize=mwSizefaces[0]/3;
    
    
     /*********************************************************************
     *********************** Second Input: vertices***************************
     ********************************************************************/
    mxArray const * const verticesMex = prhs[1];
    float const * const vertices = static_cast<float const *>(mxGetData(verticesMex));
    const mwSize*  mwSizevertices=mxGetDimensions(verticesMex);
    unsigned long verticessize=mwSizevertices[0]/3;
     /*********************************************************************
     *********************** Third Input:Input points***************************
     ********************************************************************/
    mxArray const * const pointsMex = prhs[2];
    float const * const points = static_cast<float const *>(mxGetData(pointsMex));
    const mwSize*  mwSizepoints=mxGetDimensions(pointsMex);
    if (mwSizepoints[0]>ULONG_MAX)
        mexErrMsgIdAndTxt("MEX:graphForward:TooManyNeighbours", "Too Many points!.");
    unsigned long pointssize=mwSizepoints[0]/3;
   
    
    /*********************************************************************
     *********************** Create output   ********************************
     ********************************************************************/
    mwSize outsize[1];
    outsize[0]=pointssize;
    plhs[0] = mxCreateNumericArray(1, outsize, mxINT8_CLASS, mxREAL);
    char *outInMesh = (char*)mxGetPr(plhs[0]);  // WE will NOT be freeing this pointer!

    /*********************************************************************
     *********************** Run code ********************************
     ********************************************************************/

    inMesh_gpu(faces,facessize,
               vertices,verticessize,
               points,pointssize,
               outInMesh);

    
}

