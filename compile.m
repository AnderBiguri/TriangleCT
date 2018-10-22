clear mex
mex -largeArrayDims ./Source/graphForward.cpp  ./Source/graph_ray_projection.cu -outdir ./Mex
mex -largeArrayDims ./Source/graphBackward.cpp  ./Source/graph_ray_backprojection.cu -outdir ./Mex