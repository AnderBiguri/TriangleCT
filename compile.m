clear mex
mex -largeArrayDims ./Source/graphForward.cpp  ./Source/graph_ray_projection.cpp -outdir ./Mex

mex -largeArrayDims ./Source/graphBackward.cpp  ./Source/ray_back_brute.cu -outdir ./Mex