function [nodes,elements,neighbours,boundary]=graphStruct2graphArray(graph)

nodes=(csl2mat(graph.nodes.positions));
nodes=nodes.';
nodes=nodes(:); 

elements=uint32(csl2mat(graph.elements.nodeId));
elements=elements.';
elements=elements(:); 

neighbours=int32(csl2mat(graph.elements.neighbours));
neighbours=neighbours.';
neighbours=neighbours(:); 

boundary=uint32(csl2mat(graph.boundary_elems)).';
end