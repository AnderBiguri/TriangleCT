function [nodes,elements,neighbours,boundary,tree]=graphStruct2graphArray(graph)

nodes=(csl2mat(graph.nodes.positions));
nodes=nodes.';
nodes=nodes(:); 
if ispc
elements=uint32(csl2mat(graph.elements.nodeId));
else
elements=uint64(csl2mat(graph.elements.nodeId));
end
elements=elements.';
elements=elements(:); 
if ispc
neighbours=int32(csl2mat(graph.elements.neighbours));
else
neighbours=int64(csl2mat(graph.elements.neighbours));
end
neighbours=neighbours.';
neighbours=neighbours(:); 
if ispc
boundary=uint32(csl2mat(graph.boundary_elems)).';
else
boundary=uint64(csl2mat(graph.boundary_elems)).';
end
if nargout>4
tree=graph.tree;
end
end