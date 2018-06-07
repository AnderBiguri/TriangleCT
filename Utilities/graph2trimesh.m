function [TRI,points]=graph2trimesh(graph)


TRI=csl2mat(graph.elements.nodeId);
points=csl2mat(graph.nodes.positions);

end