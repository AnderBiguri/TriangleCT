function [nodevals,grad]=meshInterpolateNodes(graph,trivals)

graph=fixgraph(graph);
nD=size(graph.elements(1).nodeId,2)-1;
assert(nD==2 || nD==3,'Only 2D or 3D meshes supported');

if nargin ==2
    grad=zeros(size(graph.nodes,2),nD);
end
% We are going to need the centroids of the elements
centroids=zeros(size(graph.elements,2),nD);
for ii=1:length(centroids)
    centroids(ii,:)= mean(csl2mat(graph.nodes(graph.elements(ii).nodeId).positions));
end


% Neighbour element based approach
for ii=1:size(graph.nodes,2)
    diri=ones(length(graph.nodes(ii).neighbour_elems),nD+1);
    dsq=zeros(length(graph.nodes(ii).neighbour_elems),1);
    for jj=1:length(graph.nodes(ii).neighbour_elems)
        diri(jj,1:nD)=centroids(graph.nodes(ii).neighbour_elems(jj),:)-graph.nodes(ii).positions;
        
    end
    gradi=trivals(graph.nodes(ii).neighbour_elems);
    aux=diri\gradi;
    if nargin ==2
        grad(ii,:)=aux(1:nD);
    end
    nodevals(ii,1)=aux(end);
end
end