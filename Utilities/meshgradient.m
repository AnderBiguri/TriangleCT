function g=meshgradient(graph,element_vals)
%MESHGRADIENT(..) computes the gradient of a triangular mesh using function
% values defined in the centroid of the mesh by fitting a plane to the
% nodes surrounding elements' centroids. 
% https://math.stackexchange.com/a/2632616/44383
% Yer the Chosen One, Harry

  


nD=size(graph.elements(1).nodeId,2)-1;
assert(nD==2 || nD==3,'Only 2D or 3D meshes supported');


g=zeros(size(graph.nodes,2),nD);

% This method needs the adjacency of the node-elements

% We are going to need the centroids of the elements
centroids=zeros(size(graph.elements,2),nD);
for ii=1:length(centroids)
   centroids(ii,:)= mean(csl2mat(graph.nodes(graph.elements(ii).nodeId).positions));
end


% Neighbour element based approach
for ii=1:size(graph.nodes,2)
    diri=ones(length(graph.nodes(ii).neighbour_elems),nD+1);
    d=zeros(length(graph.nodes(ii).neighbour_elems),1);
    for jj=1:length(graph.nodes(ii).neighbour_elems)
        diri(jj,1:nD)=centroids(graph.nodes(ii).neighbour_elems(jj),:)-graph.nodes(ii).positions;
        d(jj)=sqrt(sum((centroids(graph.nodes(ii).neighbour_elems(jj),:)-graph.nodes(ii).positions).^2,2));
    end
    gradi=element_vals(graph.nodes(ii).neighbour_elems);
    aux=lscov(diri,gradi,1/d.^2);
    g(ii,:)=aux(1:nD);
end

end