function graph=trimesh2graph(TRI,points)
%TRIMESH2GRAPH converts the triangulated mesh into a connected graph


nD=size(points,2);

assert(nD==2 || nD==3,'Only 2D or 3D meshes supported');

for ii=1:size(points,1)
    graph.nodes(ii).positions=points(ii,:);
    graph.nodes(ii).neighbour_elems=[];
end

% preallocate.
graph.elements(size(TRI,1)).neighbours=[];


% For each element, find its node Ids and neighbouring elements
for ii=1:size(TRI,1)
    nodeids=TRI(ii,:);
    elem=[];
    for jj=1:(nD+1)
        [iind,~]=find(nodeids(jj)==TRI);
        elem=[elem; iind];
    end
    u = unique(elem);
    
    
    graph.elements(ii).neighbours = uint32(sort(u(histc(elem,u)==nD)));
    
    graph.elements(ii).nodeId=TRI(ii,:);
    
    for jj=1:nD+1
        graph.nodes(TRI(ii,jj)).neighbour_elems=uint32([graph.nodes(TRI(ii,jj)).neighbour_elems,ii]);
    end
    
    % The amount of neihgbours must be nd+1 or nd (in case of the boundary)
    % ABOVE NOT TRUE
%     assert(length(graph.elements(ii).neighbours)<=(nD+1) && length(graph.elements(ii).neighbours)>=nD);
end


for ii=1:length(graph.nodes)
     graph.nodes(ii).neighbour_elems=sort(graph.nodes(ii).neighbour_elems);
end

% Check which of the elements lie in a boundary of the mesh
boundary=[];
for ii=1:size(TRI,1)
   if length(graph.elements(ii).neighbours)==nD
       boundary=[boundary ii];
   end
end
graph.boundary_elems=boundary;


end