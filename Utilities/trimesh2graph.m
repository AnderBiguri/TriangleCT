function graph=trimesh2graph(TRI,points)
%TRIMESH2GRAPH converts the triangulated mesh into a connected graph
TRI=double(TRI);
points=double(points);

nD=size(points,2);

assert(nD==2 || nD==3,'Only 2D or 3D meshes supported');

for ii=1:size(points,1)
    graph.nodes(ii).positions=single(points(ii,:));
    graph.nodes(ii).neighbour_elems=[];
end

% preallocate.
graph.elements(size(TRI,1)).neighbours=[];

n=neighbors(triangulation(TRI,points));
% For each element, find its node Ids and neighbouring elements
for ii=1:size(TRI,1)
%     nodeids=TRI(ii,:);
%     elem=[];
%     for jj=1:(nD+1)
%         [iind,~]=find(nodeids(jj)==TRI); % this is 75%
%         elem=[elem; iind];
%     end
%     u = unique(elem);
%     
    
%     graph.elements(ii).neighbours = int32((u(histc(elem,u)==nD)));
    aux=n(ii,:);
    aux(isnan(aux))=[];
    graph.elements(ii).neighbours=aux;
    graph.elements(ii).nodeId=uint32(sort(TRI(ii,:)));
    
    
    for jj=1:nD+1
        graph.nodes(TRI(ii,jj)).neighbour_elems=uint32([graph.nodes(TRI(ii,jj)).neighbour_elems,ii]);
    end
    
    % The amount of neihgbours must be nd+1 or nd (in case of the boundary)
    % ABOVE NOT TRUE
%     assert(length(graph.elements(ii).neighbours)<=(nD+1) && length(graph.elements(ii).neighbours)>=nD);
end
% sort neighbours
for ii=1:size(TRI,1)
    ind=[1 2 3; 1 2 4; 1 3 4; 2 3 4];
    nodes=graph.elements(ii).nodeId;
    newind=zeros(nD+1,1);
    for jj=1:length(graph.elements(ii).neighbours)
        nnodes=graph.elements(graph.elements(ii).neighbours(jj)).nodeId;
        for kk=1:nD+1
            if sum(ismember(nodes(ind(kk,:)),nnodes))==3
                newind(kk)=jj;
            end
        end
    end
    neighbours=zeros(1,nD+1,'int32');
    for jj=1:4
        if newind(jj)~=0
            neighbours(jj)=graph.elements(ii).neighbours(newind(jj));
        else
            neighbours(jj)=0;
        end
    end
    graph.elements(ii).neighbours=neighbours;
end


for ii=1:length(graph.nodes)
     graph.nodes(ii).neighbour_elems=sort(graph.nodes(ii).neighbour_elems);
end

% Check which of the elements lie in a boundary of the mesh
boundary=[];
for ii=1:size(TRI,1)
   if nnz(graph.elements(ii).neighbours)<nD+1
       boundary=[boundary ii];
   end
end
graph.boundary_elems=int32(boundary);

if length(graph.boundary_elems)>12
    warning(['Your boundary is not minimal (length=' num2str(length(graph.boundary_elems)) ')' newline ...
        'For optimal GPU computing speed, the length of the boundary has to be as close to 12 as possible.' newline ...
        'If your boundary is significantly larger than this, consider using the fucntion reduceBoundary()'])
end
end
