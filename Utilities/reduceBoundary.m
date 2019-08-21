function [graph,values]=reduceBoundary(graph,values,new_values)
% REDUCEBOUNDARY  reduces the boundary elements of a triangulation to 12
%   [TRI,vertices] = REDUCEBOUNDARY(TRI,vertices)
%
%   If your triangulation has too many elements in the boundary, the
%   projection and backprojection code will have trouble initializing the
%   rays and will take very long. This function adds a small box around th
%   econvex triangulation [TRI,vertices] by adding 8 new points, and
%   triangulates the result. The output mesh will have only 12 boundary
%   elements.

if nargout>1&&nargin<2
    error('Element values added as output but no input provided.')
end
if nargin>1 &&nargin<3
    new_values=0;
end
[TRI,vertices]=graph2trimesh(graph);
[~,vertices]=freeBoundary(triangulation(double(TRI),double(vertices)));
xr=[min(vertices(:,1)) max(vertices(:,1))];
yr=[min(vertices(:,2)) max(vertices(:,2))];
zr=[min(vertices(:,3)) max(vertices(:,3))];

extra=0.05;
safetypassed=false;
while ~safetypassed
    % Lets give it a bit of surrounding
    xr=xr+[-extra*diff(xr) +extra*diff(xr)];
    yr=yr+[-extra*diff(yr) +extra*diff(yr)];
    zr=zr+[-extra*diff(zr) +extra*diff(zr)];
    
    vertices2=[xr(1) yr(1) zr(1);
        xr(1) yr(1) zr(2);
        xr(1) yr(2) zr(1);
        xr(1) yr(2) zr(2);
        xr(2) yr(1) zr(1);
        xr(2) yr(1) zr(2);
        xr(2) yr(2) zr(1);
        xr(2) yr(2) zr(2);
        vertices];
    
    % Safety check
    vertices3=unique(single(vertices2),'rows');
    vertices3=double(vertices3);
    
    if isequal(size(vertices3,1),size(vertices,1)+8)
        safetypassed=true;
    else
        safetypassed=false;
        extra=extra+0.05;
    end
end
vertices=double(vertices2);
clear vertices2;
%%

TRI2=delaunay(vertices(:,1),vertices(:,2),vertices(:,3));

% now we need to know which values vertices 9-end corespond to in the real
% mesh

boundary_elem=unique(csl2mat(graph.elements(graph.boundary_elems).nodeId));
boundary_elem_nodes=csl2mat(graph.nodes(boundary_elem).positions);

TRIgoodindx=zeros(size(TRI2));
for ii=1:8
    TRIgoodindx(TRI2==ii)=ii;
end
for ii=9:size(vertices,1)
    ind=sum(vertices(ii,:)==boundary_elem_nodes,2)==3;
    TRIgoodindx(TRI2==ii)=boundary_elem(ind)+8;
end

%% we just don't know if the input is a delaunay triangulation. 
% As we don't know, we can not just return this, so we will get only the
% new triangles and add it to the previosu triangulation.

TRIgoodindx=sort(TRIgoodindx,2);
TRIgoodindx=sortrows(TRIgoodindx);
ind=find(TRIgoodindx(:,1)>8,1);


% All triangles smaller than ind contain new points, meaning are the new
% ones that we need to add to th eexisting graph.
% get an idx of the boundary triagles
% graphaux=trimesh2graph(TRI2,vertices);
% 
% 
% nvertices=length(graph.nodes);
% 
% graph.nodes(nvertices+1).positions=[xr(1) yr(1) zr(1)];
% graph.nodes(nvertices+2).positions=[xr(1) yr(1) zr(2)];
% graph.nodes(nvertices+3).positions=[xr(1) yr(2) zr(1)];
% graph.nodes(nvertices+4).positions=[xr(1) yr(2) zr(2)];
% graph.nodes(nvertices+5).positions=[xr(2) yr(1) zr(1)];
% graph.nodes(nvertices+6).positions=[xr(2) yr(1) zr(2)];
% graph.nodes(nvertices+7).positions=[xr(2) yr(2) zr(1)];
% graph.nodes(nvertices+8).positions=[xr(2) yr(2) zr(2)];


[TRI,vertices]=graph2trimesh(graph);
TRI=[TRI+8;TRIgoodindx(1:ind-1,:)]; 
vertices=[xr(1) yr(1) zr(1);
        xr(1) yr(1) zr(2);
        xr(1) yr(2) zr(1);
        xr(1) yr(2) zr(2);
        xr(2) yr(1) zr(1);
        xr(2) yr(1) zr(2);
        xr(2) yr(2) zr(1);
        xr(2) yr(2) zr(2);
        vertices];
    
graph=trimesh2graph(double(TRI),double(vertices));
if nargin>1&&nargout>1
   values=[values;ones(size(TRI(1:ind-1,:),1),1,class(values))*cast(new_values,class(values))] ;
end

end