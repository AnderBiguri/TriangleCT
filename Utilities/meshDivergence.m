function div=meshDivergence(graph,nodevects)
% computes the divergence of a triangle mesh


nD=size(graph.elements(1).nodeId,2)-1;
assert(nD==2 || nD==3,'Only 2D or 3D meshes supported');
divZ=[];
% X
divX=meshGradient(graph,nodevects(:,1));
divX=divX(:,1);
%Y
divY=meshGradient(graph,nodevects(:,2));
divY=divY(:,2);
%Z
if nD==3
  divZ=meshGradient(graph,nodevects(:,3));
  divZ=divZ(:,3);  
end
% divergence
div=sum([divX divY divZ],2);
end