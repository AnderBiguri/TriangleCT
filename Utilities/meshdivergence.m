function div=meshdivergence(graph,nodevects)
% computes the divergence of a triangle mesh


nD=size(graph.elements(1).nodeId,2)-1;
assert(nD==2 || nD==3,'Only 2D or 3D meshes supported');

divX=meshgradient(graph,nodevects(:,1));
divX=divX(:,1);
divY=meshgradient(graph,nodevects(:,2));
divY=divY(:,2);

divZ=[];
if nD==3
  divZ=meshgradient(graph,nodevects(:,3));
  divZ=divZ(:,3);  
end
div=sum([divX divY divZ],2);
end