function derv=pointDerivative(ray,graph,nodeid,lengths)
% 

% To evaluate the derivative of the mesh with the nodes we have a logn
% equation. However that equation needs a very specific order of nodes.
% This function assumes that the line does indeed intesect the triangle.

% in essence, the common node of 2 lines has to be the middle one.

% Create a TRIx3x2 matrix

nD=size(graph.nodes(1).positions,2);
nT=size(graph.elements,2);
nN=size(graph.nodes,2);

triangles=reshape(csl2mat(graph.nodes(csl2mat(graph.elements(:).nodeId)).positions),nT,nD+1,nD);


derv=zeros(2,nT);
for ii=1:nT
    nodesubid=find(graph.elements(ii).nodeId==nodeid,1);
    if isempty(nodesubid)
        % if node is not in element, then derivative is zero.
        continue;
    end
    f1=isLineSegmentIntersect(ray,[squeeze(triangles(ii,1,:))';squeeze(triangles(ii,2,:))']);
    f2=isLineSegmentIntersect(ray,[squeeze(triangles(ii,2,:))';squeeze(triangles(ii,3,:))']);
    f3=isLineSegmentIntersect(ray,[squeeze(triangles(ii,3,:))';squeeze(triangles(ii,1,:))']);
    
    % lets ignore f3 if this happens
    special=(f1+f2+f3)==3;
    
    
    if     f1&&f2
        tri=csl2mat(graph.nodes([graph.elements(ii).nodeId(1) graph.elements(ii).nodeId(2) graph.elements(ii).nodeId(3)]).positions).';
        auxind=[1 2 3];
        nodesubid=auxind(nodesubid);
    elseif f1&&f3
        tri=csl2mat(graph.nodes([graph.elements(ii).nodeId(2) graph.elements(ii).nodeId(1) graph.elements(ii).nodeId(3)]).positions).';
        auxind=[2 1 3];
        nodesubid=auxind(nodesubid);
    elseif f2&&f3
        tri=csl2mat(graph.nodes([graph.elements(ii).nodeId(1) graph.elements(ii).nodeId(3) graph.elements(ii).nodeId(2)]).positions).';
        auxind=[1 3 2];
        nodesubid=auxind(nodesubid);
    else
        % if ray does not intersect triangle, derivative is zero.
        continue;
    end
    
%     [N1,D1]=numden(lengths.dl(1,nodesubid));
%     [N2,D2]=numden(lengths.dl(2,nodesubid));
%     derv(1,ii)=double(subs(subs(N1/D1,[sym('x_r',[1 2]); sym('y_r',[1 2])],ray.'),lengths.edge,tri));
%     derv(2,ii)=double(subs(subs(N2/D2,[sym('x_r',[1 2]); sym('y_r',[1 2])],ray.'),lengths.edge,tri));
%     
    derv(1,ii)=lengths.dl{1,nodesubid}(tri(1,1),tri(1,2),tri(1,3),ray(1,1),ray(2,1), tri(2,1),tri(2,2),tri(2,3),ray(1,2) ,ray(2,2) );
    derv(2,ii)=lengths.dl{2,nodesubid}(tri(1,1),tri(1,2),tri(1,3),ray(1,1),ray(2,1), tri(2,1),tri(2,2),tri(2,3),ray(1,2) ,ray(2,2) );

    
end
end