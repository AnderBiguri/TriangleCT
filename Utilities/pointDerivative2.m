function derv=pointDerivative2(ray,graph,nodeid)
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
        tri=csl2mat(graph.nodes([graph.elements(ii).nodeId(1) graph.elements(ii).nodeId(2) graph.elements(ii).nodeId(3)]).positions);
        auxind=[1 2 3];
        nodesubid=auxind(nodesubid);
    elseif f1&&f3
        tri=csl2mat(graph.nodes([graph.elements(ii).nodeId(2) graph.elements(ii).nodeId(1) graph.elements(ii).nodeId(3)]).positions);
        auxind=[2 1 3];
        nodesubid=auxind(nodesubid);
    elseif f2&&f3
        tri=csl2mat(graph.nodes([graph.elements(ii).nodeId(1) graph.elements(ii).nodeId(3) graph.elements(ii).nodeId(2)]).positions);
        auxind=[1 3 2];
        nodesubid=auxind(nodesubid);
    else
        % if ray does not intersect triangle, derivative is zero.
        continue;
    end
    
    % All of the avobe may not be of need anymore.
    r=ray(2,:)-ray(1,:); r=r./norm(r);
    
    M1=[r; tri(2,:)-tri(3,:)].';
    M2=[r; tri(2,:)-tri(1,:)].';
    
    t1=M2\(tri(2,:)-ray(1,:)).';t1=t1(1);
    t2=M1\(tri(2,:)-ray(1,:)).';t2=t2(1);
    %     inv2x2=@(M)1/(M(1)*M(4)-M(2)*M(3))*[M(4) -M(3); -M(2) M(1)];
    %     Minv1=inv2x2([r; tri(2,:)-tri(3,:)].');
    %     Minv2=inv2x2([r; tri(2,:)-tri(1,:)].');
    %     deriv1x=[nodesubid==2 -(nodesubid==3);0 0];   deriv1y=[nodesubid==2 -(nodesubid==3);0 0];
    %     deriv2x=[nodesubid==2 -(nodesubid==1);0 0];   deriv1y=[nodesubid==2 -(nodesubid==1);0 0];
    
    if     nodesubid==1
        aux=(+M2\[0 -1;0 0]/M2)*(tri(2,:)-ray(1,:)).';
        derv(1,ii)=aux(1)*sign(t2-t1);
        aux=(+M2\[0 0;0 -1]/M2)*(tri(2,:)-ray(1,:)).';
        derv(2,ii)=aux(1)*sign(t2-t1);
    elseif nodesubid==3
        aux=(-M1\[0 -1;0 0]/M1)*(tri(2,:)-ray(1,:)).';
        derv(1,ii)=aux(1)*sign(t2-t1);
        aux=(-M1\[0 0;0 -1]/M1)*(tri(2,:)-ray(1,:)).';
        derv(2,ii)=aux(1)*sign(t2-t1);
    else
        aux=(-M1\[0 1;0 0]/M1+M2\[0 1;0 0]/M2)*(tri(2,:)-ray(1,:)).'+(M1\[1;0]-M2\[1;0]);
        derv(1,ii)=aux(1)*sign(t2-t1);
        aux=(-M1\[0 0;0 1]/M1+M2\[0 0;0 1]/M2)*(tri(2,:)-ray(1,:)).'+(M1\[0;1]-M2\[0;1]);
        derv(2,ii)=aux(1)*sign(t2-t1);
    end
    
    
end
end