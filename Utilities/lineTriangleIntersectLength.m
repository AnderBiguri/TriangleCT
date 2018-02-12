function d=lineTriangleIntersectLength(line, triangle)

vecline=line(2,:)-line(1,:);
u=zeros(3,1);
for ii=1:3
    vectri=triangle(mod(ii,3)+1,:)-triangle(ii,:);
    u(ii)=cross2d(triangle(ii,:)-line(1,:),vecline)/cross2d(vecline,vectri);
end
% assert(sum( u>0 & u<1 )==2,'3 intersections detected, imposible');
if sum( u>0 & u<1 )~=2
    d=-1;
    return
end
% In case of assertion fail, debug plot:
% hold on
% patch(triangle(:,1),triangle(:,2),[0,0,0])
% plot(line(:,1),line(:,2),'r')

intersectionInd= find(u>0 & u<1);
intersectPoints=triangle(intersectionInd,:)+u(u>0 & u<1).*(triangle(mod(intersectionInd,3)+1,:)-triangle(intersectionInd,:));
d=sqrt(sum((intersectPoints(2,:)-intersectPoints(1,:)) .^2) );

end
function val=cross2d(vec1,vec2)
    val=vec1(1)*vec2(2)-vec1(2)*vec2(1);
end