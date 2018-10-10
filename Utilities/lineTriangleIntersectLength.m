function [d,t1,t2]=lineTriangleIntersectLength(line, triangle)


if size(triangle,2)==2
    d=lineTriangleIntersectLength2D(line, triangle);
else
    [d,t1,t2]=lineTriangleIntersectLength3D(line, triangle);
end


end

function d=lineTriangleIntersectLength2D(line, triangle)
vecline=line(2,1:2)-line(1,1:2);
u=zeros(3,1);
for ii=1:3
    vectri=triangle(mod(ii,3)+1,:)-triangle(ii,:);
    u(ii)=cross2d(triangle(ii,:)-line(1,1:2),vecline)/cross2d(vecline,vectri);
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


function [d,t1,t2]=lineTriangleIntersectLength3D(line, triangle)
vecline=line(2,:)-line(1,:);
triIds=[1 2 3; 1 2 4; 1 3 4; 2 3 4];
tid=1;
t1=0;t2=0;
for ii=1:4
    [intersect(ii), ~,~, t(ii)] = mollerTrumbore(line,triangle(triIds(ii,:),:));

end
t(~intersect)=[];
switch length(t)
    case 0 % no triangle intersected
        d=-1;
        return;
    case 1 % 1 triangle intersected (touchig edge)
        d=0;
        return;
    case 2 % 2 triangles intersected normal case
    case 3 % 3 triangles intersected, edge case. 
        t=uniquetol(t,1e-6);
        if(length(t)<=1)
            d=0;
            return;
        end
end
t=sort(t);           
intersectPoints(1,:)=t(1)*vecline; %+line(1,:)
intersectPoints(2,:)=t(2)*vecline; %+line(1,:)
d=sqrt(sum((intersectPoints(2,:)-intersectPoints(1,:)) .^2) );
t1=t(1);t2=t(2);
end

function val=cross2d(vec1,vec2)
val=vec1(1)*vec2(2)-vec1(2)*vec2(1);
end