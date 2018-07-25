function [intersected]=isLineTriangleIntersect(line,triangle)



if size(triangle,2)==2
    intersected=isLineTriangleIntersect2D(line,triangle);
else
    intersected=isLineTriangleIntersect3D(line,triangle);
end
end


function intersected=isLineTriangleIntersect3D(line,triangle)

    intersected=mollerTrumbore(line,triangle([1 2 3],:));
    if intersected; return; end
    intersected=mollerTrumbore(line,triangle([1 2 4],:));
    if intersected; return; end
    intersected=mollerTrumbore(line,triangle([1 3 4],:));
    if intersected; return; end
    intersected=mollerTrumbore(line,triangle([2 3 4],:));
    if intersected; return; end
end


function [intersected]=isLineTriangleIntersect2D(line,triangle)
% From:
% https://gamedev.stackexchange.com/questions/21096/what-is-an-efficient-2d-line-segment-versus-triangle-intersection-test

% Check whether segment is outside one of the three half-planes delimited by the triangle.
f1 = whatSide(line(1,:), triangle(3,:), triangle(1,:), triangle(2,:));
f2 = whatSide(line(2,:), triangle(3,:), triangle(1,:), triangle(2,:));
if (f1 < 0 && f2 < 0) 
    intersected=false;
    return;
end
f3 = whatSide(line(1,:), triangle(1,:), triangle(2,:), triangle(3,:));
f4 = whatSide(line(2,:), triangle(1,:), triangle(2,:), triangle(3,:));
if (f3 < 0 && f4 < 0) 
    intersected=false;
    return;
end
f5 = whatSide(line(1,:), triangle(2,:), triangle(3,:), triangle(1,:));
f6 = whatSide(line(2,:), triangle(2,:), triangle(3,:), triangle(1,:));
if (f5 < 0 && f6 < 0) 
    intersected=false;
    return;
end
% Check whether triangle is totally inside one of the two half-planes delimited by the segment.
f7 = whatSide(triangle(1,:),  triangle(2,:), line(1,:), line(2,:));
f8 = whatSide( triangle(2,:), triangle(3,:), line(1,:), line(2,:));
if  (f7 > 0 && f8 > 0) 
    intersected=false;
    return;
end
% Not important for our applications, the ray will always be longer than
% the triangles.

% % If both segment points are strictly inside the triangle, we are not intersecting either
% if (f1 > 0 && f2 > 0 && f3 > 0 && f4 > 0 && f5 > 0 && f6 > 0)
%     intersected=false;
%     return;
% end

intersected=true;
end
function sideId=whatSide(P,Q,A,B)


% compare sign of Z cross product
z1=(B(1)-A(1))*(P(2)-A(2))-(P(1)-A(1))*(B(2)-A(2));
z2=(B(1)-A(1))*(Q(2)-A(2))-(Q(1)-A(1))*(B(2)-A(2));

sideId=z1*z2;
end