function out=isLineSegmentIntersect(ray,segment)
% https://stackoverflow.com/a/9997374/1485872

f1 = orientation(ray(1,:), ray(2,:),segment(1,:) );
f2 = orientation(ray(1,:), ray(2,:), segment(2,:));
f3 = orientation(segment(1,:), segment(2,:), ray(1,:));
f4 = orientation(segment(1,:), segment(2,:), ray(2,:));
out=false;
if f1~=f2 && f3~=f4
    out=true;
end

% we dont want true if they are colinear.
end

function out=orientation(p,q,r)
% // To find orientation of ordered triplet (p, q, r).
% // The function returns following values
% // 0 --> p, q and r are colinear
% // 1 --> Clockwise
% // 2 --> Counterclockwise
out= (q(2)-p(2))*(r(1)-q(1))-(q(1)-p(1))*(r(2)-q(2));
if out == 0
    return
else
    if out>0
        out=1;
    else
        out=2;
    end
end
end
% 
% function out=onSegment(p,q, r)
% % // Given three colinear points p, q, r, the function checks if
% % // point q lies on line segment 'pr'
% out=(q(1) <= max(p(1), r(1)) && q(1) >= min(p(1), r(1)) &&...
%     q(2) <= max(p(2), r(2)) && q(2) >= min(p(2), r(2)))
% end