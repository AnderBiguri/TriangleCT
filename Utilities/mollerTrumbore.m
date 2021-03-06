function [flag, u, v, t] = mollerTrumbore (ray,tri)
% Ray/triangle intersection using the algorithm proposed by Moller and Trumbore (1997).
%
% IMPORTANT NOTE: Assumes infinite legth rays.
% Input:
%    ray(1,:) : origin.
%    d : direction.
%    tri(1,:), tri(2,:), tri(3,:): vertices of the triangle.
% Output:
%    flag: (0) Reject, (1) Intersect.
%    u,v: barycentric coordinates.
%    t: distance from the ray origin.
% Author: 
%    Jesus Mena

    d=ray(2,:)-ray(1,:);
    epsilon = 0.00001;

    e1 = tri(2,:)-tri(1,:);
    e2 = tri(3,:)-tri(1,:);
    q  = cross(d,e2);
    a  = dot(e1,q); % determinant of the matrix M

    if (a>-epsilon && a<epsilon) 
        % the vector is parallel to the plane (the intersection is at infinity)
        [flag, u, v, t] = deal(0,0,0,0);
        return;
    end
    
    f = 1/a;
    s = ray(1,:)-tri(1,:);
    u = f*dot(s,q);
    
    if (u<0.0)
        % the intersection is outside of the triangle
        [flag, u, v, t] = deal(0,0,0,0);
        return;          
    end
    
    r = cross(s,e1);
    v = f*dot(d,r);
    
    if (v<0.0 || u+v>1.0)
        % the intersection is outside of the triangle
        [flag, u, v, t] = deal(0,0,0,0);
        return;
    end
    if nargout>3
        t = f*dot(e2,r); % verified! 
    end
%     fprintf("%.16f %.16f %.16f\n",u,v,t);
    flag = 1;
    return
end
