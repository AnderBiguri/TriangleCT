function pts=populateTriangle(fv,spacing,popSurf,popEdges)
pts=[];

if popSurf
    for ii=1:size(fv.faces,1)
        tri=fv.vertices(fv.faces(ii,:),:);
        pts=[pts; populateSurface(tri,spacing)];
        
    end
end
if popEdges
    pts=[pts; populateEdges(fv,spacing)];
end

pts=unique(pts,'rows');
end
function pts=populateEdges(fv,spacing)

pts=[];
nD=3;
for jj=1:size(fv.faces,1)
    hold on
    % get elemet neigbrous
    nodeids=fv.faces(jj,:);
    elem=[];
    for kk=1:nD
        [iind,~]=find(nodeids(kk)==fv.faces);
        elem=[elem; iind];
    end
    u = unique(elem);
    
    
    neighbours = uint32(sort(u(histc(elem,u)==(nD-1))));
    
    
    
    
    tri=fv.vertices(fv.faces(jj,:),:);
    normalt=cross(tri(1,:)-tri(3,:),tri(2,:)-tri(3,:));
    % Check which edges need to be populated. For that, compute normals of
    % neighbour triangles, and if parallel, do not populate edge.
    for neigh=1:size(neighbours,1)
        
        neightri=fv.vertices(fv.faces(neighbours(neigh),:),:);
        [~,~,ib]=intersect(neightri,tri,'rows'); ib=sort(ib);
        if isequal(ib,[1 2]')
            edge=1;
        elseif isequal(ib,[2 3]')
            edge=2;
        elseif isequal(ib,[1 3]')
            edge=3;
        else
            error
        end
        
        normaln=cross(neightri(1,:)-neightri(3,:),neightri(2,:)-neightri(3,:));
        theta= abs(atan2(norm(cross(normalt,normaln)),dot(normalt,normaln)));
        isparallel(edge) =abs(theta-pi)<1e-6|| abs(theta)<1e-6;
        
    end
    
    eqline=@(p1,p2,t)(p1+t.'.*(p2-p1));
    ts=@(p1,p2,sp)(linspace(0,1,ceil(sqrt(sum((p2-p1).^2)))/sp));
    for ii=1:3
        if isparallel(ii)
            continue
        end
        t=ts(tri(ii,:),tri(mod(ii,3)+1,:),spacing);
        t=t(2:end-1);
        if isempty(t)
            continue
        end
        pts=[pts; eqline(tri(ii,:),tri(mod(ii,3)+1,:),t)];
        plot3(pts(:,1),pts(:,2),pts(:,3),'r.')

    end

end
end



function pts=populateSurface(tri,spacing)
% %
% First we need to create a 2D coordinate system for the triangle
% %
% https://math.stackexchange.com/questions/856666/how-can-i-transform-a-3d-triangle-to-xy-plane
triaux=tri-tri(1,:);
u=triaux(2,:)./sqrt(sum(triaux(2,:).^2));
w=cross(u,triaux(3,:));
w=w./sqrt(sum(w.^2));
v=cross(u,w);
R=[u.',v.',w.'];

%reuse variable
triaux=(R.'*triaux.').';
% get domain
xrange=[min(triaux(:,1)) max(triaux(:,1))];
yrange=[min(triaux(:,2)) max(triaux(:,2))];
% populate
% 2)
% pts = poissonDisc([diff(xrange), diff(yrange)],1);
pts=rand(ceil(diff(xrange)*diff(yrange)/spacing),2).*[diff(xrange), diff(yrange)];
pts=pts+[xrange(1) yrange(1)];

% Delete points not in triangle
in=isPointInTriangle2D(triaux(:,1:2),pts);
pts=[pts,zeros(size(pts,1),1)];
pts=pts(in,:);
pts=(R*pts.').';
pts=pts+tri(1,:);

end

function in=isPointInTriangle2D(tri,pts)
in=0;
area= 0.5 *(-tri(2,2)*tri(3,1) + tri(1,2)*(-tri(2,1) + tri(3,1)) + tri(1,1)*(tri(2,2) - tri(3,2)) + tri(2,1)*tri(3,2));
if sign(area)==-1
    tri=[tri(1,:); tri(3,:); tri(2,:)] ;
end
s = (tri(1,2)*tri(3,1) - tri(1,1)*tri(3,2) + (tri(3,2) - tri(1,2))*pts(:,1) + (tri(1,1) - tri(3,1))*pts(:,2));
t = (tri(1,1)*tri(2,2) - tri(1,2)*tri(2,1) + (tri(1,2) - tri(2,2))*pts(:,1) + (tri(2,1) - tri(1,1))*pts(:,2));

in=s>0 & t>0 & s+t<2*abs(area);
end