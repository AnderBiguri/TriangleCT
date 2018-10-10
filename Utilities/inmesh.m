function in=inmesh(fv,points,gpu)

% gpu=1;

if gpu
    vertices=fv.vertices';
    faces=fv.faces'-1;
    points=points';
    in=inMesh(uint32(faces(:)),single(vertices(:)),single(points(:)));
%     in=in>0.5;
    return;
end


%INMESH tells you if a poitn is inside a triangulated surface mesh
maxZ=max(fv.vertices(:,3));
counts=zeros(size(points,1),1);

rangex=[min(fv.vertices(:,1)) max(fv.vertices(:,1))];
rangey=[min(fv.vertices(:,2)) max(fv.vertices(:,2))];
rangez=[min(fv.vertices(:,3)) max(fv.vertices(:,3))];
maxs=[max(fv.vertices(:,1)),max(fv.vertices(:,2)),max(fv.vertices(:,3))];
[~,dimmin]=min(diff([rangex;rangey;rangez],[],2));
% dimmin=3;
for ii=1:size(points,1)
    
    ray=[points(ii,:);points(ii,:)];ray(2,dimmin)=maxs(dimmin)+1;
    for jj=1:size(fv.faces,1)
        v=fv.vertices(fv.faces(jj,:),:);
        if all(v(:,1)<ray(1,1))||all(v(:,1)>ray(2,1))||all(v(:,2)<ray(1,2))||all(v(:,2)>ray(2,2))||all(v(:,3)<ray(1,3))||all(v(:,3)>ray(2,3))
            continue;
        end
                isin=mollerTrumbore(ray, fv.vertices(fv.faces(jj,:),:));
%         isin=watertight(ray, fv.vertices(fv.faces(jj,:),:));
        counts(ii)=counts(ii)+isin;
    end
end
in=mod(counts,2);
% in=counts;
end