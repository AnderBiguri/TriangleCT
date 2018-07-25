function in=inmesh(fv,points)
%INMESH tells you if a poitn is inside a triangulated surface mesh
%
% cla
%  lighting none

maxZ=max(fv.vertices(:,3));
counts=zeros(size(points,1),1);
for ii=1:size(points,1)
%     patch(fv,'FaceColor',       [0.8 0.8 1.0], ...
%         'FaceAlpha',       0.1,           ...
%         'EdgeColor',       'none');
%         'FaceLighting',    'gouraud',     ...
%         'AmbientStrength', 0.15);
    
    % Add a camera light, and tone down the specular highlighting
%     camlight;
%     material('dull');
%      lighting none
% 
%     % Fix the axes scaling, and set a nice view angle
%     axis('image');
%     view([-135 35]);
%     
%     hold on
    
    ray=[points(ii,:);points(ii,1:2) maxZ+1];
%     plot3(ray(:,1),ray(:,2),ray(:,3),'r');
%     plot3(ray(1,1),ray(1,2),ray(1,3),'b.','linewidth',2);
    for jj=1:size(fv.faces,1)
        v=fv.vertices(fv.faces(jj,:),:);
        if all(v(:,3)<ray(1,3))
            continue;
        end
        isin=mollerTrumbore(ray, fv.vertices(fv.faces(jj,:),:));
%         if isin
%             patch('Faces',fv.faces(jj,:),'Vertices',fv.vertices,'facealpha',0.3,'FaceColor','g');
%         else
%             patch('Faces',fv.faces(jj,:),'Vertices',fv.vertices,'facealpha',0.3,'FaceColor','r');
%         end
%         drawnow
        counts(ii)=counts(ii)+isin;
    end
%     pause
%     cla
end
in=mod(counts,2);
end