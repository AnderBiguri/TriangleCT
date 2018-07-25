function [updates,graph,pos,residualnorm]=nodeOptimization(proj,img,geo,graph,angles,niter)


nangles=size(angles,2);
% result=zeros(size(graph.elements,2),1);
residual=zeros(size(proj));
lambda=0.0001;
update=[0;0];
detector=[-(geo.DSD-geo.DSO)*ones(geo.nDetector(1),1),(-geo.sDetector(1)/2+geo.dDetector(1)/2:geo.dDetector(1):geo.sDetector(1)/2-geo.dDetector(1)/2)'];
source=[+geo.DSO,0];
R =@(theta)( [cos(theta) -sin(theta); sin(theta) cos(theta)]);

nodeIds=5; %list of nodes we want to optimize
for ii=1:niter
    residual=meshForward(img,geo,angles,graph)-proj;
    residualnorm(ii)=norm(residual(:));
    for ll=nodeIds % the ones we want.
        
        % Loop every projection
        for jj=1:nangles
            % for every pixel in the detector
            
            % this loop for the derivatives
            for kk=1:geo.nDetector(1)
                %
                
                source=[+geo.DSO, detector(kk,2)];
                ray=[source;detector(kk,:)];
                ray=(R(angles(jj))*ray.').';
                
                
                % the derivative of 1 ray, 1 node.
%                 derv1=pointDerivative(ray,graph,ll,lengths);
                derv2=pointDerivative2(ray,graph,ll);

                derv=derv2*img;
                update=update+derv*residual(kk,1,jj);
%                 graph.nodes(ll).positions= graph.nodes(ll).positions+lambda*update'./norm(update);
                %                 graph.nodes(ll).positions
                
                %                 return
                %                 residual(kk,ii)
                %                 residual(:,jj)=meshForward(img,geo,angles(jj),graph)-proj(:,jj);
                %                 cla
                %                 imshow(residual,[]);colorbar;
            end
            
        end
    end
    if norm(update)>1
        update=update'./(norm(update)+eps);
    end
    updates(ii,:)=update;
    graph.nodes(ll).positions= graph.nodes(ll).positions-0.5*update;
    pos(ii,:)=graph.nodes(ll).positions;
%     img=SART(proj,geo,angles,graph,10,'verbose',false,'InitImg',img);
    update=[0;0];
end

end