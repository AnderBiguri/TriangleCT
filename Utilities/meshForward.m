function proj=meshForward(TRIvals,geo,angles,graph)
%MESHFORWARD forward projects the X-ray data of a triangular graph geometry.
%  This fucntion forward projects X-ray images. MESHGRAPH has to be a
%  connected graph, describing a triangular FEM-type mesh
%  MESHFORWARD(TRIVALS,GEO,ANGLES,MESHGRAPH)
%              TRIVALS: numerical values of the mesh elements
%              GEO: TIGRE style geometry structure
%              ANGLES: array of angles that is desired to take the
%              projections on.
%              MESHGRAPH: a connected graph describing a FEM-type
%              triangular mesh
%
%  See also TRIMESH2GRAPH, MESHBACK, VOXEL2MESH


[TRI,nodes]=graph2trimesh(graph);TRI=double(TRI);nodes=double(nodes);
% Grab all nodes positions for speed
% nodes=csl2mat(graph.nodes(:).positions);

vmin=[min(nodes(:,1)) min(nodes(:,2)) min(nodes(:,3))];
vmax=[max(nodes(:,1)) max(nodes(:,2)) max(nodes(:,3))];

% preallocate projection data
proj=zeros(geo.nDetector(2),geo.nDetector(1),length(angles));
nangles=length(angles);
% Define Source and detector world locations, and rotation to be applied
us = ((-geo.nDetector(1)/2+0.5):1:(geo.nDetector(1)/2-0.5))*geo.dDetector(1) + geo.offDetector(1);
vs = ((-geo.nDetector(2)/2+0.5):1:(geo.nDetector(2)/2-0.5))*geo.dDetector(2) + geo.offDetector(2);
% detector=[-(geo.DSD-geo.DSO)*ones(geo.nDetector(1),1),(-geo.sDetector(1)/2+geo.dDetector(1)/2:geo.dDetector(1):geo.sDetector(1)/2-geo.dDetector(1)/2)'];
source=[+geo.DSO(1),0,0];
R =@(theta)( [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]);
% Loop every projection


tinit=0;
tcore=0;

for k=1:nangles
    % for every pixell in the detector
%     for ii=1:geo.nDetector(2)
%         for jj=1:geo.nDetector(1)
                for ii=256
                    for jj=256
            
            
            ray=[source; -(geo.DSD(1)-geo.DSO(1)), us(jj), vs(ii)];
            ray=(R(angles(k))*ray.').';
            %             plot3(ray(:,1),ray(:,2),ray(:,3),'r')
            vecline=single(ray(2,:))-single(ray(1,:));
            [flag,tmin,tmax]=rayBoxIntersection(ray(1,:),vecline,vmin,vmax);
            if flag==0
                continue;
            end
            
            % TODO: fix following
            % WARNING: for now lets assume mesh in convex.
%             initInter=1.1;
%             % find an intersection with the boundary.
%             
%             tic
            epsilon=1e-6;
%             notintersect=length(graph.boundary_elems);
%             while notintersect==length(graph.boundary_elems)
%                 notintersect=0;
%                 for bb=1:length(graph.boundary_elems)
%                     [t]=lineTriangleIntersectLength2_safe(ray,nodes(graph.elements(graph.boundary_elems(bb)).nodeId,:),epsilon);
%                     if sum(t)==0
%                         notintersect=notintersect+1;
%                     else
%                         % interesction!
%                         t1=min(t(t>0));
%                         if initInter>t1
%                             initInter=t1;
%                             initInterind=bb;
%                         end
%                     end
%                 end
%                 epsilon=epsilon*10;
%             end
%             epsilon=epsilon/10;
%             tinit=tinit+toc;
            tic
            [initInterind,t]=R_tree_search(graph.tree,ray,graph);
            tinit=toc;         

            tic;
            % if we looked at the whole boudnary and no intersection was found,
            % then the ray is outside the mesh. OUT!
            
            % now start line search from indes bb.
            % lets get the intersection of that boundary element
            [t]=lineTriangleIntersectLength2_safe(ray,nodes(graph.elements(graph.boundary_elems(initInterind)).nodeId,:),epsilon);
             
            % assuming 2 intersections
            %             if nnz(t)~=2
            %                 keyboard
            %             end
            [t2,indt]=max(t);
            t1=min(t(t>0));
       
            %             disp(['Elem: ',num2str(graph.boundary_elems(initInterind)),' t1: ',num2str(t1),' t2: ',num2str(t2)]);
            %             fprintf("%.16f %.16f %.16f\n",ray(1),ray(3),ray(5));
            %             fprintf("%.16f %.16f %.16f\n",ray(2),ray(4),ray(6));
            %             fprintf("%.16f %.16f\n",t1,t2);
            d=(t2-t1)*norm(vecline);
            %             intersectPoints(1,:)=t1*vecline; %+line(1,:)
            %             intersectPoints(2,:)=t2*vecline; %+line(1,:)
            %             d=sqrt(sum((intersectPoints(2,:)-intersectPoints(1,:)) .^2) );
            %             fprintf("%u %.16f %.16f\n",uint32(graph.boundary_elems(initInterind)),t1,t2);
            
            proj(ii,jj,k)=proj(ii,jj,k)+d*TRIvals(graph.boundary_elems(initInterind));
            
            %             neighbours=graph.elements(graph.boundary_elems(initInterind)).neighbours;
            if t1==t2
                if graph.elements(graph.boundary_elems(initInterind)).neighbours(indt)==0
                    indts=find(t==t1);
                    indt=indts(indts~=indt);
                end
            end
            current_tri=graph.elements(graph.boundary_elems(initInterind)).neighbours(indt);
            if current_tri==0
                continue;
            end
            
            
            %
            
            prev_tri=graph.boundary_elems(initInterind);
            notneighb=false;
            safe=0;
            startplot=0;
            sumt=single(0);
            while (~notneighb&&safe<=size(graph.elements,2))
                safe=safe+1;
                if(safe==size(graph.elements,2))
                    warning(['Safe distance past:', num2str(ii),' ',  num2str(jj)]);
                end
                t2prev=t2;
                t1prev=t1;
                
                [t]=lineTriangleIntersectLength2_safe((ray),(nodes(graph.elements(current_tri).nodeId,:)),0);
                if nnz(t)<=1
                    epsilon=1e-5;
                    while nnz(t)<=1
                        disp(['Safety measure triggered on RAY:',num2str(ii),' ',num2str(jj) ,' Elem: ',num2str(current_tri)]);
                        [t]=lineTriangleIntersectLength2_safe(ray,nodes(graph.elements(current_tri).nodeId,:),epsilon);
                        epsilon=epsilon*10;
                    end
                    %                     disp(['Single intersection found on RAY:',num2str(ii),' ',num2str(jj) ,' TRI:',num2str(current_tri)]);
                end
                %                 t=round(t*1000000)/1000000;
                [t2,indt]=max(t);
                [t1]=min(t+2*(t==0));
                
                %                   fprintf("%u %.16f %.16f\n",uint32(current_tri),t1,t2);
                if abs(t2-t1)<1e-8
                    t2=t1;
                    t(indt)=t1;
                end
                if abs(t2prev-t1)>1e-5
                    disp(['Fishy back-step on RAY:',num2str(ii),' ',num2str(jj) ,' Elem: ',num2str(current_tri)])
                    disp(t)
                    break;
                end
                sumt=sumt+(t2-t1);
                %                 intersectPoints(1,:)=t1*vecline; %+line(1,:)
                %                 intersectPoints(2,:)=t2*vecline; %+line(1,:)
                if sum(t)~=0
                    %                     disp(['Elem: ',num2str(current_tri),' t1: ',num2str(t1,8),' t2: ',num2str(t2,8)]);
                    %                     disp(t)
                    d=(t2-t1)*norm(vecline);
                    %                     d=sqrt(sum((intersectPoints(2,:)-intersectPoints(1,:)) .^2) );
                    %                     if current_tri==3054
                    %                         startplot=1;
                    %                         hold on
                    %                         line(ray(:,1),ray(:,2),ray(:,3),'Color','r');axis([40    82   -10     8   -90     0]);
                    %                         view(-7,12);
                    %                     end
                    %                     if startplot&& d~=0
                    %
                    %                         tetramesh(triangulation(TRI(current_tri,:),nodes(:,1),nodes(:,2),nodes(:,3)),'facealpha',0.1,'FaceColor','g','edgecolor','k');ax=axis;
                    %                         axis square
                    %
                    %                         plot3([intersectPoints(1,1)+ray(1,1) intersectPoints(2,1)+ray(1,1)],...
                    %                             [intersectPoints(1,2)+ray(1,2) intersectPoints(2,2)+ray(1,2)],...
                    %                             [intersectPoints(1,3)+ray(1,3) intersectPoints(2,3)+ray(1,3)],'b','linewidth',2);
                    % %
                    % %                         pause();
                    %
                    %                     end
                    
                    
                    proj(ii,jj,k)=proj(ii,jj,k)+d*TRIvals(current_tri);
                    if t1==t2
                        if graph.elements(current_tri).neighbours(indt)==prev_tri
                            indts=find(t==t1);
                            indt=indts(indts~=indt);
                        end
                    end
                    
                    prev_tri=current_tri;
                    current_tri= graph.elements(current_tri).neighbours(indt);
                    %                     disp(current_tri)
                    if current_tri==0
                        if abs(t2-tmax)>=1e-4
                            disp(['Premature Zero neighbour found on RAY:',num2str(ii),' ',num2str(jj) ,' TRI:',num2str(prev_tri)]);
                        end
                        break;
                    end
                    tcore=tcore+toc;
                    continue;
                end
                notneighb=true;
                disp(['Maximum safe iterations RAY:',num2str(ii),' ',num2str(jj) ,' safe:',num2str(safe)]);
                
                
            end
            if safe>size(graph.elements,2)
                proj(ii,jj,k)=-1;
            end
        end
    end
end

disp(tinit)
disp(tcore)

end