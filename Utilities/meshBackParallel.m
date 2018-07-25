function TRIvals=meshBackParallel(proj,geo,angles,graph)
%MESHFORWARD backprojects the X-ray data of a triangular graph geometry.
%  This function backprojects X-ray images. MESHGRAPH has to be a
%  connected graph, describing a triangular FEM-type mesh
%  MESHBACK(TRIVALS,GEO,ANGLES,MESHGRAPH)
%              TRIVALS: numerical values of the mesh elements
%              GEO: TIGRE style geometry structure
%              ANGLES: array of angles that is desired to take the
%              projections on.
%              MESHGRAPH: a connected graph describing a FEM-type
%              triangular mesh
%
%  See also TRIMESH2GRAPH, MESHFORWARD, VOXEL2MESH

if length(graph.elements(1).nodeId)==4
    error('3D Mesh not yet supported')
end

% Grab all nodes positions for speed
nodes=csl2mat(graph.nodes(:).positions);

% preallocate projection data
TRIvals=zeros(size(graph.elements,2),1);
nangles=length(angles);

% Define Source and detector world locations, and rotation to be applied
us = ((-geo.nDetector(1)/2+0.5):1:(geo.nDetector(1)/2-0.5))*geo.dDetector(1) + geo.offDetector(1);
vs = ((-geo.nDetector(2)/2+0.5):1:(geo.nDetector(2)/2-0.5))*geo.dDetector(2) + geo.offDetector(2);
% detector=[-(geo.DSD-geo.DSO)*ones(geo.nDetector(1),1),(-geo.sDetector(1)/2+geo.dDetector(1)/2:geo.dDetector(1):geo.sDetector(1)/2-geo.dDetector(1)/2)'];
source=[+geo.DSO,0,0];
R =@(theta)( [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]);
% Loop every projection
for k=1:nangles
    % for every pixell in the detector
    for ii=1:geo.nDetector(1)
        for jj=1:geo.nDetector(2)
            
            ray=[source; -(geo.DSD-geo.DSO), us(ii), vs(jj)];
            ray=(R(angles(k))*ray.').';

            
            
            % TODO: fix following
            % WARNING: for know lets assume mesh in convex.
            
            % find an intersection with the boundary.
            notintersect=0;
            for bb=1:length(graph.boundary_elems)
                if ~isLineTriangleIntersect(ray,nodes(graph.elements(graph.boundary_elems(bb)).nodeId,:))
                    notintersect=notintersect+1; % we got an intersection!
                else
                    break
                end
            end
            % if we looked at the whole boudnary and no intersection was found,
            % then the ray is outside the mesh. OUT!
            if notintersect==length(graph.boundary_elems)
                continue;
            end
            
            
            % now start line search from indes bb.
            % lets get the intersection of that boundary element
            conn_list=graph.elements(graph.boundary_elems(bb)).neighbours;
            d=lineTriangleIntersectLength(ray,nodes(graph.elements(graph.boundary_elems(bb)).nodeId,:));
            TRIvals(graph.boundary_elems(bb))=TRIvals(graph.boundary_elems(bb))+d*proj(ii,jj,k);
            % keep track of deleted elemets
            deleted=[];
            while ~isempty(conn_list)
                % pop next element
                current_tri=conn_list(1);
                conn_list(1)=[];
                deleted=sort([deleted current_tri]);
                
                % does it intersect?
                d=lineTriangleIntersectLength(ray,nodes(graph.elements(current_tri).nodeId,:));
                
                % id it doesnt, skip maths, go to the next element.
                if d==-1
                    continue
                end
                
                % compute that intersection.
                TRIvals(current_tri)=TRIvals(current_tri)+d*proj(ii,jj,k);
                
                newtri= graph.elements(current_tri).neighbours;
                
                % Add next neighbours to the list, but make sure that they have
                % not been deleted, nor are they already in the list
                aux=newtri(~ismembc(newtri,deleted));                          % https://stackoverflow.com/questions/8159449/a-faster-way-to-achieve-what-intersect-is-giving-me
                conn_list=[conn_list;aux(~ismembc(aux,conn_list))];            % faster setdiff
                conn_list=sort(conn_list);
            end
        end
    end
end


end