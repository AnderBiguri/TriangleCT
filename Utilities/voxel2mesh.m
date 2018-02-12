function [ varargout ] = voxel2mesh(img, geo, varargin)
%VOXEL2MESH Creates a mesh from input image
%   Creates a mesh from the inpu timage. Uses edges of the image and
%   poisson disk sampling distributed points to populate the vertices and
%   Delaunay triangulation for mesh creation.
%   optional parameters:
%       'bounded': Logical value. If true, it doesnt generate triangular
%                  mesh outside the outermost boundary of the given image.
%                  Default is false.
%       'minPoissonDistance': Minimum distance for generating a point cloud
%                             to populate the mesh default is size/10;
%       'connectedGraph': Logical value. Return a graph or triangular/point
%                         pair. Default is true (i.e. graph)

[isbounded,min_poisson_dist,isGraphOut]=parse_inputs(geo,varargin{:});

if size(img,3)==1; nD=2; else; nD=3; end


if nD==2
    % Edge index location
    edges=edge(img,'Canny');
    [x,y]= find(edges==1);
    
    % Posson disk sampling to populate the area.
    extrapoints=poissonDisc(geo.nVoxel(1:2).',min_poisson_dist);
    
    if isbounded
        % Define "in" and "out"
        in=imfill(edges,'holes');
        [incoordsx,incoordsy]=find(in==1);
        
        % make sure they are inside
        [~,ia,~] = intersect(round(extrapoints),[incoordsx(:),incoordsy(:)],'rows');
        extrapoints=extrapoints(ia);
    end
    
    % Add newly populated data to edges.
    x=[x;extrapoints(:,1)];
    y=[y;extrapoints(:,2)];
    
    % triangulate
    TRI=delaunay(x,y);
    
    % if we want the values on the elements in the original mesh, compute
    % them
    trivals=zeros(size(TRI,1),1);
    if  (isGraphOut && nargout>1) || (~isGraphOut && nargout>2)
        for ii=1:size(TRI,1)
            trivals(ii)=interp2(img,mean(y(TRI(ii,:))),mean(x(TRI(ii,:))),'linear',0);
        end
        varargout{3-isGraphOut}=trivals;            % trick with casting. outGrah is 0 or 1;  
    end
    % Same thing, for node values
     nodevals=zeros(size(x,1),1);
    if  (isGraphOut && nargout>2) || (~isGraphOut && nargout>3)
        for ii=1:size(x,1)
            nodevals(ii)=interp2(img,y(ii),x(ii),'linear',0);
        end
        varargout{4-isGraphOut}=nodevals;            % trick with casting. isGraphOut is 0 or 1;  
    end
    % Give them real world coords.
    x=x-geo.nVoxel(1)/2-0.5;
    y=y-geo.nVoxel(2)/2-0.5;
    x=x*geo.dVoxel(1);
    y=y*geo.dVoxel(2);
    
    if isGraphOut
        varargout{1}=trimesh2graph(TRI,[x(:),y(:)]);
    else
        varargout{1}=TRI;
        varargout{2}=[x(:),y(:)];
    end
    
    
end
if nD==3
    error('Not yet implemented');
end

end

function [isbounded,min_poisson_distance,isGraphOut]=parse_inputs(geo,varargin)

% create input parser
p=inputParser;
% add optional parameters
addParameter(p,'bounded',false);
addParameter(p,'minPoissonDistance',max(geo.sVoxel(1:2))/100,@isscalar);
addParameter(p,'connectedGraph',true,@islogical)
%execute
parse(p,varargin{:});
%extract
isbounded=p.Results.bounded;
min_poisson_distance=p.Results.minPoissonDistance;
isGraphOut=p.Results.connectedGraph;
end
