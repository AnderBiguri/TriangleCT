clear;clc;
InitMesh
close all
%% Load data
fv=stlread('./Models/BLOCK1.stl');
fv.vertices(:,1)=fv.vertices(:,1)+[min(fv.vertices(:,1))-max(fv.vertices(:,1))]/2;
fv.vertices(:,2)=fv.vertices(:,2)+[min(fv.vertices(:,2))-max(fv.vertices(:,2))]/2;
fv.vertices(:,3)=fv.vertices(:,3)+[min(fv.vertices(:,3))-max(fv.vertices(:,3))]/2;

%% plot it
vis=1;
if vis
patch(fv,'FaceColor',       [0.8 0.8 1.0], ...
         'FaceAlpha',       1,           ...
         'EdgeColor',       'k',        ...
         'FaceLighting',    'gouraud',     ...
         'AmbientStrength', 0.15);

% Add a camera light, and tone down the specular highlighting
camlight;
material('dull');

% Fix the axes scaling, and set a nice view angle
axis('image');
view([-135 35]);
end
% return
fv=fixSTL(fv);
%%  Lets make a 3D cuboid out of it. 

% Range of the box
xr=[min(fv.vertices(:,1)) max(fv.vertices(:,1))];% xr=xr-mean(xr);
yr=[min(fv.vertices(:,2)) max(fv.vertices(:,2))];% yr=yr-mean(yr);
zr=[min(fv.vertices(:,3)) max(fv.vertices(:,3))];% zr=zr-mean(zr);

% Lets give it a bit of surrounding
extra=0.1;
xr=xr+[-extra*diff(xr) +extra*diff(xr)];
yr=yr+[-extra*diff(yr) +extra*diff(yr)];
zr=zr+[-extra*diff(zr) +extra*diff(zr)];

% The points we want to triangulate are the surface points + 8 corners, for
% now. 
vertices=fv.vertices;
pts=[];
pts= populateTriangle(fv,2,false,true);



vertices=[vertices;
          pts;
          xr(1) yr(1) zr(1);
          xr(1) yr(1) zr(2);
          xr(1) yr(2) zr(1);
          xr(1) yr(2) zr(2);
          xr(2) yr(1) zr(1);
          xr(2) yr(1) zr(2);
          xr(2) yr(2) zr(1);
          xr(2) yr(2) zr(2)];
      
vertices=unique(single(vertices),'rows');
vertices=double(vertices);
%% We need more. Lets populate the surfaces with points

%% plot points
%
% hold on
% plot3(vertices(:,1),vertices(:,2),vertices(:,3),'r.')
%       return
%% Triangulate this      
% center the mesh

TRI=delaunay(vertices(:,1),vertices(:,2),vertices(:,3));
tic
graph=trimesh2graph(TRI,vertices);
toc
% Lets give it numerical values. For that we need to knwo which triangles ones are
% inside and which outside. Lets get their centroids

for ii=1:size(graph.elements,2)
    centroid(ii,:)=mean(csl2mat(graph.nodes(graph.elements(ii).nodeId).positions));
end
% trivals=inpolyhedron(fv,centroid);
trivals=inmesh(fv,centroid,1);
%% Lets plot this
close all
cla
hold on
axis equal
% axis([-100 100 -20 20 -70 70])

[F,P]=freeBoundary( triangulation(TRI(trivals>0,:),vertices));
trisurf(F,P(:,1),P(:,2),P(:,3), ...
       'FaceColor',[0.8 0.8 1.0],'FaceAlpha',1,'edgecolor','none');
camlight 
%% Lets make a geometry
% Geometry definition
% VARIABLE                                   DESCRIPTION                    UNITS
%-------------------------------------------------------------------------------------
% Distances
geo.DSD = 1536;                             % Distance Source Detector      (mm)
geo.DSO = 1000;                             % Distance Source Origin        (mm)
% Detector parameters
geo.nDetector=[256; 256];					% number of pixels              (px)
geo.dDetector=[1; 1]; 					% size of each pixel            (mm)
geo.sDetector=geo.nDetector.*geo.dDetector; % total size of the detector    (mm)
% Image parameters
geo.nVoxel=[256;256;1];                     % number of voxels              (vx)
geo.sVoxel=[diff(xr);diff(yr);diff(zr)];                     % total size of the image       (mm)
geo.dVoxel=geo.sVoxel./geo.nVoxel;          % size of each voxel            (mm)
% Offsets
geo.offOrigin =[0;0;0];                     % Offset of image from origin   (mm)
geo.offDetector=[0; 0];                     % Offset of Detector            (mm)
geo.accuracy=0.5;                           % Variable to define accuracy of
geo.COR=0;                                  % y direction displacement for
geo.rotDetector=[0;0;0];                    % Rotation of the detector, by

%%
% Temopraryhack
trivals(112016)=0;
trivals(111736)=0;
trivals(125964)=0;
trivals(126288)=0;

%% Test projection

angles=[pi/3];
trivals1=ones(size(trivals));
angles=linspace(0,2*pi,100);
proj=projection(single(trivals),geo,angles,graph);

return
tic
test=meshForward(trivals,geo,angles,graph);
toc
figure
imshow(test(:,:,1).',[])

%%
figure
% close all
cla
hold on
axis equal
% axis([-100 100 -20 20 -70 70])


[F,P]=freeBoundary( triangulation(TRI(separatedSIRT>0.7,:),vertices));
trisurf(F,P(:,1),P(:,2),P(:,3), ...
       'FaceColor',[0.8 0.8 1.0],'FaceAlpha',1,'edgecolor','none');
camlight