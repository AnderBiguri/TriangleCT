clear;clc;
InitMesh;

%% Load data
% no inner surfaces, purely triangles, 3.5M
% load('.\Models\standford_block_3p5m.mat')
% TRI=double(Model4_FE_tetrahedra);
% vertices=Model4_FE_vertices;
% values=Model4_FE_tetrahedra_Part;
% Inner surfaces
load('.\Models\standford_teapot.mat')
TRI=double(Model2_FE_tetrahedra);
vertices=Model2_FE_vertices;
values=Model2_FE_tetrahedra_Part;

clearvars -except TRI values vertices

%% plot if desired
figure
% close all
cla
hold on
axis equal
% axis([-100 100 -20 20 -70 70])
cmap=hsv(length(unique(values)));
alpha=linspace(0.2,1,length(unique(values)));
for ii=1:length(unique(values))

[F,P]=freeBoundary( triangulation(double(TRI(double(values)>(ii-0.5),:)),double(vertices)));
trisurf(F,P(:,1),P(:,2),P(:,3), ...
       'FaceColor',cmap(ii,:),'FaceAlpha',alpha(ii));
camlight

end
%% Create graph

m=min(vertices);
M=max(vertices);
d=M+m;
vertices=vertices-d/2;

scale=1.;
vertices=vertices*scale;
tic
graph=trimesh2graph(TRI,vertices);
toc


%% Geometry

% Geometry definition
% VARIABLE                                   DESCRIPTION                    UNITS
%-------------------------------------------------------------------------------------
% Distances
geo.DSD = 1536;                             % Distance Source Detector      (mm)
geo.DSO = 1000;                             % Distance Source Origin        (mm)
% Detector parameters
geo.nDetector=[512; 512];					% number of pixels              (px)
geo.dDetector=[0.35; 0.35]; 		     	% size of each pixel            (mm)
geo.sDetector=geo.nDetector.*geo.dDetector; % total size of the detector    (mm)
% Image parameters
geo.nVoxel=[512,512,512]';                     % number of voxels              (vx)
geo.sVoxel=[180,180,180]';                     % total size of the image       (mm)
geo.dVoxel=geo.sVoxel./geo.nVoxel;          % size of each voxel            (mm)
% Offsets
geo.offOrigin =[0;0;0];                     % Offset of image from origin   (mm)
geo.offDetector=[0; 0];                     % Offset of Detector            (mm)
geo.accuracy=0.5;                           % Variable to define accuracy of
geo.COR=0;                                  % y direction displacement for
geo.rotDetector=[0;0;0];    

%% Generate projections

angles=linspace(0,2*pi-2*pi/100,100);
%  tic
%  proj=projection(single(values-1),geo,angles,graph);
%  toc
load('bunny_proj.mat','proj')
% return;
 %%

ossart_rec=OS_SART(proj,geo,angles,graph,50);
% save paper_bunny_fig_block.mat