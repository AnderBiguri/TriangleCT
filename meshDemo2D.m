clear;clc;
close all;

%% Geometry definition
% VARIABLE                                   DESCRIPTION                    UNITS
%-------------------------------------------------------------------------------------
% Distances
geo.DSD = 1536;                             % Distance Source Detector      (mm)
geo.DSO = 1000;                             % Distance Source Origin        (mm)
% Detector parameters
geo.nDetector=[512; 1];					% number of pixels              (px)
geo.dDetector=[0.8; 0.8]; 					% size of each pixel            (mm)
geo.sDetector=geo.nDetector.*geo.dDetector; % total size of the detector    (mm)
% Image parameters
geo.nVoxel=[200;200;1];                   % number of voxels              (vx)
geo.sVoxel=[256;256;1];                   % total size of the image       (mm)
geo.dVoxel=geo.sVoxel./geo.nVoxel;          % size of each voxel            (mm)
% Offsets
geo.offOrigin =[0;0;0];                     % Offset of image from origin   (mm)
geo.offDetector=[0; 0];                     % Offset of Detector            (mm)
% These two can be also defined
% per angle

% Auxiliary
geo.accuracy=0.5;                           % Variable to define accuracy of
% 'interpolated' projection
% It defines the amoutn of
% samples per voxel.
% Recommended <=0.5             (vx/sample)

% Optional Parameters
% There is no need to define these unless you actually need them in your
% reconstruction


geo.COR=0;                                  % y direction displacement for
% centre of rotation
% correction                   (mm)
% This can also be defined per
% angle

geo.rotDetector=[0;0;0];                    % Rotation of the detector, by
% X,Y and Z axis respectively. (rad)
% This can also be defined per
% angle

% geo.mode='cone';                            % Or 'parallel'. Geometry type.

%% define projection angles
nangles=25;

angles=linspace(0,2*pi-2*pi/nangles,nangles);
%% Create sample data

P = phantom('Modified Shepp-Logan',200);
P(P<0)=0;

[graph,trivals,nodevals]=voxel2mesh(P,geo,'minPoissonDistance',1);
plotGraph2D(graph,trivals);
plotGraph2D(graph,nodevals);

% %%
tic
proj=meshForward(trivals,geo,angles,graph);
toc
tic
proj=meshBack(proj,geo,angles,graph);
toc
%%
[graphtest]=voxel2mesh(ones(size(P)),geo,'minPoissonDistance',1);

res=SART(proj,geo,angles,graphtest,10);

