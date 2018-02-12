% Test Validity of gradients
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
geo.nVoxel=[256;256;1];                   % number of voxels              (vx)
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

P = phantom('Modified Shepp-Logan',geo.nVoxel(1));
P(P<0)=0;

[graph,trivals]=voxel2mesh(P,geo,'minPoissonDistance',1);
% plotGraph2D(graph,trivals);
% plotGraph2D(graph,nodevals);

%% Test plane
[x,y]=meshgrid(1:geo.nVoxel(1),1:geo.nVoxel(2));
x=x-geo.nVoxel(1)/2-0.5;
y=y-geo.nVoxel(2)/2-0.5;
x=x*geo.dVoxel(1);
y=y*geo.dVoxel(2);


nodes=csl2mat(graph.nodes(:).positions);



a=1;
b=5;
x0=nodes(1000,2);
y0=nodes(1000,1);

testpoints=[[-1 -1 0 1 1];[-1 1 0 -1 1]].';
[TRI]=delaunay([-1 -1 0 1 1],[-1 1 0 -1 1]);
testmesh=trimesh2graph(TRI,testpoints);

plane=@(y,x)(a*x+b*y);
parabola=@(y,x)((x-x0).^2+(y-y0).^2);
% z=parabola(x,y);
zmesh=parabola(nodes(:,1),nodes(:,2));
% 
g_real=[(nodes(:,1)-y0)*2 (nodes(:,2)-x0)*2];
% zmesh=plane(graph.nodes(:,1),graph.nodes(:,2));
% ztest=plane(testmesh.nodes(:,1),testmesh.nodes(:,2));

% g_piotr=meshgradient1_2(graph,zmesh);
% g_piotr_no_weigth=meshgradient1_2(graph,zmesh);
g_flawr=meshgradient(graph,trivals);
g_flawr_nobound=g_flawr;
boundary_nodes=unique(cell2mat({graph.elements(graph.boundary_elems).nodeId}));
g_flawr_nobound(boundary_nodes,:)=repmat([0 0],[length(boundary_nodes),1]);
return;

%%
subplot(121);hold on; title('Modulus of gradient');
histogram((sqrt(sum(g_piotr.^2,2))),100);histogram((sqrt(sum(g_flawr.^2,2))),100);
yL = get(gca,'YLim');
line([sqrt(sum([a b].^2,2)) sqrt(sum([a b].^2,2))],yL,'Color','r');
legend({'@Piotr','@flawr','Real value'})

% legend({'@Piotr Benedysiuk','@flawr'})
subplot(122);hold on; title('Angle of gradient');
histogram(atan2d(g_piotr(:,2),g_piotr(:,1)),100);histogram(atan2d(g_flawr(:,2),g_flawr(:,1)),100)
yL = get(gca,'YLim');
line([atan2d(a,b) atan2d(a,b)],yL,'Color','r');

%%
close all
plotGraph2D(graph,sqrt(sum(g_piotr.^2,2)));caxis([0 600])
plotGraph2D(graph,sqrt(sum(g_flawr_nobound.^2,2)));caxis([0 600])
%%
close all
plotGraph2D(graph,atan2d(g_piotr(:,2),g_piotr(:,1)));
plotGraph2D(graph,atan2d(g_flawr(:,2),g_flawr(:,1)));
%%
close all
figure()
trisurf(cell2mat({graph.elements.nodeId}.'),graph.nodes(:,1),graph.nodes(:,2),sqrt(sum(g_piotr.^2,2)).','linestyle','none');xlabel('x');ylabel('y')
% figure()
% trisurf(cell2mat({graph.elements.nodeId}.'),graph.nodes(:,1),graph.nodes(:,2),sqrt(sum(g_piotr_no_weigth.^2,2)).','linestyle','none');xlabel('x');ylabel('y')
figure()
trisurf(cell2mat({graph.elements.nodeId}.'),graph.nodes(:,1),graph.nodes(:,2),sqrt(sum(g_flawr.^2,2)).','linestyle','none');xlabel('x');ylabel('y')

%%
close all
figure
 quiver(nodes(:,1),nodes(:,2),g_piotr(:,1),g_piotr(:,2));
%  figure
%  quiver(graph.nodes(:,1),graph.nodes(:,2),g_piotr_no_weigth(:,1),g_piotr_no_weigth(:,2));
 figure
 quiver(nodes(:,1),nodes(:,2),g_flawr(:,1),g_flawr(:,2));

%% 
return; 
%% Test parabolloid

% g_piotr=meshgradient1_2(graph,zmesh);
% g_piotr_no_weigth=meshgradient1_2(graph,zmesh);
g_flawr=meshgradient(graph,zmesh);
% 
% percntiles1 = prctile(abs(sqrt(sum(g_piotr.^2,2)))-sqrt(sum(g_real.^2,2)),[5 95]);
% idx=find(abs(sqrt(sum(g_piotr.^2,2)))-sqrt(sum(g_real.^2,2))<percntiles1(1) | abs(sqrt(sum(g_piotr.^2,2)))-sqrt(sum(g_real.^2,2))>percntiles1(2)) ;
% g_piotr(idx,:)=[]; 
g_real_piotr=g_real; 
% g_real_piotr(idx,:)=[]; 
% 
% percntiles2 =prctile(abs(sqrt(sum(g_flawr.^2,2)))-sqrt(sum(g_real.^2,2)),[5 95]);
% idx=find(abs(sqrt(sum(g_flawr.^2,2)))-sqrt(sum(g_real.^2,2))<percntiles2(1) | abs(sqrt(sum(g_flawr.^2,2)))-sqrt(sum(g_real.^2,2))>percntiles2(2)) ;
% g_flawr(idx,:)=[]; 
g_real_flawr=g_real; 
% g_real_flawr( idx,:)=[]; 
%% plot

subplot(121);
hold on; title('Error in magnitude of gradient');
histogram(abs(sqrt(sum(g_piotr.^2,2))-sqrt(sum(g_real_piotr.^2,2))),100);
histogram(abs(sqrt(sum(g_flawr.^2,2))-sqrt(sum(g_real_flawr.^2,2))),100);
yL = get(gca,'YLim');

 legend({'Method 1-2 ','Method 2'})
subplot(122);hold on; title('Error in Angle of gradient');
histogram(abs(atan2d(g_piotr(:,2),g_piotr(:,1))-atan2d(g_real_piotr(:,2),g_real_piotr(:,1))),100);
histogram(abs(atan2d(g_flawr(:,2),g_flawr(:,1))-atan2d(g_real_flawr(:,2),g_real_flawr(:,1))),100);xlim([0 20])
yL = get(gca,'YLim');
% line([atan2d(a,b) atan2d(a,b)],yL,'Color','r');
%% TEST shepp logan



gimage=imgradient(P,'central');



for ii=1:size(graph.nodes,1)
    ginterp(ii)=interp2(x,y,gimage,graph.nodes(ii,2),graph.nodes(ii,1),'linear',0);
end
g_flawr=meshgradient2(graph,nodevals);

%%

for ii=1:size(g_flawr,1)
angles(ii)=atan2d(norm(cross([g_flawr(ii,:) 0],[0 1 0])),dot([g_flawr(ii,:) 0],[0 1 0]));
end
modulus=sqrt(sum(g_flawr.^2,2));

angles(modulus<0.2)=0;
% plotGraph2D(graph,sqrt(sum(g_flawr.^2,2)));
plotGraph2D(graph,angles);

%  quiver(graph.nodes(:,1),graph.nodes(:,2),g_flawr(:,1),g_flawr(:,2));
% 
% figure;
% imagesc(gimage.');axis equal;axis off;axis ij; colormap(parula); colorbar
% plotGraph2D(graph,ginterp.');