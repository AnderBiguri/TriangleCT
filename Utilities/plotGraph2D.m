function plotGraph2D(graph,TRIvals,varargin)

[linestyle,cmap,clims]=parse_inputs(TRIvals,varargin{:});

if size(TRIvals,1)==1 && size(TRIvals,2)>1 
    TRIvals=TRIvals.';
end
figure('units','normalized','outerposition',[0 0 1 1])
if ~isempty(TRIvals)
    cmap=colormap(cmap); % cmap can be a string, convert to numerical
    set(gcf,'color',cmap(1,:))
    colormap(cmap);
    if length(TRIvals)==length(graph.elements)
        fcolor='flat';
    else
        fcolor='interp';
    end
    patch('faces',cell2mat({graph.elements.nodeId}.'),'vertices',cell2mat({graph.nodes.positions}.'),'FaceVertexCData',TRIvals,'FaceColor' , fcolor,'linestyle',linestyle)
    c=colorbar;
    c.Color=imcomplement(cmap(1,:));
    
    caxis(clims)

else
    nodes=cell2mat({graph.nodes(:).positions}.');
    triplot(cell2mat({graph.elements.nodeId}.'),nodes(:,1),nodes(:,2),'color','black');
end
axis equal
axis off

set(gcf, 'InvertHardCopy', 'off');
end
function [linestyle,cmap,clims]=parse_inputs(TRIvals,varargin)
% create input parser
p=inputParser;
% add optional parameters
addParameter(p,'linestyle','none',@ischar);

validationFcn=@(x)isa(x,'double');
addParameter(p,'clims',[min([TRIvals(:); 0]) max(TRIvals(:))],validationFcn);

validationFcn=@(x)assert(ischar(x)||(size(x,2)==3));
addParameter(p,'colormap','gray',validationFcn );

%execute
parse(p,varargin{:});
%extract
linestyle=p.Results.linestyle;
cmap=p.Results.colormap;
clims=p.Results.clims;
end