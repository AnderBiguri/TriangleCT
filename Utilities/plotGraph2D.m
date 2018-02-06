function plotGraph2D(graph,TRIvals,varargin)

[linestyle,cmap]=parse_inputs(varargin{:});
cmap=colormap(cmap);
if size(TRIvals,1)==1 && size(TRIvals,2)>1 
    TRIvals=TRIvals.';
end
figure('units','normalized','outerposition',[0 0 1 1])
if ~isempty(TRIvals)
    set(gcf,'color',cmap(1,:))
    
    colormap(cmap);
    if length(TRIvals)==length(graph.elements)
        fcolor='flat';
    else
        fcolor='interp';
    end
    patch('faces',cell2mat({graph.elements.nodeId}.'),'vertices',cell2mat({graph.nodes.positions}.'),'FaceVertexCData',TRIvals,'FaceColor' , fcolor,'linestyle',linestyle)
    c=colorbar;
    c.Color=[1 1 1];
else
    nodes=cell2mat({graph.nodes(:).positions}.');
    triplot(cell2mat({graph.elements.nodeId}.'),nodes(:,1),nodes(:,2),'color','black');
end
% caxis([0 1])
axis equal
axis off
end
function [linestyle,cmap]=parse_inputs(varargin)
% create input parser
p=inputParser;
% add optional parameters
addParameter(p,'linestyle','none',@ischar);
validationFcn=@(x)assert(ischar(x)||(size(x,2)==3));
addParameter(p,'colormap','gray',validationFcn );

%execute
parse(p,varargin{:});
%extract
linestyle=p.Results.linestyle;
cmap=p.Results.colormap;

end