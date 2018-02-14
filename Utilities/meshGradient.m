function g=meshgradient(graph,graph_vals,varargin)
%MESHGRADIENT(..) computes the gradient of a triangular mesh using function
% values defined in the centroid of the mesh by fitting a plane to the
% nodes surrounding elements' centroids.
% https://math.stackexchange.com/a/2632616/44383
% Yer the Chosen One, Harry

[method]=parse_inputs(varargin{:});
graph=fixgraph(graph);

nD=size(graph.elements(1).nodeId,2)-1;
assert(nD==2 || nD==3,'Only 2D or 3D meshes supported');



g=zeros(size(graph.nodes,2),nD);

% This method needs the adjacency of the node-elements

% If fucntion values have been given in elements 
if size(graph_vals,1)==size(graph.elements,2)
    
    
    % We are going to need the centroids of the elements
    centroids=zeros(size(graph.elements,2),nD);
    for ii=1:length(centroids)
        centroids(ii,:)= mean(csl2mat(graph.nodes(graph.elements(ii).nodeId).positions));
    end
    
    
    % Neighbour element based approach
    for ii=1:size(graph.nodes,2)
        diri=ones(length(graph.nodes(ii).neighbour_elems),nD+1);
        dsq=zeros(length(graph.nodes(ii).neighbour_elems),1);
        for jj=1:length(graph.nodes(ii).neighbour_elems)
            diri(jj,1:nD)=centroids(graph.nodes(ii).neighbour_elems(jj),:)-graph.nodes(ii).positions;
            if strcmpi(method,'Sobel')
                dsq(jj)=sum((centroids(graph.nodes(ii).neighbour_elems(jj),:)-graph.nodes(ii).positions).^2,2);
            end
        end
        gradi=graph_vals(graph.nodes(ii).neighbour_elems);
        if strcmpi(method,'Sobel')
            aux=lscov(diri,gradi,1./dsq);
        elseif strcmpi(method,'Prewitt')
            aux=diri\gradi;
        end
        g(ii,:)=aux(1:nD);
    end
    
% If fucntion values have been given in nodes 
elseif size(graph_vals,1)==size(graph.nodes,2)
    
    for ii=1:size(graph.nodes,2)
        diri=ones(length(graph.nodes(ii).neighbour_nodes),nD+1);
        dsq=zeros(length(graph.nodes(ii).neighbour_nodes),1);
        neighbours=csl2mat(graph.nodes(graph.nodes(ii).neighbour_nodes).positions);
        for jj=1:length(graph.nodes(ii).neighbour_nodes)
            
            diri(jj,1:nD)=neighbours(jj,:)-graph.nodes(ii).positions;
            
            if strcmpi(method,'Sobel')
                dsq(jj)=sum((neighbours(jj,:)-graph.nodes(ii).positions).^2,2);
            end
        end
        gradi=graph_vals(graph.nodes(ii).neighbour_nodes);
        if strcmpi(method,'Sobel')
            aux=lscov(diri,gradi,1./dsq);
        elseif strcmpi(method,'Prewitt')
            aux=diri\gradi;
        end
        g(ii,:)=aux(1:nD);
    end
    
end
end
function method=parse_inputs(varargin)

p=inputParser;
% add optional parameters
validationFcn=@(x)( ischar(x)  &&  ( strcmpi(x,'Prewitt') || strcmpi(x,'Sobel')  ));
addParameter(p,'method','Sobel',validationFcn);
parse(p,varargin{:});
%extract
method=p.Results.method;

end