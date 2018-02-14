function [nodevals,grad]=meshInterpolateNodes(graph,trivals)

graph=fixgraph(graph);

if nargin ==2
   grad=zeros(size(graph.nodes,2),nD); 
end
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
    gradi=trivals(graph.nodes(ii).neighbour_elems);
    if strcmpi(method,'Sobel')
        aux=lscov(diri,gradi,1./dsq);
    elseif strcmpi(method,'Prewitt')
        aux=diri\gradi;
    end
    if nargin ==2
        grad(ii,:)=aux(1:nD);
    end
    nodevals(ii)=aux(end);
end

end