function graph=fixgraph(graph)
% FIXGRAPH fixes the structure of teh input graph, if needed.


% if nodes are not built as single elemetns in a graph
if size(graph.nodes,1)~=1
    nodes=graph.nodes;
    nD=size(nodes,2);
    graph=rmfield(graph,'nodes');
    for ii=1:size(nodes,1)
        graph.nodes(ii).positions=nodes(ii,:);
        graph.nodes(ii).neighbour_elems=[];
    end
    
    for ii=1:size(graph.elements,2)
        aux= graph.elements(ii).nodeId;
        for jj=1:nD+1
            graph.nodes(aux(jj)).neighbour_elems=[graph.nodes(aux(jj)).neighbour_elems,ii];
        end
    end
     for ii=1:size(nodes,1)
         graph.nodes(ii).neighbour_elems=sort(graph.nodes(ii).neighbour_elems);
     end
end


end