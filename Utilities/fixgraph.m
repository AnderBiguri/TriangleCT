function graph=fixgraph(graph)
% FIXGRAPH fixes the structure of the input graph, if needed.


% if nodes are not built as single elements in a graph
if size(graph.nodes,1)~=1
    nodes=graph.nodes;
    nD=size(nodes,2);
    graph=rmfield(graph,'nodes');
    for ii=1:size(nodes,1)
        graph.nodes(ii).positions=single(nodes(ii,:));
        graph.nodes(ii).neighbour_elems=[];
        graph.nodes(ii).neighbour_nodes=[];
    end
    
    for ii=1:size(graph.elements,2)
        aux= graph.elements(ii).nodeId;
        for jj=1:nD+1
            graph.nodes(aux(jj)).neighbour_elems=uint32([graph.nodes(aux(jj)).neighbour_elems,ii]);
            graph.nodes(aux(jj)).neighbour_nodes=uint32(unique([graph.nodes(aux(jj)).neighbour_nodes aux]));
        end
    end
     for ii=1:size(nodes,1)
         graph.nodes(ii).neighbour_elems=sort(graph.nodes(ii).neighbour_elems);
         
         graph.nodes(ii).neighbour_nodes=sort(graph.nodes(ii).neighbour_nodes);
         graph.nodes(ii).neighbour_nodes(graph.nodes(ii).neighbour_nodes==ii)=[];
     end
end
% neihgbour nodes are missing
if ~isfield(graph.nodes(1),'neighbour_nodes')
    nD=size(graph.nodes(1).positions,2);
    for ii=1:size(graph.nodes)
        graph.nodes(ii).neighbour_nodes=[];
    end
     for ii=1:size(graph.elements,2)
        aux= graph.elements(ii).nodeId;
        for jj=1:nD+1
            graph.nodes(aux(jj)).neighbour_nodes=uint32(unique([graph.nodes(aux(jj)).neighbour_nodes aux]));
            
        end
    end
     for ii=1:size(graph.nodes,2)
         graph.nodes(ii).neighbour_nodes=sort(graph.nodes(ii).neighbour_nodes);
         graph.nodes(ii).neighbour_nodes(graph.nodes(ii).neighbour_nodes==ii)=[];
     end
end

end