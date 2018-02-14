function nodevalsout=meshFilter(graph,nodevalsin,filter)

if ~isa(filter, 'function_handle')
    error("3rd argumetn filter shoudl be a fucntion handle");
end
nodevalsout=zeros(size(nodevalsin));
% By graph depth, or by distance?

distance=3; % sigmaGaussian=filtervals/3

distance=distance^2; % we will compare squares, because its cheaper
% By distance:
for ii=1:size(graph.nodes,2)
    tocheck=graph.nodes(ii).neighbour_nodes;
    deleted=uint32([ii]);
    nodesfilter=uint32([ii]); 
    while ~isempty(tocheck)
        pop=tocheck(1); tocheck(1)=[];
        % if we have already used it, ignore
        if ismembc(pop,deleted)
            continue
        end
        deleted=sort([deleted pop]);
        % if its farther than we want, ignore it   (As it is a delaunay,
        % there is no chance of having a node that is closer, linked by a
        % far node) (... right?)
        if sum((graph.nodes(ii).positions-graph.nodes(pop).positions).^2)>distance
            continue            
        end
        % add it to the nodes to use for filtering
        nodesfilter=[nodesfilter pop];  
        tocheck=unique([tocheck graph.nodes(pop).neighbour_nodes]);
    end
    filtervals=filter(graph.nodes(ii).positions,csl2mat(graph.nodes(nodesfilter).positions));
    nodevalsout(ii)=sum(nodevalsin(nodesfilter).*filtervals./sum(filtervals));
    
end
    
    
    
end