function [node,t]=R_tree_search(tree,ray,graph)
ray=ray';
ray=ray(:);


isin = rayBoxIntersection(ray(1:3), ray(4:6)-ray(1:3), tree.bin_box(tree.root,4:6), tree.bin_box(tree.root,1:3));
if ~isin
   node=0;
   return;
end
% query is a point
finished=false;
current_path=tree.root;
unchecked=[];
node=[];
while ~finished
    
    if ~isempty(current_path) && ~tree.isleaf(current_path) 
         isin=zeros(1,tree.M);
         for ii=1:tree.bin_n_elements(current_path)
             
             isin(ii)=rayBoxIntersection(ray(1:3), ray(4:6)-ray(1:3), tree.bin_box(tree.bin_elements(current_path,ii),4:6),tree.bin_box(tree.bin_elements(current_path,ii),1:3));
         end

         % depth first
         intersect=find(isin);
         if ~isempty(intersect)
             if length(intersect)>1
                unchecked=[unchecked tree.bin_elements(current_path,intersect(2:end)) ];
             end
             current_path=tree.bin_elements(current_path,intersect(1));
         else
             current_path=[];
         end
    else
       node=[node current_path]; % if it is a leaf it intersects
%        if ~isempty(node)
%            assert(isequal(unique(node),sort(node)));
%        end
       % do we have more?
       if ~isempty(unchecked)
            current_path=unchecked(end);
            unchecked(end)=[];
       else
           finished=true;
       end
    end
    

end

% now which element do they actually intersect?

%1) unpack elements
elements=[];
for ii=1:length(node)
%    assert(tree.isleaf(node(ii)));
   elements=[elements tree.bin_elements(node(ii),1:tree.bin_n_elements(node(ii)))];
end
% assert(isequal(unique(elements),sort(elements)))
% Bug here
% elements=unique(elements);
for ii=1:length(elements)
    
   triangle=csl2mat(graph.nodes(graph.elements(graph.boundary_elems(elements(ii))).nodeId).positions);
   [intersects(ii),t(ii,:)]=isLineTriangleIntersect(reshape(ray',[],2)',triangle);
             
%    intersects(ii)=rayBoxIntersection(ray(1:3), ray(4:6), tree.MBR(elements(ii),4:6),tree.MBR(elements(ii),1:3));
end

intersect_elements=elements(intersects==1);
t=t(intersects==1);
[t,ind]=min(t);
node=intersect_elements(ind);
end