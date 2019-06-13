function [tree]=R_tree(MBR,m,M)
%R_TREE generates a R-tree search structure for the N-dimensional object
% that is bounded by the minimum bounding regon (MBR). Each leaf has M
% maximum objects and m<=M/2 minimum objects.
%
%      MBR is a [nO x nD*2] matrix, where nO is the number of objects to
%           order, nD is the number of dimensions. Each nO's MBR is
%           bounded by 2 nD sized points.
%


%% Grab some initial parameters and error checking
if m>M/2
    error('m has to be <=M/2')
end

if size(MBR,3)>1
    MBR=reshape(MBR,size(MBR,1),size(MBR,2)*2);
end
nD=size(MBR,2)/2;

%% Initialize structure of R-tree

% This variables legth is the number of nodes in the tree. Each value
% is the numbr of elements in the bins/leafs
tree.bin_n_elements=0;

% This array contains which elements are in each bin. It is of size
% length(bin_n_elements) x M+1
tree.bin_elements=zeros(1,M+1); % the extra one allows us to not overflow when its full

% This array stores the BB of each node. It is of size
% length(bin_n_elements) x nD*2
tree.bin_box=zeros(1,nD*2);
tree.bin_box(1,1:end/2)=-Inf;
tree.bin_box(1,end/2+1:end)=Inf;

% Maximum number of elements inside the tree
tree.M=M;

% minimum number of elements in a tree
tree.m=m;

% number of dimensions of data
tree.nD=nD;

% Store all objects minimum bounding regions
tree.MBR=MBR;

% Is this node a leaf node?
tree.isleaf=true;

% where is the root.
tree.root=1;
%% Build it


for ii=1:size(MBR,1)
    tree=R_tree_insert(MBR(ii,:),ii,tree);
    
    
    
    obj=[];
    for jj=1:length(tree.isleaf)
        if tree.isleaf(jj)
            obj=[obj tree.bin_elements(jj,1:tree.bin_n_elements(jj))];
        end
    end
    assert(isequal(unique(obj),sort(obj)));
    
end

end




function [tree]=R_tree_insert(MBR,oid,tree)


inserted=false;
current_path=[tree.root];
while ~inserted
    % Traverse the tree from root RN to the appropiate leaf. At each level,
    % select the node, L, whose MBR will require the minimum are
    % enlargement to cover the new MBR.
    
    
    if ~tree.isleaf(current_path(end))
        % we are not in a leaf
        ind=tree.bin_elements(current_path(end),1:tree.bin_n_elements(current_path(end)));
        % BUT, are we in node that points to leafs? (R*tree)
        if all(tree.isleaf(ind))
            % then the upmost priority is minimizng overlap.
            assert(isequal(length(tree.isleaf),size(tree.bin_box,1)));
            oe=[];
            for ii=1:tree.bin_n_elements(current_path(end))
                oe(ii)=overlap_enlargement(tree,current_path(end),ii,MBR);
            end
            min_increase=min(oe);
            idx=find(min_increase==oe);
            if length(idx)~=1
                vol_increase=zeros(1,length(idx));
                vol         =zeros(1,length(idx));
                for ii=1:length(idx)
                    vol(ii)=         prod(     tree.bin_box(tree.bin_elements(current_path(end),idx(ii)),1:end/2)               -     tree.bin_box(tree.bin_elements(current_path(end),idx(ii)),end/2+1:end));
                    vol_increase(ii)=prod(max([tree.bin_box(tree.bin_elements(current_path(end),idx(ii)),1:end/2);MBR(1:end/2)])-min([tree.bin_box(tree.bin_elements(current_path(end),idx(ii)),end/2+1:end);MBR(end/2+1:end)]));
                    vol_increase(ii)=vol_increase(ii)-vol(ii);
                end
                min_increase=min(vol_increase);
                idx2=find(min_increase==vol_increase);
                %                 idx=global_idx(idx);
                % if they are the same, then smalles volume wins
                if length(idx2)~=1
                    [~,idxidx]=min(vol(idx2));
                    idx=idx(idx2(idxidx));
                else
                    idx=idx2; 
                end
                
                %idx is the next path.
                current_path=[current_path tree.bin_elements(current_path(end),idx)];
                continue;
            else
                current_path=[current_path tree.bin_elements(current_path(end),idx)];
                continue;
            end
        end
        
        
        
        % So lets check all the nodes within the current node to see which
        % one we want to chose to insert the current object
        vol_increase=zeros(1,tree.bin_n_elements(current_path(end)));
        vol         =zeros(1,tree.bin_n_elements(current_path(end)));
        for ii=1:tree.bin_n_elements(current_path(end))
            vol(ii)=prod(tree.bin_box(tree.bin_elements(current_path(end),ii),1:end/2)-tree.bin_box(tree.bin_elements(current_path(end),ii),end/2+1:end));
            vol_increase(ii)=prod(max([tree.bin_box(tree.bin_elements(current_path(end),ii),1:end/2);MBR(1:end/2)])-min([tree.bin_box(tree.bin_elements(current_path(end),ii),end/2+1:end);MBR(end/2+1:end)]));
            vol_increase(ii)=vol_increase(ii)-vol(ii);
        end
        
        
        min_increase=min(vol_increase);
        idx=find(min_increase==vol_increase);
        % if they are the same, then smalles volume wins
        if length(idx)~=1
            [~,idxidx]=min(vol(idx));
            idx=idx(idxidx);
        end
        
        %idx is the next path.
        current_path=[current_path tree.bin_elements(current_path(end),idx)];
        
    else
        % we are in a leaf node
        
        % insert?
        if tree.bin_n_elements(current_path(end))<tree.M
            % INSERT
            tree=insert_element(MBR,tree,oid,current_path);
            
        else
            % oh.... we need to SPLIT
            
            
            % lets insert it anyways
            tree=insert_element(MBR,tree,oid,current_path);
            
            % This code implements the Quadratic split method.
            
            tree=quadratic_split(tree,current_path);
            
        end
        % done!
        inserted=true;
    end
end
end


function tree=insert_element(MBR_o,tree,oid,current_path)
% increase number of objects counter for this leaf
tree.bin_n_elements(current_path(end))=tree.bin_n_elements(current_path(end))+1;
% add object identifier to current leaf list
tree.bin_elements(current_path(end),tree.bin_n_elements(current_path(end)))=oid;
% increase bounding boxes from here to root.
auxbox=tree.bin_box(current_path(end),:);
tree.bin_box(current_path(end),:)=[max([auxbox(1:end/2); MBR_o(1:end/2)])  min([auxbox(end/2+1:end); MBR_o(end/2+1:end)])];
for ii=length(current_path)-1:-1:1
    % grab next current one and make sure it fits in its father node
    tree.bin_box(current_path(ii),:)=...
        [max([tree.bin_box(current_path(ii),      1:end/2);  tree.bin_box(current_path(ii+1),      1:end/2)]) ...
        min([tree.bin_box(current_path(ii),end/2+1:end);    tree.bin_box(current_path(ii+1),end/2+1:end)])];
end
end

function tree=quadratic_split(tree,current_path)


% grab all the elements
oids=tree.bin_elements(current_path(end),1: tree.bin_n_elements(current_path(end)));

% extract MBRs.
if tree.isleaf(current_path(end))
    leafMBR=zeros(length(oids),6);
    for ii=1:length(oids)
        leafMBR(ii,:)=tree.MBR(oids(ii),:);
    end
else
    % its not a leaf node, so the information is not in tree.MBR, but in
    % tree.bin_box
    leafMBR=zeros(length(oids),6);
    for ii=1:length(oids)
        leafMBR(ii,:)=tree.bin_box(oids(ii),:);
    end
end
% Find the 2 objects that will create largest dead space O(n^2)
ds=zeros(length(oids));
for ii=1:length(oids)
    for jj=ii+1:length(oids)
        ds(ii,jj)=dead_space(leafMBR(ii,:),leafMBR(jj,:));
    end
end
[~,ix]=max(ds(:));
[ix,jx]=ind2sub(size(ds),ix);
% lets create the two new bins
bin_elements1=oids(ix);
bin_elements2=oids(jx);

MBR1=leafMBR(ix,:);
MBR2=leafMBR(jx,:);

inserted=sort([ix jx]);
% Unitil there are not remaining objects, insert the object for which
% the difference of dead space if assigned to each of the two nodes is
% maximized in the node that requires less enlargement of its
% respective MBR
% i.e. choose the object that, if inserted in leaf 1 and leaf 2, will
% have maximum A=abs(DS1-DS2);
% then insert it in either 1 or 2, whichever needs to grow less.

while length(inserted)<length(oids)
    ds=-1*ones(length(oids),1);
    for ii=1:length(oids)
        if ismembc(ii,inserted) %fast ismember for sorted arrays
            continue; %skip if its already inserted
        end
        ds(ii)=abs(dead_space(leafMBR(ii,:),MBR1)-dead_space(leafMBR(ii,:),MBR2));
    end
    [~,ix]=max(ds);
    % now lets insert ix.
    volume1=prod(MBR1(1:end/2)-MBR1(end/2+1:end));
    newMBR1=[max([MBR1(1:end/2);leafMBR(ix,1:end/2)])  min( [MBR1(end/2+1:end); leafMBR(ix, end/2+1:end)])];
    volume1_grow=prod(newMBR1(1:end/2)-newMBR1(end/2+1:end))-volume1;
    
    volume2=prod(MBR2(1:end/2)-MBR2(end/2+1:end));
    newMBR2=[max([MBR2(1:end/2);leafMBR(ix,1:end/2)])  min( [MBR2(end/2+1:end); leafMBR(ix, end/2+1:end)])];
    volume2_grow=prod(newMBR2(1:end/2)-newMBR2(end/2+1:end))-volume2;
    
    % tie breaker if needed
    [~,auxidx]=min([length(bin_elements1) length(bin_elements2)]);
    if volume1_grow==volume2_grow
        if auxidx==1
            volume1_grow=volume1_grow-2*eps;
        else
            volume2_grow=volume2_grow-2*eps;
        end
    end
    % insert in whoever minimizes the volume increase
    if (volume1_grow<volume2_grow)
        MBR1=newMBR1;
        bin_elements1=sort([bin_elements1 oids(ix)]);
    else
        MBR2=newMBR2;
        bin_elements2=sort([bin_elements2 oids(ix)]);
    end
    inserted=sort([inserted ix]);
    
    
    
    
    % We may have inserted all possible so that the other one has m entries
    if (length(bin_elements1)>=tree.M+1-tree.m) && length(inserted)<tree.M+1
        % insert everything in 2 and break;
        toinsert=setdiff(1:length(oids),inserted);
        bin_elements2=sort([bin_elements2 oids(toinsert)]);
        auxMRB2=[];
        for ii=1:length(toinsert) %same as tree.m
            auxMRB2(ii,:)=leafMBR(toinsert(ii),:);
        end
        MBR2=[max([MBR2(1:end/2);auxMRB2(:,1:end/2)])  min( [MBR2(end/2+1:end); auxMRB2(:, end/2+1:end)])];
        
        break;
    elseif (length(bin_elements2)>=tree.M+1-tree.m ) && length(inserted)<tree.M+1
        % insert everything in 1 and break;
        toinsert=setdiff(1:length(oids),inserted);
        bin_elements1=sort([bin_elements1 oids(toinsert)]);
        auxMRB1=[];
        for ii=1:length(toinsert) %same as tree.m
            auxMRB1(ii,:)=leafMBR(toinsert(ii),:);
        end
        MBR1=[max([MBR1(1:end/2);auxMRB1(:,1:end/2)])  min( [MBR1(end/2+1:end); auxMRB1(:, end/2+1:end)])];
        break;
    end
    % otherwise keep as usual
end

% whelp, thats it. We now have split the node. We need to insert the
% information to the tree now.

% we need to create 1 new node, and replace the current one with one of the
% news ones
tree.bin_n_elements(current_path(end))=length(bin_elements1);
tree.bin_n_elements=[tree.bin_n_elements length(bin_elements2)];
% If the other one was leaf then this two are, and otherwise.
tree.isleaf=[tree.isleaf tree.isleaf(current_path(end))];

% Fill the elements
tree.bin_elements(current_path(end),:)=zeros(1,tree.M+1);
tree.bin_elements(current_path(end),1:length(bin_elements1))=bin_elements1;

tree.bin_elements=[tree.bin_elements;zeros(1,tree.M+1)];
tree.bin_elements(end,1:length(bin_elements2))=bin_elements2;

% Now update the MBRs of each node
tree.bin_box(current_path(end),:)=MBR1;
tree.bin_box=[tree.bin_box;zeros(1,size(tree.bin_box,2))];
tree.bin_box(end,:)=MBR2;

assert(isequal(length(tree.isleaf),size(tree.bin_box,1)));

if length(current_path)~=1
    
    % We have now updated the nodes, but we need to propagate this information
    tree.bin_n_elements(current_path(end-1))=tree.bin_n_elements(current_path(end-1))+1;
    % the last node is the new node.
    % The other one is also new, but we just modified it, so no need to
    % change anything.
    tree.bin_elements(current_path(end-1),tree.bin_n_elements(current_path(end-1)))=length(tree.bin_n_elements);
    
    
    % update all the MBRs to fit the 2 new nodes.
    for ii=length(current_path)-1:-1:1
        tree.bin_box(current_path(ii),:)=[max([tree.bin_box(current_path(ii),1:end/2); tree.bin_box(current_path(ii+1),1:end/2)]) min([tree.bin_box(current_path(ii),end/2+1:end); tree.bin_box(current_path(ii+1),end/2+1:end)])];
    end
    
    %% CAUTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    % we might have inserted 1 too many items and we might need to split
    % the upper node. D:
    
    % recursive?!?!?!
    if tree.bin_n_elements(current_path(end-1))>tree.M
        tree=quadratic_split(tree,current_path(1:end-1));
    end
    
else
    % if we were splitting the root, then there is no need to update upper
    % nodes (because they dont exist, duh). But we need to create a new
    % root!
    
    % we need to create a second node to be the new root
    tree.bin_elements=[tree.bin_elements;zeros(1,tree.M+1)];
    % and put the old root there, plus the bin created from the current
    % split.
    tree.bin_elements(end,1:2)=[tree.root length(tree.bin_n_elements)];
    
    % now, lets store how many elements this new root has (2)
    tree.bin_n_elements=[ tree.bin_n_elements 2];
    
    % set up its MBR
    tree.bin_box(end+1,:)=[max([MBR1(1:end/2);MBR2(1:end/2)]) min([MBR1(end/2+1:end);MBR2(end/2+1:end)])];
    
    % is leaf? nono
    tree.isleaf=[tree.isleaf false];
    
    % update root
    tree.root=length(tree.bin_n_elements);
    
end
obj=[];
for jj=1:length(tree.isleaf)
    if tree.isleaf(jj)
        obj=[obj tree.bin_elements(jj,1:tree.bin_n_elements(jj))];
    end
end
assert(isequal(unique(obj),sort(obj)));
end

function ds=dead_space(MBR1,MBR2)

volume1=prod(MBR1(1:end/2)-MBR1(end/2+1:end));
volume2=prod(MBR2(1:end/2)-MBR2(end/2+1:end));
volume= prod(max([MBR1(1:end/2);MBR2(1:end/2)])-min([MBR1(end/2+1:end ); MBR2(end/2+1:end)]));
ds=volume-volume1-volume2;
end

function oe=overlap_enlargement(tree,node,leaf,MBR)
% sanity check
ind=tree.bin_elements(node,1:tree.bin_n_elements(node));
assert(all(tree.isleaf(ind)),'Not all children are leaf! wooot?');
% Now lets compute how much overlap enlargement is there if I insert MBR
% into leaf number "leaf".
% 1) Compute how much overlap there is by leaf "leaf"
ind_leaf=tree.bin_elements(node,leaf);
overlap=zeros(2,tree.bin_n_elements(node));
for ii=1:tree.bin_n_elements(node)
    if ii==leaf
        continue;
    end
    overlap(1,ii)=intersection_volume(tree.bin_box(ind_leaf,:),tree.bin_box(ind(ii),:));
end


% 2) now we introduce the new element and see how much overlap there is
% now.
% (we can introduce it because we are not returning tree.
tree.bin_box(ind_leaf,:)=[max([tree.bin_box(ind_leaf,1:end/2);MBR(1:end/2)]) min([tree.bin_box(ind_leaf,end/2+1:end);MBR(end/2+1:end)])];
for ii=1:tree.bin_n_elements(node)
    if ii==leaf
        continue;
    end
    overlap(2,ii)=intersection_volume(tree.bin_box(ind_leaf,:),tree.bin_box(ind(ii),:));
end

% this is the "overlap enlargement"
oe=sum(overlap(2,:))-sum(overlap(1,:));

end

function si=intersection_volume(MBR1,MBR2)

Mx=[MBR1(1) MBR2(1)];
My=[MBR1(2) MBR2(2)];
Mz=[MBR1(3) MBR2(3)];

mx=[MBR1(4) MBR2(4)];
my=[MBR1(5) MBR2(5)];
mz=[MBR1(6) MBR2(6)];

lx=max(0,min(Mx) -max(mx));
ly=max(0,min(My) -max(my));
lz=max(0,min(Mz) -max(mz));

si=lx*ly*lz;



end