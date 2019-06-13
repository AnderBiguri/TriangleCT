function plot_R_tree(tree)

    centroids=(tree.MBR(:,1:end/2)-tree.MBR(:,end/2+1:end))/2+tree.MBR(:,end/2+1:end);
    scatter3(centroids(:,1),centroids(:,2),centroids(:,3),'g.');
    hold on
    
    colors=[0 0 0; 0 0 1; 1 0 0];
    
    alldrawn=false;
    %start with root
    toplot=tree.root;
    depth=0;
    isleaf=tree.isleaf(toplot);
    while ~alldrawn
        width=max(5-depth,1);
        for ii=1:length(toplot)
            plotcube(tree.bin_box(toplot(ii),1:3)-tree.bin_box(toplot(ii),4:6),tree.bin_box(toplot(ii),4:6),0,colors(mod(depth,size(colors,2))+1,:),width);
            text(tree.bin_box(toplot(ii),1)+(depth)*0.01,tree.bin_box(toplot(ii),2)+depth*0.01,tree.bin_box(toplot(ii),3)+depth*0.01,num2str(toplot(ii)));
        end
        
        toplot_new=[];
        isleaf_new=[];
        for ii=1:length(toplot)
            if ~ isleaf(ii)
                toplot_new=[toplot_new tree.bin_elements(toplot(ii),:)];
            end
            
        end
        toplot_new(toplot_new==0)=[];
        toplot=toplot_new;
        isleaf=tree.isleaf(toplot);
        depth=depth+1;
        if isempty(toplot)
            alldrawn=true;
        end
            
    end
    
    
    
    
    hold off
    axis equal
    axis off
    

end