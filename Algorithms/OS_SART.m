function [result]=OS_SART(proj,geo,angles,graph, niter,varargin)


[verbose,mask]=parse_inputs(graph,varargin{:});


[alphablocks,orig_index]=order_subsets(angles,20,'random');


result=zeros(size(graph.elements,2),1,'single');
W=projection(ones(size(result),'single'),geo,angles,graph);
W(W<.0000001)=Inf;


for jj=1:size(proj,3)
    V(:,jj)=backprojection(ones(size(proj),'single'),geo,angles(jj),graph)+eps;
end

for ii=1:niter
    if (ii==1 && verbose==1); tic; end
    
    
    for jj=1:length(alphablocks)
        residual=proj(:,:,orig_index{jj})-projection(result,geo,angles(orig_index{jj}),graph);
        result=result+mask.*backprojection(residual./W(:,:,orig_index{jj}),geo,angles(orig_index{jj}),graph)./sum(V(:,orig_index{jj}),2);
        result(result<0)=0;

    end
    
    if (ii==1 && verbose==1)
        expected_time=toc*niter;
        disp('SART');
        disp(['Expected duration  :    ',secs2hms(expected_time)]);
        disp(['Exected finish time:    ',datestr(datetime('now')+seconds(expected_time))]);
        disp('');
    end
end


end

%% Fucntions
function [verbose,mask]=parse_inputs(graph,varargin)
% create input parser
p=inputParser;
% add optional parameters
addParameter(p,'verbose',true,@islogical);
addParameter(p,'mask',true,@(x)islogical(x)&&(length(x)==length(graph.elements)));
%execute
parse(p,varargin{:});
%extract
verbose=p.Results.verbose;
mask=p.Results.mask;
end