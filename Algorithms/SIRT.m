function [result]=SIRT(proj,geo,angles,graph, niter,varargin)


[verbose]=parse_inputs(varargin{:});
result=zeros(size(graph.elements,2),1,'single');
W=projection(ones(size(result),'single'),geo,angles,graph);
W(W<.0000001)=Inf;
V=backprojection(ones(size(proj),'single'),geo,angles,graph)+eps;

for ii=1:niter
    if (ii==1 && verbose==1); tic; end

        residual=proj-projection(result,geo,angles,graph);
        result=result+0.5*backprojection(residual./W,geo,angles,graph)./V;
        result(result<0)=0;

    
    
    if (ii==1 && verbose==1)
        expected_time=toc*niter;
        disp('SIRT');
        disp(['Expected duration  :    ',secs2hms(expected_time)]);
        disp(['Exected finish time:    ',datestr(datetime('now')+seconds(expected_time))]);
        disp('');
    end
end


end

%% Fucntions
function [verbose]=parse_inputs(varargin)
% create input parser
p=inputParser;
% add optional parameters
addParameter(p,'verbose',true,@islogical);

%execute
parse(p,varargin{:});
%extract
verbose=p.Results.verbose;

end