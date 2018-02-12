function [result]=SART(proj,geo,angles,graph, niter,varargin)


[verbose]=parse_inputs(varargin{:});
result=zeros(size(graph.elements,2),1);
W=meshForward(ones(size(result)),geo,angles,graph);
W(W<.0000001)=Inf;
for jj=1:size(proj,2)
    V(:,jj)=meshBack(ones(size(proj)),geo,angles(jj),graph)+eps;
end

for ii=1:niter
    if (ii==1 && verbose==1); tic; end
    
    
    for jj=1:size(proj,2)
        residual=proj(:,jj)-meshForward(result,geo,angles(jj),graph);
        result=result+0.5*meshBack(residual./W(:,jj),geo,angles(jj),graph)./V(:,jj);
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