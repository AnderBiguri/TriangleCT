function phi=meshDRLSE(graph,trivals,phi,niter,t,mu,lambda,alpha)

%      C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation", 
%        IEEE Trans. Image Processing, vol. 19 (12), pp.3243-3254, 2010.
%

% create a default phi, if not given


nodevals=meshInterpolateNodes(graph,trivals);
for ii=1:niter
    gphi=meshGradient(graph,phi);
    gmod=sqrt(sum((gphi).^2,2));
    gmod(gmod<1e-10)=0;
    edgeTerm=dirac_Lipschitz(phi).*meshDivergence(graph,nodevals.*(gphi./(gmod+1e-10)));
    areaTerm=dirac_Lipschitz(phi).*nodevals;
    distRegTerm=meshDivergence(graph,dps(gmod).*gphi);
    phi=phi+t*(mu*distRegTerm+lambda*edgeTerm+alpha*areaTerm);
end
end


function d=dirac_Lipschitz(x,epsilon)
if nargin==1
    epsilon=1.5;
end
d=(1/2/epsilon)*(1+cos(pi*x/epsilon));
b = (x<=epsilon) & (x>=-epsilon);
d = d.*b;
end

function out=dps(gmod)
% cond=gmod>1;
out=(gmod-1).*(gmod>1)+sin(2*pi*gmod)/(2*pi).*((gmod>=0) & (gmod<=1));
out=((out~=0).*out+(out==0)*1)./((gmod~=0).*gmod+(gmod==0)*1);  % compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)

end

