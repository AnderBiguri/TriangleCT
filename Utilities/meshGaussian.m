function gauss=meshGaussian(mean,samples,sigmadiag)
% Gives the calues of a Gaussian fucntion in arbitray samples
% For now not cross-sigmas


% If no sigma is ginve, set it up cosidering the dimensionality
if nargin<3
    sigmadiag=ones(size(mean,2),1);
end
gauss=zeros(size(samples,1),1);
for dim=1:size(mean,2)
  gauss=gauss-(samples(:,dim)-mean(dim)).^2/2*sigmadiag(dim)^2;
end
gauss=exp(gauss)/prod(sigmadiag)/2/pi;



