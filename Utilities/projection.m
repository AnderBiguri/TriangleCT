function proj=projection(trivals,geo,angles,graph)
geo=checkGeo(geo,angles);

%% Angles
assert(isreal(angles),'projection:InvalidInput','Angles shoudl be real (non-complex)');
assert(size(angles,1)==1 | size(angles,1)==3 ,'projection:InvalidInput','Angles shoudl be of size 1xN or 3xN');
angles=double(angles); %in case they were single.
if size(angles,1)==1
   angles=repmat(angles,[3 1]);
   angles(2,:)=0;
   angles(3,:)=0;
end
proj=graphForward(trivals,geo,angles,graph);

end