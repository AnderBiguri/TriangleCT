function image=backprojection(proj,geo,angles,graph)
geo=checkGeo(geo,angles);

%% Angles

assert(isa(proj,'single'),'backprojection:InvalidInput','projections should be single prevision')
assert(isreal(angles),'backprojection:InvalidInput','Angles shoudl be real (non-complex)');
assert(size(angles,1)==1 | size(angles,1)==3 ,'backprojection:InvalidInput','Angles shoudl be of size 1xN or 3xN');
angles=double(angles); %in case they were single.
if size(angles,1)==1
   angles=repmat(angles,[3 1]);
   angles(2,:)=0;
   angles(3,:)=0;
end
[nodes,elements,neighbours,boundary]=graphStruct2graphArray(graph);
elements=elements-1;
neighbours=neighbours-1;
boundary=boundary-1;
image=graphBackward(proj,geo,angles,elements,nodes,neighbours,boundary);

end