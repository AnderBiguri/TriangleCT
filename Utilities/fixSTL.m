function fv=fixSTL(fv)

[v2,~,ic]=unique(fv.vertices,'rows');

faces=fv.faces;
for ii=1: size(ic,1)
   fv.faces(faces==ii)=ic(ii); 
end
fv.vertices=v2;
end