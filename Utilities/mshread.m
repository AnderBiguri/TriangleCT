function [nodes,faces,elements ] = mshread( file )


fid = fopen(file, 'rt');
tline=fgetl(fid);
nline=1;
while ~strcmp(tline,'$Nodes')&&ischar(tline)
    tline=fgetl(fid);
    nline=nline+1;
end

N_n=fscanf(fid,'%d',1);
nline=nline+1+N_n;
nodes=fscanf(fid,'%d %f %f %f',[4,N_n]);
nodes=nodes(2:end,:).';%remove id

while ~strcmp(tline,'$Elements')&&ischar(tline)
    tline=fgetl(fid);
    nline=nline+1;
end
N_e=fscanf(fid,'%d',1);
fgetl(fid);
nline=nline;
fclose(fid);

elemenface=readtable(file,'FileType','text','Delimiter',' ','HeaderLines',nline,...
    'Format', '%f%f%f%f%f%f%f%f%f','ReadVariableNames', false,'TreatAsEmpty','$EndElements');
elemenface=table2array(elemenface(1:end-1,:));
elemenind=find(~isnan(elemenface(:,9)),1);
faces=elemenface(1:elemenind-1,6:8);
elements=elemenface(elemenind:end,6:9);
% 
% 
% N_n      = dlmread(file,'',[9 1 9 1]);
% N_n1      = dlmread(file,'',[10 3 10 3]);
% N_n2      = dlmread(file,'',[11+N_n1 3 11+N_n1 3]);
% 
% N_f      = dlmread(file,'',[15+N_n 3 15+N_n 3]);
% N_e      = dlmread(file,'',[16+N_n+N_f 3 16+N_n+N_f 3]);
% 
% node_id     = dlmread(file,'',[11 0 11+N_n 0]);
% 
% nodes       = dlmread(file,'',[11 1 10+N_n1 3]);
% nodes2       = dlmread(file,'',[12+N_n1 1 11+N_n1+N_n2 3]);
% nodes=[nodes;nodes2];
% faces       = dlmread(file,'',[16+N_n 1 15+N_n+N_f 3]);
% elements       = dlmread(file,'',[17+N_n+N_f 1 16+N_n+N_f+N_e 4]);