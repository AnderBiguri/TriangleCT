function out=csl2mat(varargin)
% csl2mat(IN) converts the input coma separated list into a matrix

out=cell2mat({varargin{:}}.');
end