%% This code shows how the weigted gradient is proportionally equivalent to the sobel operator.
clear; clc

%% Sobel operator
% A is our image
Ax=sym('x',[3 3]); % X coordinates
Ay=sym('y',[3 3]); % Y coordinates
A=sym('f',[3 3]);  % fucntion values

% Sobel matrices
Gx=[+1 0 -1;2 0 -2; 1 0 -1];
Gy=[1 2 1; 0 0 0; -1 -2 -1];

%A Apply Sobel operator
gradx=sum(sum(Gx.*A));
grady=sum(sum(Gy.*A));

% modulus of the sobel operator
grad_mod=sqrt(gradx^2+grady^2);
grad_angle=atan2(grady,gradx);

%% Weigthed least squares method (assume center of pixels are the data points

% Directional vectors
u=[Ax(:)-Ax(2,2),Ay(:)-Ay(2,2), ones(9,1)];

% substitute real values
u=subs(u,{'x1_1','x2_1','x3_1'},[-1 -1 -1]);
u=subs(u,{'x1_2','x2_2','x3_2'},[0 0 0]);
u=subs(u,{'x1_3','x2_3','x3_3'},[1 1 1]);
u=subs(u,{'y1_1','y1_2','y1_3'},[1 1 1]);
u=subs(u,{'y2_1','y2_2','y2_3'},[0 0 0]);
u=subs(u,{'y3_1','y3_2','y3_3'},[-1 -1 -1]);


% fucntion values
f=A(:);
% weigthed distances
% (OK, I cheated here, by directly appliying the numerical values)
W=diag([1/sqrt(2) 1 1/sqrt(2) 1 0 1 1/sqrt(2) 1 1/sqrt(2)]).^2;

grad2=(u.'*W*u)^-1*u.'*W*f;



grad2_mod=sqrt(grad2(1)^2+grad2(2)^2);
grad2_angle=atan2(grad2(2),grad2(1));
%% Now lets give numerical values


simplify(grad_mod/grad2_mod)
% imgvals=rand(1,9);
% 
% res=double(subs(grad_mod,{'f1_1' 'f1_2' 'f1_3' 'f2_1' 'f2_2' 'f2_3' 'f3_1' 'f3_2' 'f3_3'},imgvals))
% res2=double(subs(grad2_mod,{'f1_1' 'f1_2' 'f1_3' 'f2_1' 'f2_2' 'f2_3' 'f3_1' 'f3_2' 'f3_3'},imgvals))
% 
% p=res/res2