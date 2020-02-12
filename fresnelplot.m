function fresnelplot(n1,n2,k2,plotopt)
thetadeg = (0:0.1:90);
theta = thetadeg*pi/180;
[a,b,c] = intrc(n1,n2,k2,theta);
if nargin<4 || isempty(plotopt)
    fprintf('No plotting option specified. Type \''help fresnelplot\'' for further details.\n');
    return;
end