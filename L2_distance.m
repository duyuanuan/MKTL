function [dis] = L2_distance(data1,data2,bsqrt)
%L2_DISTANCE Calculate the distance matrix of two sets of data or one set of data
%   data1:  nsample_a * nFeature
%   data2:  nsample_a * nFeature
%   bsqrt:  Whether to sqrt the distance matrix
if (~exist('bsqrt','var')) || isempty(bsqrt)
    bsqrt = 1;
end
if (~exist('data2','var')) || isempty(data2)
    N = size(data1,1);
    G = data1*data1';
    H = repmat(diag(G), 1, N);
    dis = H+H'-2*G;
else
    t1 = sum(data1.^2, 2);
    t2 = reshape(sum(data2.^2, 2), 1, size(data2, 1));
    t3 = data1*data2';
    dis = t1 + t2 - 2*t3;
end
dis = abs(dis);
if bsqrt
    dis = sqrt(dis);
end

