function [label,loc,data_cluster] = K_means(X,center)
k = size(center,1);
[n,~] = size(X);
loc = center;
flag = 1;
label = zeros(n,1);
% epoch=0;
while flag ~= 0
    dis = L2_distance(X, loc);
    [~,I] = sort(dis,2);
    label_new = I(:,1);
    if isequal(label_new,label)
        flag = 0;       % stop k-means
    else
        label = label_new;
        for i = 1:k
            data_cluster{i} = X(label==i,:);
            loc(i,:) = mean(data_cluster{i},1);
            loc(isnan(loc)) = 0;
        end
    end
%     epoch=epoch+1;
end
% fprintf('epoch:%i\n',epoch);