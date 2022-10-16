function [prob,Yt] = kNN(Xs,Ys,Xt,options)
%NNA (Nearest Neighbor Assign)
%   prob - probility vector
%   Yt   - label
dis_weight = options.dis_weight;
k = options.kp;

dis = L2_distance(Xs,Xt);

[dis2,I2] = sort(dis,1);  
    
index2 = I2(1:k,:); % the nearst Xs of Xt
dis2 = dis2(1:k,:);

n_Class = length(unique(Ys));
Nt = size(Xt,1);
prob = zeros(Nt,n_Class);

for i = 1:Nt
    label_t = Ys(index2(:,i));
    dis_label_t = dis2(:,i);
    if dis_weight==0    % without distance weight
        Statis = tabulate(label_t);
        prob(i,:) = [Statis(:,3)./100;zeros(n_Class-length(Statis(:,3)),1)];
    else    % with distance weight
        prob(i,:) = dis_weight_vote(dis_label_t,label_t,n_Class);
    end
end
    [~,Yt] = sort(prob,2,'descend');
    Yt = Yt(:,1);
end
