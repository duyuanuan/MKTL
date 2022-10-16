function probility = dis_weight_vote(dis,label_t,n_Class)
    dis = 1./dis;
    probility = zeros(n_Class,1);
    for i = 1:length(dis)
        probility(label_t(i)) = probility(label_t(i))+dis(i);
    end
    probility = probility./sum(dis);
end