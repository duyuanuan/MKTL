function [prob,Yt] = MKTL(Xs,Xt,Xs_label,options)   % prob: ��������    Yt: ��ǩ

core_num_arr = options.core_num_arr;
pred_strategy = options.pred_strategy;
n_Class = length(unique(Xs_label));
if length(core_num_arr)==1
    core_num_arr = core_num_arr*ones(1,n_Class);
elseif (length(core_num_arr)>1 && length(core_num_arr)<n_Class) || length(core_num_arr)>n_Class
    error("The size of core_num_arr is incorrect!");
end
Nt = size(Xt,1);
core_net = [];
I = 1;
prob = zeros(Nt,n_Class);

%% ��Դ���ݽ��о��࣬�õ�����
for class_i = 1:n_Class
    n = core_num_arr(class_i);
    data = Xs(Xs_label==class_i,:);
    N_data = size(data,1);
    if n<=size(data,1)
        center_t = data(1:n,:);                                 % #####################
%         center_t = data(randperm(N_data,n),:);                % %%%%%%%%%%%%%%%%%%%%%
    else
        center_t = data;
    end
    
    [~,loc_t,~] = kmeans(data,size(center_t,1),'Start',center_t);
    core_net(I:(I+size(loc_t,1)-1),:) = loc_t;
%     [~,loc_t,~] = K_means(data',center_t');
%     core_net(:,I:(I+size(loc_t,1)-1)) = loc_t';

    I = I+size(loc_t,1);
end

%% center_num < core_num
num_r = sum(core_num_arr)-I+1;
if sum(core_num_arr)<Nt && num_r > 0
    core_r = Xt(1:num_r,:);         % Ϊ�˷���ʵ�����ĸ��֣�����ѡȡû�������ݵ�ǰn�����ݣ����������ѡ��
%     core_r = Xt(randperm(Nt,n),num_r);  % ���ѡ������
    core_net = [core_net;core_r];
end

%% deal with the nan array
index_nan = isnan(core_net);
if sum(index_nan(1,:))>1
    core_net(index_nan) = Xt(1:sum(index_nan(1,:)),:);
end

%% ʹ�ú�����Ŀ�����ݽ��о���
[label_cluster,agent,~] = K_means(Xt,core_net);

%% ���ࣨ�����ƣ�Agent����ͶƱ�ƣ�Mass����
options.core_num_arr = core_num_arr;
cluster_num = size(agent,1);

if strcmp(pred_strategy ,'Agent')
    X_p = agent;
elseif strcmp(pred_strategy,'Mass')
    X_p = Xt;
end

[prob_cluster,~] = kNN(Xs,Xs_label,X_p,options);

%% ��ǩ����
for i = 1:cluster_num
    if strcmp(pred_strategy ,'Agent')
        prob(label_cluster==i,:) = repmat(prob_cluster(i,:),sum(label_cluster==i),1);
    elseif strcmp(pred_strategy,'Mass')
        prob(label_cluster==i,:) = repmat(mean(prob_cluster(label_cluster==i,:)),sum(label_cluster==i),1);
    end
end

[~,Yt] = sort(prob,2,'descend');
Yt = Yt(:,1);