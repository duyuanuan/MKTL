clear all;
clc;
% close all;
%%
source_domains = {'c','c','i','i','p','p'};
target_domains = {'i','p','c','p','c','i'};

%% parameter
options.core_num_arr = 1;           % n: int or int array
options.kp = 20;                    % kNN的参数
options.dis_weight = 1;             % 使用距离的倒数作为kNN的投票权重: '1':使用    '0':不使用
options.pred_strategy = 'Agent';    % 'Agent' or 'Mass'
%%
arr_A_kNN = [1,1,1,3,1,1];
arr_M_kNN = [1,1,1,3,1,1];

if(strcmp(options.pred_strategy,'Agent'))
    arr_n = arr_A_kNN;
elseif(strcmp(options.pred_strategy,'Mass'))
    arr_n = arr_M_kNN;
end

%%
for i_sample = 1:length(source_domains)
    src = char(source_domains{i_sample});
    tgt = char(target_domains{i_sample});
    fprintf('==============================%s to %s==============================\n',src,tgt)
%     data = load("path of source data");
    data = load(strcat('Data/imageCLEF_resnet50/',src,'_',src,'.csv'));
	Xs = data(1:end,1:end-1);
    Xs_label = data(1:end,end) + 1;
	Xs = Xs ./ repmat(sum(Xs, 2), 1, size(Xs, 2));
	Xs = zscore(Xs, 1)';
    
%     data = load("path of target data");
    data = load(strcat('Data/imageCLEF_resnet50/',src,'_',tgt,'.csv'));
	Xt = data(1:end,1:end-1);
    Xt_label = data(1:end,end) + 1;
	Xt = Xt ./ repmat(sum(Xt, 2), 1, size(Xt, 2));
	Xt = zscore(Xt, 1)';
    
    clear data;
    
    Xs_r = Xs./repmat(sqrt(sum(Xs.^2)), [size(Xs,1) 1]);
    Xt_r = Xt./repmat(sqrt(sum(Xt.^2)), [size(Xt,1) 1]);
    X = [Xs_r,Xt_r];
       
%     [p,y] = kNN(Xs_r',Xs_label,Xt_r',options);
%     acc_kNN(i_sample) = length(find(y==Xt_label))/size(Xt_label,1)*100;
%     fprintf("kNN acc: %f\n",acc_kNN(i_sample));

    options.core_num_arr = arr_n(i_sample);
    fprintf('Parameter:\tn:%i\tkp:%i\n',options.core_num_arr,options.kp);
    tic;
    [prob,Yt] = MKTL(Xs_r',Xt_r',Xs_label,options);
    toc;
    acc =  length(find(Yt==Xt_label))/size(Xt_label,1)*100;
    fprintf('KTL(%s-kNN) acc:%f\n',options.pred_strategy,acc);
end
