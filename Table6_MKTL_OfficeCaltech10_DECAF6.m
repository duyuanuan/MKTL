clear all;
clc;
close all;
%%
source_domains = {'Caltech10', 'Caltech10', 'Caltech10', 'Amazon', 'Amazon', 'Amazon', 'Webcam', 'Webcam', 'Webcam', 'Dslr', 'Dslr', 'Dslr'};
target_domains = {'Amazon', 'Webcam', 'Dslr', 'Caltech10', 'Webcam', 'Dslr', 'Caltech10', 'Amazon', 'Dslr', 'Caltech10', 'Amazon', 'Webcam'};

%% parameter
options.core_num_arr = 1;           % n: int or int array
options.kp = 3;                     % kNN的参数
options.dis_weight = 1;             % 使用距离的倒数作为kNN的投票权重: '1':使用    '0':不使用
options.pred_strategy = 'Agent';    % 'Agent' or 'Mass'

%%
arr_A_kNN = [1,5,6,17,7,1,2,1,10,3,5,20];
arr_M_kNN = [1,7,6,17,1,1,2,1,1,1,3,2];

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
    load(strcat('Data/Office+Caltech10_DeCAF6/', src, '_decaf.mat'));
    Xs = fea';
    Xs_label = gnd;
    
    load(strcat('Data/Office+Caltech10_DeCAF6/', tgt, '_decaf.mat'));
    Xt = fea';
    Xt_label = gnd;
    
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
    fprintf('MKTL(%s-kNN) acc:%f\n',options.pred_strategy,acc);
end