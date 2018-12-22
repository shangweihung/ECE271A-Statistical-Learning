close all 
clear all
clc

load('TrainingSamplesDCT_subsets_8.mat')
load('Alpha.mat')
load('Prior_1.mat')
% Load Zig-Zag pattern
Zigzag=load('Zig-Zag Pattern.txt');
Zigzag=Zigzag+1;

%% 
% Change dataset here
FG = D1_FG;
BG = D1_BG;
[row_f,col_f] = size(FG);
[row_b,col_b] = size(BG);

% Calculate Prior
Prior_FG=row_f/(row_f+row_b);          
Prior_BG=row_b/(row_f+row_b);

% calculate for each column's mean
mu_fg=sum(FG)/row_f;
mu_bg=sum(BG)/row_b;

% calculate covariance
covar_fg=cov(FG);
covar_bg=cov(BG);

% Load Ground Truth
% Load Test sample 
Img_ori=imread('cheetah.bmp');                       %Use zero padding for edge & corner
Img = padarray(Img_ori,[7 7],'symmetric','pre');     %classified pixel:right bottom
Img=im2double(Img);
Gt=imread('cheetah_mask.bmp');
Gt=im2double(Gt);

Errors_Bay=zeros(1,length(alpha));
Errors_MAP=zeros(1,length(alpha));
Errors_ML=zeros(1,length(alpha));

%% Predictive solution
for k=1:length(alpha)
    k
    % Generate data 
    [row,col]=size(Img);
    A=zeros((row-7)*(col-7),64);
    index=1;
    for i=1:row-7
        for j=1:col-7
            Field=Img(i:i+7,j:j+7);
            DCT=dct2(Field);
            DCT_64(Zigzag)=DCT;  % turn 8*8 into 1*64 with Zigzag pattern
            A(index,:)=DCT_64;
            index=index+1;
        end
    end
    
    
    %calculate mu1 for both class 
    Sigma_0_FG = diag(W0)*alpha(k);
    sig_FG = covar_fg/row_f;
    first_term_FG = Sigma_0_FG /(sig_FG + Sigma_0_FG) * mu_fg';
    second_term_FG = sig_FG /(sig_FG + Sigma_0_FG) * mu0_FG';
    mu_1_FG = first_term_FG + second_term_FG;
    mu_1_FG = mu_1_FG';
    
    Sigma_0_BG = diag(W0)*alpha(k);
    sig_BG = covar_bg/row_b;
    first_term_BG = Sigma_0_BG /(sig_BG + Sigma_0_BG) * mu_bg';
    second_term_BG = sig_BG /(sig_BG + Sigma_0_BG) * mu0_BG';
    mu_1_BG = first_term_BG + second_term_BG;
    mu_1_BG = mu_1_BG';
    
    %calculate var1 for both class 
    var_1_FG = (sig_FG * Sigma_0_FG) /(sig_FG + Sigma_0_FG);
    var_1_BG = (sig_BG * Sigma_0_BG) /(sig_BG + Sigma_0_BG);
    
    % for X|T
    varXT_FG = covar_fg + var_1_FG;
    varXT_BG = covar_bg + var_1_BG;
    
    
    % Plug into bayesian decision rule
    alp_fg=log(((2*pi)^64)*det(varXT_FG))-2*log(Prior_FG);
    alp_bg=log(((2*pi)^64)*det(varXT_BG))-2*log(Prior_BG);
    
    g_fg=zeros((col-7)*(row-7),1);
    g_bg=zeros((col-7)*(row-7),1);
    temp_dxy_fg=zeros((col-7)*(row-7),1);
    temp_dxy_bg=zeros((col-7)*(row-7),1);

    predict=zeros(1,(col-7)*(row-7));
    for index=1:(col-7)*(row-7)
        temp_dxy_fg(index)=(A(index,:)-mu_1_FG) * (inv(varXT_FG)* (A(index,:)-mu_1_FG)');
        temp_dxy_bg(index)=(A(index,:)-mu_1_BG) * (inv(varXT_BG)* (A(index,:)-mu_1_BG)');
        g_fg(index)=1 / (1+ exp( temp_dxy_fg(index) - temp_dxy_bg(index) + alp_fg - alp_bg));
        g_bg(index)=1 / (1+ exp( temp_dxy_bg(index) - temp_dxy_fg(index) + alp_bg - alp_fg));

        if g_fg(index)>0.5
            predict(1,index)=1;
        end
    end
    
    predict_2d=reshape(predict,[col-7,row-7]);
    predict_2d=predict_2d';

    Errors_Bay(k)=ProbOfError(predict_2d,Gt,Prior_FG,Prior_BG);
    Errors_Bay(k)    

%% MAP    
    % Plug into bayesian decision rule
    alp_fg=log(((2*pi)^64)*det(covar_fg))-2*log(Prior_FG);
    alp_bg=log(((2*pi)^64)*det(covar_bg))-2*log(Prior_BG);
    
    g_fg=zeros((col-7)*(row-7),1);
    g_bg=zeros((col-7)*(row-7),1);
    temp_dxy_fg=zeros((col-7)*(row-7),1);
    temp_dxy_bg=zeros((col-7)*(row-7),1);

    predict=zeros(1,(col-7)*(row-7));
    for index=1:(col-7)*(row-7)
        temp_dxy_fg(index)=(A(index,:)-mu_1_FG) * (inv(covar_fg)* (A(index,:)-mu_1_FG)');
        temp_dxy_bg(index)=(A(index,:)-mu_1_BG) * (inv(covar_bg)* (A(index,:)-mu_1_BG)');
        g_fg(index)=1 / (1+ exp( temp_dxy_fg(index) - temp_dxy_bg(index) + alp_fg - alp_bg));
        g_bg(index)=1 / (1+ exp( temp_dxy_bg(index) - temp_dxy_fg(index) + alp_bg - alp_fg));

        if g_fg(index)>0.5
            predict(1,index)=1;
        end
    end
    
    predict_2d=reshape(predict,[col-7,row-7]);
    predict_2d=predict_2d';

    Errors_MAP(k)=ProbOfError(predict_2d,Gt,Prior_FG,Prior_BG);
    Errors_MAP(k)

%% ML
    % Plug into bayesian decision rule
    alp_fg=log(((2*pi)^64)*det(covar_fg))-2*log(Prior_FG);
    alp_bg=log(((2*pi)^64)*det(covar_bg))-2*log(Prior_BG);
    
    g_fg=zeros((col-7)*(row-7),1);
    g_bg=zeros((col-7)*(row-7),1);
    temp_dxy_fg=zeros((col-7)*(row-7),1);
    temp_dxy_bg=zeros((col-7)*(row-7),1);

    predict=zeros(1,(col-7)*(row-7));
    for index=1:(col-7)*(row-7)
        temp_dxy_fg(index)=(A(index,:)-mu_fg) * (inv(covar_fg)* (A(index,:)-mu_fg)');
        temp_dxy_bg(index)=(A(index,:)-mu_bg) * (inv(covar_bg)* (A(index,:)-mu_bg)');
        g_fg(index)=1 / (1+ exp( temp_dxy_fg(index) - temp_dxy_bg(index) + alp_fg - alp_bg));
        g_bg(index)=1 / (1+ exp( temp_dxy_bg(index) - temp_dxy_fg(index) + alp_bg - alp_fg));

        if g_fg(index)>0.5
            predict(1,index)=1;
        end
        
    end
    
    predict_2d=reshape(predict,[col-7,row-7]);
    predict_2d=predict_2d';

    Errors_ML(k)=ProbOfError(predict_2d,Gt,Prior_FG,Prior_BG);
    Errors_ML(k)
end 



    figure;
    semilogx(alpha, Errors_Bay, '-r',alpha, Errors_MAP, '-g',alpha, Errors_ML, '-b');
    grid
    ylim([0.1460 0.1500])
    legend('Predictive','MAP','ML')
    xlabel('alpha');
    ylabel('probability of error');
