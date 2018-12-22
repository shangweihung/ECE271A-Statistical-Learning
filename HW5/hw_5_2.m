close all 
clear all
clc

load('TrainingSamplesDCT_8_new.mat')

[row_f,col_f] = size(TrainsampleDCT_FG);
FG_index=zeros(1,row_f);
[row_b,col_b] = size(TrainsampleDCT_BG);
BG_index=zeros(1,row_b);

Prior_FG=row_f/(row_f+row_b);          % Calculate Prior
Prior_BG=row_b/(row_f+row_b);


% Load Test sample 
Img_ori=imread('cheetah.bmp');            %Use zero padding for edge & corner
% Img = padarray(Img_ori,[7 7],'symmetric','pre');     %classified pixel:right bottom
Img=im2double(Img_ori);
[row,col]=size(Img);

% Load Zig-Zag pattern
Zigzag=load('Zig-Zag Pattern.txt');
Zigzag=Zigzag+1;

%% Perform 8*8 DCT
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

% load Ground Truth
Gt=imread('cheetah_mask.bmp');
Gt=im2double(Gt);


dimen=64;
c=[1,2,4,8,16,32];
%% Learn 5 mixtures of C=8 components

pi_BG = cell(6,1);
mu_BG = cell(6,1);
var_BG= cell(6,1);
pi_FG = cell(6,1);
mu_FG = cell(6,1);
var_FG= cell(6,1);

% background
for num = 1:1:6
    num
    % paramter initialization
    pi_init = rand(1, c(num));
    pi_init = pi_init / sum(pi_init);
    mu_init = rand(c(num), dimen);
    var_init = rand(c(num),dimen);
    var_init(var_init < 0.001) = 0.001;
    [mu, var,prior] = EM(dimen, mu_init, var_init, pi_init,c(num), TrainsampleDCT_BG );
      
    % save the EM results in cell arrays
    pi_BG{num} = prior;
    mu_BG{num} = mu;
    var_BG{num} = var;
end
disp('background')
% foreground
for num = 1:1:6
    num
   % parameter initialization
    pi_init = rand(1, c(num));
    pi_init = pi_init / sum(pi_init);
    mu_init = rand(c(num), dimen);
    var_init = rand(c(num),dimen);
    var_init(var_init < 0.001) = 0.001;
        
   [mu, var,prior] = EM (dimen, mu_init, var_init, pi_init,c(num), TrainsampleDCT_FG );
       
   % save the EM results in cell arrays
   pi_FG{num} = prior;
   mu_FG{num} = mu;
   var_FG{num} = var;
end
disp('foreground')



Errorrate=zeros(6,11);
% p index for # of components [1,2,4,8,16,32]
for p = 1:1:6
    p
    cnt2=1;
    for dim = [1,2,4,8,16,24,32,40,48,56,64]
        predict=zeros(1,(col-7)*(row-7));    

        for index=1:(col-7)*(row-7)
             Prob_BG=0;
             for i=1:c(p)
                 Prob_BG = Prob_BG + mvnpdf(A(index,1:dim),mu_BG{p}(i,1:dim),diag(var_BG{p}(i,1:dim)))* pi_BG{p}(i);    
             end
             Prob_FG=0;
             for i=1:c(p)
                 Prob_FG = Prob_FG + mvnpdf(A(index,1:dim),mu_FG{p}(i,1:dim),diag(var_FG{p}(i,1:dim)))* pi_FG{p}(i);    
             end

             if (Prob_BG * Prior_BG < Prob_FG * Prior_FG)
                 predict(1,index)=1;
             end

        end

        predict_2d=reshape(predict,[col-7,row-7]);
        predict_2d=predict_2d';

        result=zeros(255,270);
        result(1:row-7,1:col-7)=predict_2d;

        Errorrate(p,cnt2)=ProbOfError(result,Gt,Prior_FG,Prior_BG);
        dim
        Errorrate(p,cnt2)
                                    
        cnt2=cnt2+1;
    end  
    

end

figure
for i=1:1:6
    plot([1,2,4,8,16,24,32,40,48,56,64],Errorrate(i,:),'*-')
    hold on
end
legend('C = 1','C = 2','C = 4','C = 8','C = 16','C = 32')
ylim([0.04 0.09])
title('Prob. Of Error VS. # of features')
    