close all 
clear all
clc

load('TrainingSamplesDCT_8_new.mat')

%% Problem a
[row_f,col_f] = size(TrainsampleDCT_FG);
FG_index=zeros(1,row_f);
[row_b,col_b] = size(TrainsampleDCT_BG);
BG_index=zeros(1,row_b);

Prior_FG=row_f/(row_f+row_b);          % Calculate Prior
Prior_BG=row_b/(row_f+row_b);

figure;
str = {'cheetah','grass'};
bar([Prior_FG,Prior_BG])
title('Histogram of Prior Probabilities')
xlabel('Class')
ylabel('Prior')
set(gca, 'XTickLabel',str), 


disp('Prior Probability (cheetah):')
disp(Prior_FG)
disp('Prior Probability (grass):')
disp(Prior_BG)

%% Problem b
% calculate for each column's mean
mu_fg=sum(TrainsampleDCT_FG)/row_f;
mu_bg=sum(TrainsampleDCT_BG)/row_b;
% calculate for each column's std
std_fg=std(TrainsampleDCT_FG);
std_bg=std(TrainsampleDCT_BG);


for i=1:col_b
    % create Gaussian function for each column
    % I use overall 6 std to be my x-axis
    margin_fg_axis(:,i)=[(mu_fg(i)-3*std_fg(i)):(std_fg(i)/50):(mu_fg(i)+3*std_fg(i))];
    margin_FG(:,i)=normpdf(margin_fg_axis(:,i),mu_fg(i),std_fg(i));    
    
    margin_bg_axis(:,i)=[(mu_bg(i)-3*std_bg(i)):(std_bg(i)/50):(mu_bg(i)+3*std_bg(i))];
    margin_BG(:,i)=normpdf(margin_bg_axis(:,i),mu_bg(i),std_bg(i)); 

end

% plot the marginal density into 8*8 plots with the dimension from 1-64
for i=1:2
    figure;
    for j=1:32
        subplot(4,8,j)
        plot(margin_fg_axis(:,(i-1)*32+j),margin_FG(:,(i-1)*32+j),'-r',margin_bg_axis(:,(i-1)*32+j),margin_BG(:,(i-1)*32+j),'-b')
        grid on
        title(['Feature ',num2str((i-1)*32+j)])
    end
end
% plot the best 8 feature marginal densities
best = [1,11,14,23,25,26,32,40];
figure;
for i=1:8
    subplot(2,4,i)
    plot(margin_fg_axis(:,best(i)),margin_FG(:,best(i)),'-r',margin_bg_axis(:,best(i)),margin_BG(:,best(i)),'-b')
    grid on
    title(['Feature ',num2str(best(i))])
 
end
% plot the worst 8 feature marginal densities
worst=[3,4,5,59,60,62,63,64];
figure;
for i=1:8
    subplot(2,4,i)
    plot(margin_fg_axis(:,worst(i)),margin_FG(:,worst(i)),'-r',margin_bg_axis(:,worst(i)),margin_BG(:,worst(i)),'-b')
    grid on
    title(['Feature ',num2str(worst(i))])
 
end
%% Problem c
% Load Test sample 
Img_ori=imread('cheetah.bmp');            %Use zero padding for edge & corner
Img = padarray(Img_ori,[7 7],'symmetric','pre');     %classified pixel:right bottom

Img=im2double(Img);
[row,col]=size(Img);

% build 64-dimension Gaussian
covar_fg=cov(TrainsampleDCT_FG);
covar_bg=cov(TrainsampleDCT_BG);

% Load Zig-Zag pattern
Zigzag=load('Zig-Zag Pattern.txt');
Zigzag=Zigzag+1;

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
    index=index+1;
end

predict_2d=reshape(predict,[col-7,row-7]);
predict_2d=predict_2d';

figure;
imagesc(predict_2d);
title('Prediction')
colormap(gray(255));

%% Calculate the Probability of Error
% Load Ground Truth
Gt=imread('cheetah_mask.bmp');
Gt=im2double(Gt);

Gt_FG=0;
Gt_BG=0;
for i=1:row-7
    for j=1:col-7
        if Gt(i,j)==1 
            Gt_FG=Gt_FG+1;
        end
        if Gt(i,j)==0 
            Gt_BG=Gt_BG+1;
        end
    end
end

Errors_FG=0;
Errors_BG=0;
for i=1:row-7
    for j=1:col-7
        if Gt(i,j)==1 && predict_2d(i,j)==0 % FG pixels, misclassifcied as BG
            Errors_FG=Errors_FG+1;
        end
        if Gt(i,j)==0 && predict_2d(i,j)==1
            Errors_BG=Errors_BG+1;
        end
    end
end

Errors_FG_p=Errors_FG/Gt_FG;    %Type II (False Negative)
Errors_BG_p=Errors_BG/Gt_BG;    %Type I  (False Positive)

Errors=Errors_FG_p*Prior_FG + Errors_BG_p*Prior_BG;

disp('Error Rate (%):')
disp(Errors*100)
