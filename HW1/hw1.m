close all 
clear all
clc

load('TrainingSamplesDCT_8.mat')

%% Problem a
[row_f,col_f] = size(TrainsampleDCT_FG);
FG_index=zeros(1,row_f);
[row_b,col_b] = size(TrainsampleDCT_BG);
BG_index=zeros(1,row_b);

Prior_FG=row_f/(row_f+row_b);          % Calculate Prior
Prior_BG=row_b/(row_f+row_b);
disp('Prior Probability (cheetah):')
disp(Prior_FG)
disp('Prior Probability (grass):')
disp(Prior_BG)
%% Problem b
for i=1:row_f
    [maximum,index]=sort(abs(TrainsampleDCT_FG(i,:)),'descend');
    FG_index(i)=index(2);              %choose 2nd largest coefficient
end

for i=1:row_b
    [maximum,index]=sort(abs(TrainsampleDCT_BG(i,:)),'descend');
    BG_index(i)=index(2);   
end

binranges=0.5:1:63.5;
[bincounts_FG] = histc(FG_index,binranges);
[bincounts_BG] = histc(BG_index,binranges);
bincounts_FG_p=bincounts_FG/sum(bincounts_FG);
bincounts_BG_p=bincounts_BG/sum(bincounts_BG);

figure;
bar(binranges,bincounts_FG_p,'histc')
xlabel('Index of 2nd Largest Coefficient')
ylabel('P(X|cheetah)')
title('Histogram (Foreground training samples)')
figure;
bar(binranges,bincounts_BG_p,'histc')
xlabel('Index of 2nd Largest Coefficient')
ylabel('P(X|grass)')
title('Histogram (Background training samples)')

%% Problem c
% Load Test sample 
Img_ori=imread('cheetah.bmp');            %Use zero padding for edge & corner
Img = padarray(Img_ori,[7 7],'post');     %classified pixel:left top
Img=im2double(Img);
[row,col]=size(Img);

% Load Zig-Zag pattern
Zigzag=load('Zig-Zag Pattern.txt');
Zigzag=Zigzag+1;

Map_index=zeros(row-7,col-7);

for i=1:row-7
    for j=1:col-7
        Field=Img(i:i+7,j:j+7);
        DCT=dct2(Field);
        DCT_64(Zigzag)=DCT;             %assign value following zigzag pattern
        [value,index]=sort(abs(DCT_64),'descend'); %ignore the 1st coefficient
        Map_index(i,j)=index(2);          
    end
end

A=zeros(row-7,col-7);
for i=1:row-7                           %Do noy predict padding area
    for j=1:col-7                       %Bayesian Decision Rule
        if bincounts_FG_p(1,Map_index(i,j))*Prior_FG > bincounts_BG_p(1,Map_index(i,j))*Prior_BG
            A(i,j)=1;
        else
            A(i,j)=0;
        end
        
    end
end
figure;
imagesc(A);
title('Prediction')
colormap(gray(255));

%% Problem d
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
        if Gt(i,j)==1 && A(i,j)==0 % FG pixels, misclassifcied as BG
            Errors_FG=Errors_FG+1;
        end
        if Gt(i,j)==0 && A(i,j)==1
            Errors_BG=Errors_BG+1;
        end
    end
end

Errors_FG_p=Errors_FG/(Gt_FG);    %Type II (False Negative)
Errors_BG_p=Errors_BG/(Gt_BG);    %Type I (False Positive)

Errors=Errors_FG_p*Prior_FG + Errors_BG_p*Prior_BG;

disp('Error Rate (%):')
disp(Errors*100)
