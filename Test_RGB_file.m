close all
clear all
clc
%% Loading file and dividing channels
file=imread('figure.jpg');
file=im2single(file);
imshow(file)
red_part=file(:,:,1);
blue_part=file(:,:,2);
green_part=file(:,:,3);
clear file

%% Generating convolution core
a=fspecial('gaussian',7,2);
%% Corrupting the channels
corrupted_red=conv2((red_part),a,'same'); 
corrupted_blue=conv2((blue_part),a,'same');
corrupted_green=conv2((green_part),a,'same');
%% Adding noise
corrupted_red=corrupted_red+randn(size(corrupted_red))*1e-3;
corrupted_blue=corrupted_blue+randn(size(corrupted_red))*1e-3;
corrupted_green=corrupted_green+randn(size(corrupted_red))*1e-3;
%% Showing result
figure
imshow(cat(3,(corrupted_red),(corrupted_blue),(corrupted_green)))
%% Loading in the GPU memory
a_gpu=gpuArray(a);
obj1_gpu=@(x) conv2(x,a_gpu,'same'); % Determine convolution operation
obj2_gpu=@(x) conv2(x,(a_gpu).','same');
%% DEconvolution
disp('GPU based algorithm started')
tic
disp('Red channel')
[red] = fista_gpu(gpuArray(corrupted_red),obj1_gpu,obj2_gpu,0.000001,3,100);
toc;
red=gather(red);
tic
disp('Blue channel')
[blue] = fista_gpu(gpuArray(corrupted_blue),obj1_gpu,obj2_gpu,0.000001,3,100);
toc
blue=gather(blue); 
tic;
disp('Green channel')
[green] = fista_gpu(gpuArray(corrupted_green),obj1_gpu,obj2_gpu,0.000001,3,100);
toc  
green=gather(green); 
%% Showing result
figure
imshow(cat(3,red,blue,green))
