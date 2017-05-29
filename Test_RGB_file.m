close all
clear all
clc
% constants
max_it=100;
iterations=(1:max_it);
lambda=0.000001;
alpha=3;
noise=1e-3;
conv_kernel=7;

%% Loading file and dividing channels
	file=imread('figure.jpg');
	file=im2single(file);
	imshow(file)
	red_part=file(:,:,1);
	blue_part=file(:,:,2);
	green_part=file(:,:,3);
	clear file

%% Generating convolution core
	a=fspecial('gaussian',conv_kernel,2);
%% Corrupting the channels
	corrupted_red=conv2((red_part),a,'same'); 
	corrupted_blue=conv2((blue_part),a,'same');
	corrupted_green=conv2((green_part),a,'same');
%% Adding noise
	corrupted_red=corrupted_red+randn(size(corrupted_red))*noise;
	corrupted_blue=corrupted_blue+randn(size(corrupted_red))*noise;
	corrupted_green=corrupted_green+randn(size(corrupted_red))*noise;
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
	[red] = fista_gpu(gpuArray(corrupted_red),obj1_gpu,obj2_gpu,lambda,alpha,max_it);
toc;
	red=gather(red);
tic
disp('Blue channel')
	[blue] = fista_gpu(gpuArray(corrupted_blue),obj1_gpu,obj2_gpu,lambda,alpha,max_it);
toc
	blue=gather(blue); 
tic;
disp('Green channel')
	[green] = fista_gpu(gpuArray(corrupted_green),obj1_gpu,obj2_gpu,lambda,alpha,max_it);
toc  
	green=gather(green); 
%% Showing result
		figure
		imshow(cat(3,red,blue,green))
