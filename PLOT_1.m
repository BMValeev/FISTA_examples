close all
clear all
tic
% constants
max_it=1000;
iterations=(1:max_it);
lambda=0.000001;
alpha=2;
noise=1e-3;
conv_kernel=7;

% Loading Image from file
file=imread('lena512.bmp');
file=im2double(file);
% Concatenating and increasing size
filegray1=cat(2,cat(1,file,file),cat(1,file,file));
filegray1=cat(2,cat(1,filegray1,filegray1),cat(1,filegray1,filegray1));
size(filegray1)


% Definition of the convolutional kernel
a=fspecial('gaussian',conv_kernel,2);
obj1=@(x) conv2(x,a,'same');
obj2=@(x) conv2(x,a.','same');
corrupted=conv2(filegray1,a,'same');
corrupted=corrupted+randn(size(corrupted))*noise;

% Definition of the convolutional kernel in GPU memory
a_gpu=gpuArray(a);
obj1_gpu=@(x) conv2(x,a_gpu,'same');
obj2_gpu=@(x) conv2(x,(a_gpu).','same');
corrupted_gpu=conv2(gpuArray(filegray1),a_gpu,'same');
corrupted_gpu=corrupted_gpu+gpuArray.randn(size(corrupted_gpu))*noise;

disp('CPU based algorithm started')
tic
	[x,timeline_cpu] = fista_normal_time(corrupted,obj1,obj2,lambda,alpha,max_it);
toc
disp('Algorithm on GPU started')
tic
	[x1,timeline_gpu] = fista_gpu_time(corrupted_gpu,obj1_gpu,obj2_gpu,lambda,alpha,max_it);
toc 


	%Plotting
	semilogy(iterations,timeline_cpu,'b','LineWidth',2); hold on; grid on;
	semilogy(iterations,timeline_gpu,'r','LineWidth',2); set(gca,'FontSize',14);
	title('Скорость работы по итерациям','FontSize',14);
	xlabel('Итерации','FontSize',14)
	ylabel('Время работы','FontSize',14)
	hleg=legend(['Алгоритм реализованный на CPU'],['Алгоритм реализованный на GPU'])
	set(hleg,'FontSize',14 )
