close all
clear all
tic
file=imread('lena512.bmp');
file=im2double(file);
filegray1=cat(2,cat(1,file,file),cat(1,file,file));
filegray1=cat(2,cat(1,filegray1,filegray1),cat(1,filegray1,filegray1));
size(filegray1)
a=fspecial('gaussian',7,2);
obj1=@(x) conv2(x,a,'same');
obj2=@(x) conv2(x,a.','same');
a_gpu=gpuArray(a);
obj1_gpu=@(x) conv2(x,a_gpu,'same');
obj2_gpu=@(x) conv2(x,(a_gpu).','same');
corrupted=conv2(filegray1,a,'same');
corrupted_gpu=conv2(gpuArray(filegray1),a_gpu,'same');
corrupted=corrupted+randn(size(corrupted))*1e-3;
corrupted_gpu=corrupted_gpu+gpuArray.randn(size(corrupted_gpu))*1e-3;

disp('CPU based algorithm started')
tic
[x,timeline_cpu] = fista_normal_time(corrupted,obj1,obj2,0.000001,2,1000);
toc
disp('Algorithm on GPU started')

tic
[x1,timeline_gpu] = fista_gpu_time(corrupted_gpu,obj1_gpu,obj2_gpu,0.000001,2,1000);
toc 

iterations=(1:1000);
semilogy(iterations,timeline_cpu,'b','LineWidth',2); hold on; grid on;
semilogy(iterations,timeline_gpu,'r','LineWidth',2); set(gca,'FontSize',14);
title('Скорость работы по итерациям','FontSize',14);
xlabel('Итерации','FontSize',14)
ylabel('Время работы','FontSize',14)
hleg=legend(['Алгоритм реализованный на CPU'],['Алгоритм реализованный на GPU'])
set(hleg,'FontSize',14 )
