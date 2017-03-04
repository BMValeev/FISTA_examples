close all
clear all
tic
file=imread('lena512.bmp');
filegray1=im2double(file);
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
[x,J] = fista_normal_J(corrupted,obj1,obj2,0.000001,2,500);
toc
tic
[x1,J1] = ista_fns(corrupted,obj1,obj2,0.000001,2,500,filegray1);
toc
figure
subplot(1,3,1);
semilogy(J,'b','LineWidth',2);hold on
semilogy(J1,'r','LineWidth',2)
xlabel('Итерации','FontSize',14)
ylabel('Норма невязки, абсолютная','FontSize',14)
title('Сходимость алгоритмов \lambda=10^{-6} \alpha=2','FontSize',14)
grid on
set(gca,'FontSize',14)
axis([0 500 1e-3 1e1]);
hleg=legend(['FISTA (Улучшенный алгоритм)'],['ISTA (Базовый алгоритм)'])
set(hleg,'FontSize',14)


tic
[x,J_1] = fista_normal_J(corrupted,obj1,obj2,0.0001,2,500);
toc
tic
[x,J_2] = fista_normal_J(corrupted,obj1,obj2,0.00001,2,500);
toc
tic
[x,J_3] = fista_normal_J(corrupted,obj1,obj2,0.000001,2,500);
toc
subplot(1,3,2);
semilogy(J_1,'b','LineWidth',2);hold on
semilogy(J_2,'r','LineWidth',2)
semilogy(J_3,'k','LineWidth',2)
xlabel('Итерации','FontSize',14)
axis([0 500  1e-3 1e1]);
ylabel('Норма невязки, абсолютная','FontSize',14)
title('Сходимость алгоритма от \lambda при \alpha=2','FontSize',14)
grid on
set(gca,'FontSize',14)
hleg=legend(['FISTA при \lambda=10^{-4})'],['FISTA при \lambda=10^{-5})'],['FISTA при \lambda=10^{-6})'])
set(hleg,'FontSize',14)


tic
[x,J_11] = fista_normal_J(corrupted,obj1,obj2,0.000001,2,500);
toc
tic
[x,J_22] = fista_normal_J(corrupted,obj1,obj2,0.000001,10,500);
toc
tic
[x,J_33] = fista_normal_J(corrupted,obj1,obj2,0.000001,50,500);
toc
subplot(1,3,3);
semilogy(J_11,'b','LineWidth',2);hold on
semilogy(J_22,'r','LineWidth',2)
semilogy(J_33,'k','LineWidth',2)
xlabel('Итерации','FontSize',14)
axis([0 500  1e-3 1e1]);
ylabel('Норма невязки, абсолютная','FontSize',14)
title('Сходимость алгоритма от \alpha при \lambda=10^{-4}','FontSize',14)
grid on
set(gca,'FontSize',14)
hleg=legend(['FISTA при \alpha=2)'],['FISTA при \alpha=10)'],['FISTA при \alpha=50)'])
set(hleg,'FontSize',14)