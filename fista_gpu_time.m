function [x,time] = fista_gpu_time(y,H,Ht,lambda,alpha,Nit)
x=Ht(y);
y_k=x;
t_1=1;
T=lambda/alpha;
time=zeros(Nit,1);
tic
for k = 1:Nit
x_old=x;
x=soft_gpu((y_k+(1/alpha)*Ht(y-H(y_k))), T);
t_0=t_1-1;    
t_1=0.5+sqrt(0.25+t_1^2);
y_k=x+(t_0/t_1)*(x-x_old);
time(k)=toc;
end

end