function [x,J] = ista_fns(y,H,Ht,lambda,alpha,Nit,real)
J = zeros(1, Nit); 
x = Ht(y); 
T = lambda/(2*alpha);
Hreal=H(real);
tic 
for k = 1:Nit
Hx = H(x);
J(k)=0.5*norm(H(x)-y,2)^2+lambda*norm(x,1);
x = soft(x + (Ht(y - Hx))/alpha, T);
end
