function [x,J] = fista_normal_J(y,H,Ht,lambda,alpha,Nit)
	J = zeros(1, Nit); % Objective function
	x=Ht(y);
	y_k=x;
	t_1=1;
	for k = 1:Nit
		x_old=x;
		x=soft((y_k+(1/alpha)*Ht(y-H(y_k))), lambda/alpha);
		t_0=t_1;    
		t_1=0.5+sqrt(0.25+t_1^2);
		y_k=x+((t_0-1)/t_1)*(x-x_old);
		J(k)=0.5*norm(H(x)-y,2)^2+lambda*norm(x,1);
	end
end
