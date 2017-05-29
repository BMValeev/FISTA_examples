function [y] = soft(x,T)
	eq=ge(abs(x),abs(T));
	y=eq.*sign(x).*(abs(x)-abs(T));
end
    
