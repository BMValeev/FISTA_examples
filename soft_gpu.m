function [y] = soft_gpu(x,T)
%y=gpuArray.zeros(size(x));
%eq=gpuArray.zeros(size(x));
eq=ge(abs(x),abs(T));
y=eq.*sign(x).*(abs(x)-abs(T));
end
    