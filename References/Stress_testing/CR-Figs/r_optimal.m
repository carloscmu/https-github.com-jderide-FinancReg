% Function that computes r_optimal as in equation 8.
% Inputs: x (policy vector), s (vector in footnote 4), lambda, ub (upper bound distribution)

function r = r_optimal(x,s,lambda,ub,a0,a1)  
r = zeros(length(x),1); %initial values for r
ee = ub/2; % assuming uniform distribution[0,ub] for epsilon_I
for i = 1:length(x)    
   if x(i) < (lambda/(1-s(i)*ub))
    r(i) = (lambda/(1-s(i)*ub));
   elseif x(i) >= (lambda/(1-s(i)*ub)) && x(i) < (a0/(a1*(1-ee))) 
    r(i) = x(i);
   end         
end

