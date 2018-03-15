% Function that computes epsilon_final as equation 7.
% Inputs: adj (adjanceny matrix), p (prob of propagation), ei(initial shock)

function f = ep(adj,p,ei)  
[n ~] =size(adj);

s = ei;
for i = 1:(n-1)    
   s = s + ((p^i).*(adj^i - diag(diag(adj^i))))*ei;         
end

f = s;