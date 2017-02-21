function [decode] = HMMViterbiDiscrete(Y,N,T,pi, A, E)
    
    NbLatent = size(A,1);
    decode = zeros(size(Y));
    delta =  zeros(NbLatent, N, T);
    delta_i = zeros(NbLatent, N);
    
    pi = log2(pi);
    A = log2(A);
    E=log2(E);
    
    % Initializing the delta
    delta (:,:,1) = repmat(pi, 1,N) + E(:, Y(:,1));
    
    
    for k = 1: NbLatent
       for t=2: T
           for n= 1: N
               delta_i(k,n) = max(A(:,k) + delta(:,n,t-1));      
           end
           delta(:,:,t) =  delta_i+E(:, Y(:,t));
       end
    end

    for t=1:T
        [M, I] = max(delta(:,:,t));
        decode(:,t) = I.';
    end
    
    
    
end