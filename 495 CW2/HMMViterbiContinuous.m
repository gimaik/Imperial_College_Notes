function [decode] = HMMViterbiContinuous(Y,N,T,pi, A, mu, sigma2)
    
    NbLatent = size(A,1);
    decode = zeros(size(Y));
    delta =  zeros(NbLatent, N, T);
    delta_i = zeros(NbLatent, N);
    z       = zeros(size(A,1),N,T);
    
    % Generating the emission probability for the sequence
    z = normEmissionProba(N, T, Y,mu, sigma2);
    
    % Taking log of the pi and A
    pi = log2(pi);
    A = log2(A);
   
    % Initializing the delta for time t=1
    delta (:,:,1) = repmat(pi, 1,N) + z(:,:,1);
    
    % Filling up the delta table for time t=2:T
    for k = 1: NbLatent
       for t=2: T
           for n= 1: N
               delta_i(k,n) = max(A(:,k) + delta(:,n,t-1));      
           end
           delta(:,:,t) =  delta_i+z(:,:,t);
       end
    end
    
    % Return the decoding
    for t=1:T
        [M, I] = max(delta(:,:,t));
        decode(:,t) = I.';
    end
    
end