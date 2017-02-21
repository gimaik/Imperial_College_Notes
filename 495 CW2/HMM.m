function [pi, A, E, decode] = HMM(Y,N,T,pi,A,E, tol, iter, type)

if strcmp(type, 'discrete')
    i=0;
    norm = sum(sum(A.*A));

    while (norm>tol && i <iter)
        Aold = A;
        [post_latent, post_transit] = HMMExpectationDiscrete(Y,N,T,pi,A,E);
        [pi, A, E] = HMMMaximizationDiscrete(Y,N,T,post_latent, post_transit);
    
        i=i+1;
        tol = sum(sum(abs(Aold-A)));
    end
    
    decode = HMMViterbiDiscrete(Y,N,T,pi, A, E);

    
elseif strcmp(type, 'normal')
    i=0;
    norm = sum(sum(A.*A));
    
    mu = E.mu;
    sigma2 = E.sigma2;

    while (norm>tol && i <iter)
        Aold = A;
        [post_latent, post_transit] = HMMExpectationContinuous (Y,N,T,pi,A,mu,sigma2);
        [pi, A, mu, sigma2] = HMMMaximizationContinuous(Y,N,T, mu, sigma2,post_latent, post_transit);
       
    
        i=i+1;
        tol = sum(sum(abs(Aold-A)));
    end
    
    decode = HMMViterbiContinuous(Y,N,T,pi, A, E, mu, sigma2);
    
    
end
    
    
end

