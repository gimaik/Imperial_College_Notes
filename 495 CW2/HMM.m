function [pi, A, E, decode] = HMM(Y,N,T,pi,A,E, tol, iter, type)


    % This function combines both the E-step and M-step for both the
    % discrete case and normal distribution case. Also, the EM algorithm
    % will stop under two criteria, either there's convergence in the
    % transitions probability matrix or if the maximum number of iteration
    % is reached.


    % Initializing the stopping criterion
    norm = sum(sum(A.*A));
    i=0;
    
    
    % The EM for the discrete case
    if strcmp(type, 'discrete')

        while (norm>tol && i <iter)
            Aold = A;
            [post_latent, post_transit] = HMMExpectationDiscrete(Y,N,T,pi,A,E);
            [pi, A, E] = HMMMaximizationDiscrete(Y,N,T,post_latent, post_transit);

            norm = sum(sum(abs(Aold-A)));
            i=i+1;
        end

        decode = HMMViterbiDiscrete(Y,N,T,pi, A, E);

    % The EM for the continuous case
    elseif strcmp(type, 'continuous')

        norm = sum(sum(A.*A));
        mu = E.mu;
        sigma2 = E.sigma2;

        while (norm>tol && i <iter)
            Aold = A;
            [post_latent, post_transit] = HMMExpectationContinuous (Y,N,T,pi,A,mu,sigma2);
            [pi, A, mu, sigma2] = HMMMaximizationContinuous(Y,N,T, mu, sigma2,post_latent, post_transit);

            norm = sum(sum(abs(Aold-A)));
            i=i+1;
        end

        E.mu=mu;
        E.sigma2=sigma2;
        decode = HMMViterbiContinuous(Y,N,T,pi, A, mu, sigma2);

    end
    
end

