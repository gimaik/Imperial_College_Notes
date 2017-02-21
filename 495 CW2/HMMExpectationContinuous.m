function [post_latent, post_transit,z] = HMMExpectationContinuous (Y,N,T,pi,A,mu,sigma2)

    % This function computes the posterior for the latent variable and the
    % smoothed transition probability. These two posterior probabilities
    % will be used in the EM algorithm to estimate the parameters of the
    % model. This just replace the emission probability of the discrete
    % case with the emission probability of the normal distribution.

    % Defining constants used
    NbLatent = size(A,1);
    
    % Initializing the arrays used in the function
    alpha   = zeros(size(A,1),N,T);
    beta    = zeros(size(A,1),N,T);
    z       = zeros(size(A,1),N,T);
    c       = zeros(size(A,1), N,T);
    post_latent     = zeros(size(A,1),N,T);
    post_transit    = zeros(size(A,1),size(A,2),N,T);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GENERATING THE NORMAL EMISSION PROBABILITY FOR SEQUENCE Y %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    z = normEmissionProba(N, T, Y,mu, sqrt(sigma2));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % THE FORWARD PASS: Computation of alpha %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Computing alpha(t_1)   
    alpha(:,:,1)    = z(:,:,1).*repmat(pi,1,N);
    c(:,:,1)        = repmat(sum(alpha(:,:,1),1),NbLatent,1);
    alpha(:,:,1)    = alpha(:,:,1)./c(:,:,1); 
    
    
    % Computing alpha(t_2 to t_T)
    for t=2:T
        alpha(:,:,t)  = z(:,:,t).*(A.'*alpha(:,:,t-1));
        c(:,:,t) = repmat(sum(alpha(:,:,t), 1),NbLatent,1);
        alpha(:,:,t) = alpha(:,:,t)./c(:,:,t);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % THE BACKWARD PASS: Computation of beta %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Computing beta(T) 
    beta(:,:,T) = 1;

    % Computing beta (T-1:1)
    for t=T-1:-1:1
        beta(:,:,t) = A*(z(:,:,t+1).*beta(:,:,t+1));
        beta(:,:,t) = beta(:,:,t)./c(:,:,t+1);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % COMPUTATION OF THE POSTERIORS %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Computing the posterior for the latent variable P(z(t) | X(1:T))
    for t = 1:T
        post_latent(:,:,t) = alpha(:,:,t).*beta(:,:,t);
    end
   
    % Computing the smoothed transition probability P[z(t), z(t-1)| X(1:T)]
    for t = 2:T
        proba_emit = z(:,:,t);
        
        for n = 1: N
           x = proba_emit(:,n);
           a = alpha(:,n,t-1);
           b = beta(:,n,t);
           
           post_transit (:,:, n,t) = A.*(a*(x.*b).')./repmat(c(:,n,t),1,NbLatent);
            
        end
    end
        

end
