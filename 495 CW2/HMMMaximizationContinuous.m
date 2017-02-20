function [firstMoment, secondMoment, sequence, pi, A, E] = HMMMaximizationContinuous(Y,N,T,E,post_latent, post_transit)

    
    NbLatent    =   size(post_latent,1); 
    mu = [E.mu(1); E.mu(2)];
    pi = sum(post_latent(:,:,1),2)/sum(sum(post_latent(:,:,1),2));
    A = sum(sum(post_transit,4),3);
    A = A./ repmat(sum(A,2), 1, size(A,1));
    
    firstMoment = zeros(size(post_latent));
    secondMoment = zeros(size(post_latent));
    sequence = zeros(size(post_latent));
    sequence2 = zeros(size(post_latent));
    SumOfEZ =  sum(sum(post_latent,3),2);
    
    for t = 1:T
        sequence(:,:,t) = repmat(Y(:,t).',NbLatent, 1);  
        firstMoment(:,:,t) = post_latent(:,:,t).*sequence(:,:,t);
    end
        
    
    mu = repmat(mu,1,N);
    
    for t =1:T
        sequence2(:,:,t) = power(sequence(:,:,t)- mu,2);
    end 
    
    for t=1:T
        
        secondMoment(:,:,t) = post_latent(:,:,t).*sequence2(:,:,t);
    end
    E.mu = (sum(sum(firstMoment,3),2)./SumOfEZ);
    E.sigma2=sum(sum(secondMoment,3),2)/SumOfEZ;
    
    

end