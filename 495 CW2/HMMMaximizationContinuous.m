function [pi, A, mu, sigma2] = HMMMaximizationContinuous(Y,N,T, mu, sigma2,post_latent, post_transit)

    
    NbLatent    =   size(post_latent,1); 
    
    pi = sum(post_latent(:,:,1),2)/sum(sum(post_latent(:,:,1),2));
    A = sum(sum(post_transit,4),3);
    A = A./ repmat(sum(A,2), 1, size(A,1));
    
    firstMoment = zeros(size(post_latent));
    secondMoment = zeros(size(post_latent));
    sequence = zeros(size(post_latent));
    sequence2 = zeros(size(post_latent));
    
    MU = repmat(mu.',1,N);
    
    SumOfEZ = sum(sum(post_latent,3),2);
    
    for t = 1:T
        sequence(:,:,t) = repmat(Y(:,t).',NbLatent, 1);  
        sequence2(:,:,t) = power(sequence(:,:,t)- MU,2);
        firstMoment(:,:,t) = post_latent(:,:,t).*sequence(:,:,t);
    end
        

    for t =1:T
       
        secondMoment(:,:,t) = post_latent(:,:,t).*sequence2(:,:,t);
    end
    mu = (sum(sum(firstMoment,3),2)./SumOfEZ).';
    sigma2=sum(sum(secondMoment,3),2)./SumOfEZ;
    
    

end