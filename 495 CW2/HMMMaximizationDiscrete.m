function [pi, A, E] = HMMMaximizationDiscrete(Y,N,T,post_latent, post_transit)

    
    NbLatent    =   size(post_latent,1); 
    NbObserved  =   max(max(Y));

    B = zeros(NbObserved, size(post_latent,1),size(post_latent,2),size(post_latent,3));
    E = zeros(NbLatent, NbObserved);
    
    pi = sum(post_latent(:,:,1),2)/sum(sum(post_latent(:,:,1),2));

    A = sum(sum(post_transit,4),3);
    A = A./ repmat(sum(A,2), 1, size(A,1));
    
    for y = 1:NbObserved
        for t = 1:T
            Obs = repmat(Y(:,t).',size(post_latent,1),1);
            Indicator = repmat(Y(:,t).'==y,size(post_latent,1),1);
            B(y,:,:,t) = post_latent(:,:,t).*Indicator;
        end
    end
    
    E = sum(sum(B,4),3).';
    E = E./repmat(sum(E,2),1,NbObserved);
    


end