function [z] = normEmissionProba(N, T, Y,mu,sigma2)

    %This function takes the total sample in each time point and generate
    %the normal values for all possible latent states. 
    % x will be the value of all the sequences at a fixed time point
    % Note that matlab normpdf function takes in the mu and the standard
    % deviation.
    
    sigma2 = sqrt(sigma2);
    nbLatent = max(size(mu));
    z = zeros(nbLatent, N, T);
   
    for t = 1: T
        for k = 1: nbLatent
               
            z(k,:,t) = normpdf(Y(:,t), mu(k), sigma2(k));
            
        end
    end

end