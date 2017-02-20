clear all;
clc;

%%%%%%%%%%%%%%%%%%%
% DATA GENERATION %
%%%%%%%%%%%%%%%%%%%
N  = 6;         % number of sequences
T  = 2;        % length of the sequence

% Initial Probability at time 1
pi = [0.5; 0.5]; 

% Transition probability matrix for data generation
A  = [0.4 0.6 ; 0.4 0.6 ];       

% One Dimensional Gaussians 
E.mu    =[0.1 0.5]; %%the means of each of the Gaussians
E.sigma2=[0.4 0.8]; %%the variances

% Y is the set of generated observations 
% S is the set of ground truth sequence of latent vectors 
[ Y, S ] = HmmGenerateData(N, T, pi, A, E, 'normal'); 


%%%%%%%%%%%%%%%%
% EM ALGORITHM %
%%%%%%%%%%%%%%%%
% Initializing parameters
pi = [0.25; 0.75]; 
A  = [0.2 0.8 ; 0.7 0.3 ];       

% One Dimensional Gaussians 
E.mu    =[0.7 0.6]; %%the means of each of the Gaussians
E.sigma2=[0.5 0.4]; %%the variances

% Running EM Algorithm
for i = 1:1000
    i
    
    [post_latent, post_transit,z] = HMMExpectationContinuous (Y,N,T,pi,A,E);   
    post_latent(:,:,:)
    
    
    
    [firstMoment, secondMoment, sequence, pi, A, E] = HMMMaximizationContinuous(Y,N,T,E,post_latent, post_transit);
    %pi
    %A 
    %E.mu
    %E.sigma2
end

pi;
A;
E.mu;
E.sigma2;