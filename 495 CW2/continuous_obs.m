clear all;
clc;

%%%%%%%%%%%%%%%%%%%
% DATA GENERATION %
%%%%%%%%%%%%%%%%%%%
N  = 100;         % number of sequences
T  = 500;        % length of the sequence

% Initial Probability at time 1
pi = [0.7; 0.3]; 

% Transition probability matrix for data generation
A  = [0.30 0.70 ; 0.40 0.60 ];       

% One Dimensional Gaussians 
E.mu    =[0.1 5.0]; %%the means of each of the Gaussians
E.sigma2=[0.4 0.8]; %%the variances

% Y is the set of generated observations 
% S is the set of ground truth sequence of latent vectors 
[ Y, S ] = HmmGenerateData(N, T, pi, A, E, 'normal'); 


%%%%%%%%%%%%%%%%
% EM ALGORITHM %
%%%%%%%%%%%%%%%%
% Initializing parameters
pi = [0.51; 0.49]; 
A  = [0.40 0.60 ; 0.40 0.60 ];        

% One Dimensional Gaussians 
E.mu    =[1 5.0]; %%the means of each of the Gaussians
E.sigma2=[3 50]; %%the variances


% Running EM Algorithm
[pi_e, A_e, E_e, decode] = HMM(Y,N,T,pi,A,E, 1e-9, 1000, 'continuous');
accuracy = sum(sum(decode==S))/(N*T);

