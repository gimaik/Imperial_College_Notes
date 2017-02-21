clear all;
clc;

%%%%%%%%%%%%%%%%%%%
% DATA GENERATION %
%%%%%%%%%%%%%%%%%%%
N  = 300;      % number of sequences
T  = 10;       % length of the sequence

% Initial probability at time 1
pi = [0.50; 0.50];  

% Transition probability matrix for data generation
A  =    [0.40   0.60;
        0.60    0.40];

% Emission probability used to generate the data
E = [   1/6    1/6    1/6    1/6    1/6    1/6;      
        1/20   2/10   1/20   1/10   1/10    1/2];

% Y is the set of generated observations 
% S is the set of ground truth sequence of latent vectors     
[ Y, S ] = HmmGenerateData(N, T, pi, A, E ); 

%%%%%%%%%%%%%%%%
% EM ALGORITHM %
%%%%%%%%%%%%%%%%

%Initializing parameters
pi =    [0.52;  0.48];
A  =    [0.30   0.70;
         0.80   0.20];
E = [   0.16    0.16    0.16    0.16    0.16    0.20;  
        0.10    0.10    0.10    0.10    0.10    0.50];

% Running EM Algorithm
[pi_e, A_e, E_e, decode] = HMM(Y,N,T,pi,A,E, 1e-6, 200, 'discrete');

accuracy = sum(sum(decode==S))/(N*T);
pi_e
A_e
E_e

pi =    [0.52;  0.48];
A  =    [0.30   0.70;
         0.80   0.20];
E = [   0.16    0.16    0.16    0.16    0.16    0.20;  
        0.10    0.10    0.10    0.10    0.10    0.50];

%[ESTTR,ESTEMIT] = hmmtrain(Y,A,E)

