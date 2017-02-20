clear all;
clc;

%%%%%%%%%%%%%%%%%%%
% DATA GENERATION %
%%%%%%%%%%%%%%%%%%%
N  = 250;      % number of sequences
T  = 10;       % length of the sequence

% Initial probability at time 1
pi = [0.50; 0.50];  

% Transition probability matrix for data generation
A  =    [0.40   0.60;
        0.60    0.40];

% Emission probability used to generate the data
E = [   1/6    1/6    1/6    1/6    1/6    1/6;      %p(x_t|y_{t}) 
        1/20   2/10   1/20   1/10   1/10    1/2];

% Y is the set of generated observations 
% S is the set of ground truth sequence of latent vectors     
[ Y, S ] = HmmGenerateData(N, T, pi, A, E ); 

%%%%%%%%%%%%%%%%
% EM ALGORITHM %
%%%%%%%%%%%%%%%%

% Initializing parameters
pi =    [0.52;  0.48];
A  =    [0.30   0.70;
         0.80   0.20];
E = [   0.16    0.16    0.16    0.16    0.16    0.20;  
        0.10    0.10    0.10    0.10    0.10    0.50];

% Running EM Algorithm
Aold = A;
tol = sum(sum(Aold.*Aold));

%for i = 1:2000
i=0;
while (tol>1e-6)
   
    Aold = A;
    [alpha, beta, c, post_latent, post_transit] = HMMExpectationDiscrete(Y,N,T,pi,A,E);
    [pi, A, E] = HMMMaximizationDiscrete(Y,N,T,post_latent, post_transit);
    
    i=i+1;
    tol = sum(sum(abs(Aold-A)));

end

A
E

[ESTTR,ESTEMIT] = hmmtrain(Y,A,E)

