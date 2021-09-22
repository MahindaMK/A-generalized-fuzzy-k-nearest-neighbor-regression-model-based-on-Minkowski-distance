function [predicted] = Md_FKNNreg(xtrain, ytrain, xtest, K, p, m)

% A generalized fuzzy k-nearest neighbor regression model based on
% Minkowski distance (Md-FKNNreg)
% INPUTS:
    % xtrain: train data is a n-by-m data matrix consisting of 
    % n patterns and m features(variables)
    % ytrain: n-dimensional output vector of Xtrain data  
    % Xtest: Test data is a D-by-m data matrix consisting of D
    % patterns and m features
    % ytest: D-dimensional output vector of Xtest data
    % K: Number of nearest neighbors to be selected
    % p: Parameter value for Minkowski distance
    % m: Scaling factor for fuzzy weights

% OUTPUTS:
    % predicted: Predicted y values for each test pattern in xtest
    

% Reference:
    % Kumbure, M.M. and Luukka, P. (2021) A generalized fuzzy k-nearest neighbor 
    % regression model based on Minkowski distance. Granular Computing

% Created by Mahinda Mailagaha Kumbure, 06/2021 
% Based on Keller's definition of the fuzzy k-nearest neighbor algorithm.


num_xtest  = size(xtest,1); % number test patterns

% scaling factor for fuzzy weights could be. 
    % m = 2;
    % m = 1.5;
    % m = 2.9;

if nargin<6
     m = 2;
end

% Initialization
predicted = zeros(num_xtest, length(K));

%% BEGIN Md-FKNNreg

% For each test pattern
for i=1:num_xtest
    
    % computer the Minkowski distances from test pattern to each train
    % pattern, (here, 'pdist2' MATLAB built-in function is used)
     mink_dis   = pdist2(xtrain, xtest(i,:),'minkowski',p);
     distances  = mink_dis';
    
    [~, indeces]    = sort(distances); % Sort the distances
     neighbor_index = indeces(1:K);    % Find the indeces of nearest neighbors
    
    % compute fuzzy weights:
        % though this weight calculation should be: 
        % weight = distances(neighbor_index).^(-2/(m-1)), 
        % but since we did not take sqrt above and the inverse 
        % 2th power the weights are: weight = sqrt(distances(neighbor_index)).^(-2/(m-1));
        % which is equaliavent to:

 	    weight = distances(neighbor_index).^(-1/(m-1));
 
 	    % Set the Inf (infite) weights, if there are any, to  1.
        if max(isinf(weight))
             warning(['Some of the weights are Inf for sample: '...
             num2str(i) '. These weights are set to 1.']);
             weight(isinf(weight)) = 1;
        end
    
    % find the predicted output 
        xtest_out = weight*ytrain(neighbor_index,:)/(sum(weight));
        predicted(i) = xtest_out;


end

