function [predicted] = FKNNreg(xtrain, ytrain, xtest, K, fuzzy, m)

% Fuzzy k-nearest neighbor regression model (FKNNreg)
% INPUTS:
    % xtrain: train data is a n-by-m data matrix consisting of 
    % n patterns and m features(variables)
    % ytrain: n-dimensional output vector of Xtrain data  
    % Xtest: Test data is a D-by-m data matrix consisting of D
    % patterns and m features
    % ytest: D-dimensional output vector of Xtest data
    % K: Number of nearest neighbors to be selected
    % m: Scaling factor for fuzzy weights (e.g., 1.5, 2)
    % if fuzzy = 1, then the function is FKNNreg, and if fuzzy = 0, then KNNreg
    
% OUTPUTS:
    % predicted: Predicted y values for each test pattern in xtest
    

% Reference:
    % Kumbure, M.M. and Luukka, P. (2021) A generalized fuzzy k-nearest neighbor 
    % regression model based on Minkowski distance. Granular Computing

% Created by Mahinda Mailagaha Kumbure, 06/2021 
% Based on Keller's definition of the fuzzy k-nearest neighbor algorithm.

if nargin<5
    fuzzy = true;
end

if nargin<6
    m = 1.5;
end


num_test  = size(xtest,1); % number test patterns


% Initialization
predicted = zeros(num_test, length(K));

%% BEGIN FKNNreg

% For each test pattern

for i=1:num_test

    % computer the Euclidean distances from test pattern to each train
    % pattern, (here, 'pdist2' MATLAB built-in function is used)
     euc_dis   = pdist2(xtrain, xtest(i,:),'euclidean');
     distances = euc_dis';
     
    [~, indeces]   = sort(distances); % Sort the distances
	neighbor_index = indeces(1:K);    % Find the indexes of nearest neighbors
    weight         = ones(1,length(neighbor_index)); % Initialization of weights
    
	if fuzzy
 	    % compute fuzzy weights:
        % though this weight calculation should be: 
        % weight = distances(neighbor_index).^(-2/(m-1)), 
        % but since we did not take sqrt above and the inverse 
        % 2th power the weights are: weight = sqrt(distances(neighbor_index)).^(-2/(m-1));
        % which is equaliavent to:
 	    weight = distances(neighbor_index).^(-1/(m-1));
 
 	    % set the Inf (infite) weights, if there are any, to  1.
        if max(isinf(weight))
           warning(['Some of the weights are Inf for sample: ' ...
           num2str(i) '. These weights are set to 1.']);
           weight(isinf(weight)) = 1;
        end
    end
    
    % find the predicted output 
	test_out     = weight*ytrain(neighbor_index,:)/(sum(weight));
	predicted(i) = test_out;

end

