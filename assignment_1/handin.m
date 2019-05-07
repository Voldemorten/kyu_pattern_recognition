clear ; close all; %clc
function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
%
%               Note that X is a matrix where each column is a
%               feature and each row is an example. You need
%               to perform the normalization separately for
%               each feature.
%
% Hint: You might find the 'mean' and 'std' functions useful.
%

	mu = mean(X);

	sigma = std(X);
	for i = 1 : size(X,2),
		X_norm(:, i) = (X(:, i) - mu(:, i)) / sigma(i);
	end;
% ============================================================

end
fprintf("Loading data...\n");

% data = dlmread ("real_estate_data_cleaned.csv", ";",1,0);
% % strip date
% data = data(:,2:size(data,2));
white_wines = dlmread("winequality-white.csv",";",1,0);
data = white_wines;


% split data in .8, .1, .1
N = length(data);
train_data = data(1:N/10*8, :);
%length of train data
N_td = length(train_data);

val_data = data(N_td+1:N/10*9, :);
N_val = length(val_data);

test_data = data(N_td+N_val+1:N, :);
N_test = length(test_data);

% strips result column from train_data into y
X = train_data(:,1:size(train_data, 2)-1);

%normalize data
[X mu sigma] = featureNormalize(X);

y = train_data(:, size(train_data,2));

fprintf("data loaded…\n");
% Vectorizing:
% step 1: add a column of ones to X
X = [ones(N_td,1), X];
% step 2: initlize theta to 0;0;…;c
theta = zeros(size(X,2),1);
% found by guessing..
alpha = 0.01; %this is for the housing example
it = 500;

function cost = calcCost(X, y, theta, N)
	% Hypothesis is defined by h_theta(x) = X*theta
	H = X*theta;
	% The cost function is defined as 1/2N*sum((H-y)^2)
	cost = sum((H-y) .^2)/(2*N);
endfunction;

function [theta, his] = gradientDescent(X, y, theta, alpha, N, it)
	his = zeros(it, 1);
	for i = 1 : it
		%Se papir for at forstå hvorfor den er sådan her.
		% theta = theta - alpha * (1/N) * ((X*theta - y)' * X)';
		theta = theta - alpha * (1/N) * (((X*theta) - y)' * X)';
		% Vectorized
	    his(i) = calcCost(X, y, theta, N);
	end
endfunction

fprintf("Calculating theta…\n");
[theta, his] = gradientDescent(X, y, theta, alpha, N_td, it);
% [theta, his] = gradientDescent(X, y, theta, alpha, m, it);

% Plot the linear fit
% hold on; % keep previous plot visible
% plot(X(:,2), X*theta, '-')
% legend('Training data', 'Linear regression')
% hold off % don't overlay any more plots on this figure
% figure;
plot(his);
% figure;
% plot(X(:,2), y, 'rx', 'MarkerSize', 10);
% hold on;
% plot(X(:,2), X*theta, '-');

% demo house =
% demoHouse = [1 32 84.87882 10]; %expected 37.9
%
fprintf("Testing results…\n");
correct = 0;
for i = 1 : length(X)
	predicted = round(sum(X(i,:)' .*theta));
	% fprintf('predicted = %.f , correct = %.f\n', predicted, y(i));
	yr = round(y(i));
	if (predicted <= yr+.5 && predicted >= yr-.5)
		correct++;
	end
end
fprintf('Correct results %.2f percent\n', correct/length(X)*100);

%I have my theta now
% how to validate?
% how to determine alpha and iterations.
