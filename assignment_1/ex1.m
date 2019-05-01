clear ; close all; %clc

fprintf("Loading data...\n");
% reads data from csv excluding line 1
% the format is: "fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
% In this assignment we want to predict quality
% red_wines = dlmread ("winequality-red.csv", ";",1,0);
% white_wines = dlmread("winequality-white.csv",";",1,0);
% data = vertcat(red_wines, white_wines);

%trying with a different dataset
data = dlmread ("real_estate_data_cleaned.csv", ";",1,0);
% strip date
data = data(:,2:size(data,2));

% data = dlmread ("OnlineNewsPopularity.csv", ",",1,1);
% data = load("ex1data1.txt");


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
% X = train_data(:,1)
y = train_data(:, size(train_data,2));

fprintf("data loaded…\n");

% fprintf('First 10 examples from the dataset: \n');
% fprintf(' x = [%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% No need to normalize

% Vectorizing:
% step 1: add a column of ones to X
X = [ones(N_td,1), X];
% step 2: initlize theta to 0;0;…;c
theta = zeros(size(X,2),1);
% found by guessing..
alpha = 0.00000001; %this is for the housing example
it = 200;

% alpha = 0.0001; %this is for the wines example
% it = 500;

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
		theta = theta - alpha * (1/N) * ((X*theta - y)' * X)';
		% Vectorized
	    his(i) = calcCost(X, y, theta, N);
	end
endfunction

fprintf("Calculating theta…\n");
[theta, his] = gradientDescent(X, y, theta, alpha, N_td, it);
plot(his);
% figure;
% plot(X(:,2), y, 'rx', 'MarkerSize', 10);
% hold on;
% plot(X(:,2), X*theta, '-');

% fprintf("Testing results…\n");
% correct = 0;
% for i = 1 : length(X)
% 	predicted = round(sum(X(i,:)' .*theta));
% 	% fprintf('predicted = %.f , correct = %.f\n', predicted, y(i));
% 	if (predicted == y(i))
% 		correct++;
% 	end
% end
% fprintf('Correct results %.2f percent\n', correct/length(X)*100);

%I have my theta now
% how to validate?
% how to determine alpha and iterations.
