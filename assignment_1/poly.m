% clear;
close all;

%functions
function [X_norm mu sigma] = featureNormalize(X)
	mu = mean(X);
	sigma = std(X);
	X_norm = (X - mu) ./ sigma; %% note that the './' means element by element division
end

function cost = calcCost(X, y, theta, N)
	% Hypothesis is defined by h_theta(x) = X*theta
	H = X*theta;
	% The cost function is defined as 1/2N*sum((H-y)^2)
	cost = sum((H-y) .^2)/(2*N);
endfunction;

function [theta, his] = gradientDescent(X, y, theta, alpha, N, it)
	his = zeros(it, 1);
	for i = 1 : it
		theta = theta - alpha * (1/N) * (((X*theta) - y)' * X)';
	    his(i) = calcCost(X, y, theta, N); %illustraion purposes
	end
endfunction

function X_norm = featureNormWithExisting(X, mu, sigma)
	X_norm = (X - mu) ./ sigma;
endfunction

function theta = normalEquation(X, y)
	theta = pinv(X'*X)*X'*y;
endfunction

function errors = testData(test_data, mu, sigma, theta)
	X_test = test_data(:,1:size(test_data, 2)-1);
	X_test = featureNormWithExisting(X_test, mu, sigma);
	X_test = [ones(length(X_test),1), X_test];
	y_test = test_data(:, size(test_data,2));
	res = ones(length(X_test),2);
	errors = ones(25,2);
	for j = 1 : 25
		errormargin = j/10;
		correct = 0;
		for i = 1 : length(X_test)
			predicted = sum(X_test(i,:)' .*theta);
			res(i, 1) = predicted;
			res(i, 2) = y_test(i);
			if (predicted <= y_test(i)+errormargin && predicted >= y_test(i)-errormargin)
				correct++;
			end
		end
		fprintf('With an error margin of %.2f, the correct results are %.2f percent\n', errormargin, correct/length(X_test)*100);
		errors(j,1) = errormargin;
		errors(j,2) = correct/length(X_test);
	end
endfunction

function R2 = computeRSquared(y, predictions)
	SS_tot = sum((y-mean(y)).^2);
	SS_res = sum((y-predictions).^2);
	R2 = 1-(SS_res/SS_tot);
endfunction

function [train_data, val_data, test_data] = splitData(data)
	% split data in .8, .1, .1
	N = length(data);
	train_data = data(1:N/10*8, :);
	val_data = data(length(train_data)+1:N/10*9, :);
	test_data = data(length(train_data)+length(val_data)+1:N, :);
endfunction

function [X, y] = sepXY(data)
	X = data(:,1:size(data, 2)-1);
	y = data(:, size(data,2));
endfunction

% ------------ constants --------------
% found by trial and error
alpha = 0.01;
it = 500;
testing = 0;
% -------------- main -----------------


% loading data
% data = dlmread("winequality-white.csv",";",1,0);
data = load('ex1data1.txt');
data = [1 1; 2 3; 3 5; 4 6; 5 6.5; 6 6; 7 5; 8 3; 9 1];

% ----- housing example -----

[train_data, val_data, testData] = splitData(data);
% ignore split for now
train_data = data;


% strips result column from train_data into y
[X, y] = sepXY(train_data);
[X mu sigma] = featureNormalize(X); %normalize. Important to keep mean and std for later use, when testing on non-normalized data.

% Vectorizing:
% step 1: add a column of ones to X
X = [ones(length(train_data),1), X];
% step 2: initlize theta to 0;0;…;c
theta = zeros(size(X,2),1);
% step 3: calculate theta
[theta, his] = gradientDescent(X, y, theta, alpha, length(train_data), it);

% plot cost to see conversion
if testing
	figure; plot(his); ylabel('cost'); xlabel('iterations');
	errors = testData(test_data, mu, sigma, theta);
	figure; plot(errors(:,1), errors(:,2)); xlabel('error margin'); ylabel('% correct');
end

%% plotting data
figure; hold on;
plot(X(:,2),y,'rx');
plot(X(:,2), X*theta, '-')
legend('data points','linear regression');
fprintf('Linear regression: R^2 = %.4f\n', computeRSquared(y, X*theta));

% Let's try with an extra parameter
XP = [train_data(:,1) train_data(:,1).^2];
[XP mu sigma] = featureNormalize(XP);
XP = [ones(length(y),1) XP]; % add ones
theta2 = zeros(size(XP,2),1); % initlize thetas
[theta2, his] = gradientDescent(XP, y, theta2, alpha, length(XP), it);
% plotting
plot(XP(:,2), XP*theta2, '-');
legend('data points','linear regression','2nd order poly (GD)');
fprintf('2nd order poly (GD): R^2 = %.4f\n', computeRSquared(y, XP*theta2));

% All of the above could also be solved by using the normal normal equation
% Lets revert to training data before feature scaling

% XN = [ones(length(y),1) train_data(:,1) train_data(:,1).^2];
XN = [train_data(:,1) train_data(:,1).^2];
[XN mu sigma] = featureNormalize(XN);
XN = [ones(length(y),1) XN]; % add ones
% find theta
thetaN = normalEquation(XN, y);
% plotting
plot(XN(:,2), XN*thetaN, '-');
legend('data points','linear regression','2nd order poly (GD)','2nd order poly (NE)');
fprintf('2nd order poly (NE): R^2 = %.4f\n', computeRSquared(y, XN*thetaN));

% And lets try again with native polyfit equation
P2 = flip(polyfit(train_data(:,1),y,2));
XPN = [ones(length(y),1) train_data(:,1) train_data(:,1).^2];
ppoly = XPN * P2';
plot(X(:,2), ppoly, '-');
legend('data points','linear regression','2nd order poly (GD)','2nd order poly (NE)','2n order polyfit');
fprintf('2nd order poly (NE): R^2 = %.4f\n', computeRSquared(y, XPN*P2'));




%sum((predicted-y).^2)
%sum((predicted-mean(y)).^2)
