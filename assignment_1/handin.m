clear;
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

	%trying another basis function
	mu = mean(X)
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
data = dlmread('winequality-white.csv',';',1,0);

[train_data, val_data, test_data] = splitData(data);

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

if testing
	% plot cost to see convergence
	figure; plot(his); ylabel('cost'); xlabel('iterations');
	errors = testData(test_data, mu, sigma, theta);
	figure; plot(errors(:,1), errors(:,2)); xlabel('error margin'); ylabel('% correct');
	% test r-squared
end
fprintf('R^2 = %.4f\n', computeRSquared(y, X*theta));

% lets try with the normal normal equation
XN = sepXY(train_data); %We separate X and y from the original training data
XN = [ones(length(y),1) XN]; % again we add ones
thetaN = normalEquation(XN, y); % then we find theta
fprintf('With normal equation: R^2 = %.4f\n', computeRSquared(y, XN*thetaN)); %and compute R squared

function erms = errorfunction(X, theta, y)
	e = 0.5 * sum((X*theta-y).^2);
	erms = sqrt(2*e/length(y));
endfunction

% Polynomial regression
% Model selection
M = 3;
X = sepXY(train_data);
[Xval yval] = sepXY(val_data);
XvalL = Xval;
XvalLO = [ones(length(yval),1) XvalL];
XL = X;
errorhis = [];
errorhis(1,1) = 1;
errorhis(1,2) = errorfunction(XN, thetaN, y);
errorhis(1,3) = errorfunction(XvalLO, thetaN, yval);
% Adding M dimensions
for i = 2 : M
	% fprintf('M:%.f\n', i)
	XL = [XL X .^i];
	XvalL = [XvalL Xval .^i];
	% fprintf('Size of XL:%.f\n',size(XL,2));
	% add ones to new matric
	XLO = [ones(length(y),1) XL];
	XvalLO = [ones(length(yval),1) XvalL];

	thetaL = normalEquation(XLO, y);
	% rsquared = computeRSquared(y, XLO*thetaL);
	% fprintf('%.f order poly (NE): R^2 = %.4f\n', i, rsquared);
	errorhis(i,1) = i;
	errorhis(i,2) = errorfunction(XLO, thetaL, y);
	errorhis(i,3) = errorfunction(XvalLO, thetaL, yval);
end
if testing
	figure;
	plot(errorhis(:,1),errorhis(:,2),'-');
	hold on;
	plot(errorhis(:,1),errorhis(:,3),'-');
	legend('training', 'validation'); xlabel('M'); ylabel('erms');
end


% initlize theta
% init_theta = zeros(size(XN,2),1);
% options = optimset('GradObj', 'on', 'MaxIter', 400);
%
% %  Run fminunc to obtain the optimal theta
% %  This function will return theta and the cost
% [thetaL, costL] = fminunc(@(t)(maximumLikelihoodCost(t, X, y)), init_theta, options);
% fprintf('2nd order poly (ML): R^2 = %.4f\n', computeRSquared(y, XL*thetaL));
