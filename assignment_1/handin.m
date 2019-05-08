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

% ------------ constants --------------
% found by trial and error
alpha = 0.01;
it = 500;
debug = 0;
% -------------- main -----------------


% loading data
% data = dlmread("winequality-white.csv",";",1,0);
data = load("ex1data1.txt");

% ----- housing example -----


% split data in .8, .1, .1
N = length(data);
train_data = data(1:N/10*8, :);
train_data = [1 1; 2 3; 3 5; 4 6; 5 6.5; 6 6; 7 5; 8 3; 9 1];
%length of train data
N_td = length(train_data);

val_data = data(N_td+1:N/10*9, :);
N_val = length(val_data);

test_data = data(N_td+N_val+1:N, :);
N_test = length(test_data);

% strips result column from train_data into y
X = train_data(:,1:size(train_data, 2)-1);
%UNCOMMENT
% [X mu sigma] = featureNormalize(X); %normalize. Important to keep mean and std for later use, when testing on non-normalized data.
y = train_data(:, size(train_data,2));

% Vectorizing:
% step 1: add a column of ones to X
X = [ones(N_td,1), X];
% step 2: initlize theta to 0;0;…;c
theta = zeros(size(X,2),1);
% step 3: calculate theta
[theta, his] = gradientDescent(X, y, theta, alpha, N_td, it);

% plot cost to see conversion
if debug
	figure; plot(his); ylabel('cost'); xlabel('iterations');
end

% test results
if debug
	X_test = test_data(:,1:size(test_data, 2)-1);
	X_test = featureNormWithExisting(X_test, mu, sigma);
	X_test = [ones(N_test,1), X_test];
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
end

if debug
	figure; plot(errors(:,1), errors(:,2)); xlabel('error margin'); ylabel('% correct');
end

figure; hold on;
plot(X(:,2),y,'rx');
plot(X(:,2), X*theta, '-')

% P1 = polyfit(X(:,2),y,1);
P1 = flip(polyfit(X(:,2),y,1));
ppoly = X * P1';
plot(X(:,2), ppoly, '-')

P2 = flip(polyfit(X(:,2),y,2));
XN = [X X(:,2).^2];
ppoly2 = XN * P2';
plot(X(:,2), ppoly2, '-')
legend('points','regression','p1','p2')
% hold off;

SS_tot = sum((y-mean(y)).^2);
SS_res = sum((y-X*theta).^2);
R2 = 1-(SS_res/SS_tot);
fprintf('R2 with theta: %.4f\n',R2);

%polyfit
SS_tot = sum((y-mean(y)).^2);
SS_res = sum((y-ppoly).^2);
R2 = 1-(SS_res/SS_tot);
fprintf('R2 with polyfit: %.4f\n',R2);

SS_tot = sum((y-mean(y)).^2);
SS_res = sum((y-ppoly2).^2);
R2 = 1-(SS_res/SS_tot);
fprintf('R2 with polyfit2: %.4f\n',R2);




%sum((predicted-y).^2)
%sum((predicted-mean(y)).^2)
