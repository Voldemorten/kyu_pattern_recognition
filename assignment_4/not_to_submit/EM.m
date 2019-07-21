% load packages
pkg load statistics;

clear; close all;

% -------------------
% e-step!
% -------------------
function w = eStep(X, N, K, mu, sigma, prior)
	% initlize w, which is the matrix, that contains the probabilities for each data point, to be in cluster 1 or 2.
	% That means it should be a N x K matrix. N data points and K clusters.
	w = ones(N, K);

	% calculate the probability for each datapoint, given the the current mu and the current covariance assigned to each cluster. In first iteration the mu's are different, but the covariance are the same. We have to do this for each cluster.
	pdf = ones(N, K);
    for (j = 1 : K)
        pdf(:, j) = pdfGaussian(X, mu(j, :), sigma{j});
    end
	%multiple each value with the prior
	pdf = pdf .* prior;

	% normalizing -> makes w sum to 1.
	w = pdf ./ sum(pdf, 2);
	% done!
endfunction

% -------------------
% M-step!
% -------------------
function [prior, mu, sigma] = mStep(w, N, K, mu, sigma, X, prior)
	prior = sum(w)/N; % new prior 1x2
	for (j = 1 : K)
		mu(j, :) = (w(:,j)' * X) ./ sum(w);

		% To make it eassier, we subtract mu from X
		X_mu = X - mu(j,:);

		% And then we compute the contribution covariance for each row in X
		sigma_ = zeros(2, 2);
		for (i = 1 : N)
			sigma_ = sigma_ + (w(i, j) .* (X_mu(i, :)' * X_mu(i, :)));
		end
		sigma{j} = sigma_ ./ sum(w(:, j));
	end
endfunction

% -------------------
% Helpers!
% -------------------
function pdf = pdfGaussian(X, mu, sigma)
	% To make it eassier, we subtract the mean from each datapoint in X immediately
	X = X-mu;
	pdf = (1 / sqrt((2*pi)^2 * det(sigma))) * (exp(-1/2 * sum((X * inv(sigma) .* X),2)));
endfunction

function converged = converged(mu, mu_, epsilon)
	converged = all(abs(mu-mu_) <= epsilon);
end

% -------------------
% Main
% -------------------

% constants
% m_1 = [3 5];
% m_2 = [0 25];
% sig_1 = [1 1; 10 10];
% sig_2 = [5 1; 5 1];
m_1 = [1 2];
m_2 = [-1 -2];
sig_1 = [3 .2; .2 2];
sig_2 = [2 0; 0 1];
n = 300; % number of random datapoint to generate x 2
K = 2; % classes or clusters

% debug
show_plots = 1;

% Exercise 1 - Initialize data
R1 = mvnrnd(m_1,sig_1,n);
R2 = mvnrnd(m_2,sig_2,n);

if show_plots
	% figure; plot(R1(:,1),R1(:,2),'r+'); hold on; plot(R2(:,1),R2(:,2),'bo');
	% axis([-10 10 -10 10], "square"); title('Original data');
	% printf("Showing original plot. Press enter to draw P1 and P2");
	% input("","s");

	figure(1);
	hold off;
	plot(R1(:, 1), R1(:, 2), 'bo');
	hold on;
	plot(R2(:, 1), R2(:, 2), 'ro');

	set(gcf,'color','white') % White background for the figure.

	plot(m_1(1), m_1(2), 'kx');
	plot(m_2(1), m_2(2), 'kx');

	% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
	% the input values over the grid.
	gridSize = 100;
	u = linspace(-6, 6, gridSize);
	[A B] = meshgrid(u, u);
	gridX = [A(:), B(:)];

	% Calculate the Gaussian response for every value in the grid.
	z1 = pdfGaussian(gridX, m_1, sig_1);
	z2 = pdfGaussian(gridX, m_2, sig_2);


	% Reshape the responses back into a 2D grid to be plotted with contour.
	Z1 = reshape(z1, gridSize, gridSize);
	Z2 = reshape(z2, gridSize, gridSize);

	% Plot the contour lines to show the pdf over the data.
	[c, h] = contour(u, u, Z1);
	[c, h] = contour(u, u, Z2);
	axis([-6 6 -6 6])

	title('Original Data and PDFs');
end

% -------------------
% initialization
% -------------------
% Merge data
X = [R1;R2];
% Set initial parameters

% Get number of observations
N = size(X,1)

% Initialize mu with the two first datapoints.
mu = X([1 2], :);

% Initialize the covariance matrix for each cluster to be equal to the covariance of the full data set
sigma = [];
for (j = 1 : K)
    sigma{j} = cov(X);
end

%I Initlize the prior probability for each data point. We are assuming that the probability is equal
prior = [.5 .5];

for(i = 1 : 1000)
	i
	w = eStep(X, N, K, mu, sigma, prior);

	mu_ = mu;
	[prior, mu, sigma] = mStep(w, N, K, mu, sigma, X, prior);

	if(converged(mu, mu_, 0.00001))
		break;
	end
end


if(show_plots)
	figure(2);
	hold off;
	plot(R1(:, 1), R1(:, 2), 'bo');
	hold on;
	plot(R2(:, 1), R2(:, 2), 'ro');

	set(gcf,'color','white') % White background for the figure.

	plot(m_1(1), m_1(2), 'kx');
	plot(m_2(1), m_2(2), 'kx');

	% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
	% the input values over the grid.
	gridSize = 100;
	u = linspace(-6, 6, gridSize);
	[A B] = meshgrid(u, u);
	gridX = [A(:), B(:)];

	% Calculate the Gaussian response for every value in the grid.
	z1 = pdfGaussian(gridX, mu(1, :), sigma{1});
	z2 = pdfGaussian(gridX, mu(2, :), sigma{2});


	% Reshape the responses back into a 2D grid to be plotted with contour.
	Z1 = reshape(z1, gridSize, gridSize);
	Z2 = reshape(z2, gridSize, gridSize);

	% Plot the contour lines to show the pdf over the data.
	[c, h] = contour(u, u, Z1);
	[c, h] = contour(u, u, Z2);
	axis([-6 6 -6 6])

	title('Original Data and Estimated PDFs');
end
