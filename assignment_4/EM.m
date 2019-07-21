% load packages
pkg load statistics;

clear; close all;

% -------------------
% e-step!
% -------------------
function w = eStep(X, N, K, mu, sigma, prior, D)
	% initlize w, which is the matrix, that contains the probabilities for each data point, to be in cluster 1 or 2.
	% That means it should be a N x K matrix. N data points and K clusters.
	w = zeros(N, K);
	% calculate the probability for each datapoint, given the the current mu and the current covariance assigned to each cluster. In first iteration the mu's are different, but the covariance are the same. We have to do this for each cluster.
	for k=1:K
		w(:,k) = pdfGaussian(X, mu{k}, sigma{k}, D);
	end
	%multiple each value with the prior
	w = w.*prior';
	% normalizing -> makes w sum to 1.
	w = w ./ sum(w,2);
	% done!
endfunction

% -------------------
% M-step!
% -------------------
function [prior, mu, sigma] = mStep(w, N, K, mu, sigma, X, prior)
	prior = mean(w,1)';
	% Calculate the new means for each of the classes/distributions
	mu_ = (w'*X)'./sum(w,1);
	for k=1:K
	  % update the mean of the class/distribution
	  mu{k} = mu_(:,k)';
	  % update the covariance matrix of the class/distribution
	  X_mu = X - mu{k};
	  X_mu_ = (w(:,k).*X_mu)';
	  sigma{k} = (X_mu_*X_mu) ./ sum(w(:,k));
	end
endfunction

% -------------------
% Helpers!
% -------------------
function pdf = pdfGaussian(X, mu, sigma, D)
	% To make it eassier, we subtract the mean from each datapoint in X immediately
	Xm = X-mu;
	pdf = exp(-0.5 * sum((Xm) * inv(sigma) .* (Xm),2)) / sqrt((2*pi)^D * det(sigma));
endfunction

function converged = converged(mu, mu_, epsilon)
	m_m = [mu_{1};mu_{2}];
	mm = [mu{1}; mu{2}];
	converged = all(abs(mm-m_m) <= epsilon);
end

% -------------------
% initialization
% -------------------
X = csvread('x.csv');

% Set initial parameters
N = size(X,1); % Total number of observations
D = size(X,2); % Dimensions
K = 4; % Number of classes i want to sort the data into.


% Initialize the covariance matrix for each cluster to be equal to the covariance of the full data set
% Also Initialize mu to be the two first points in X
sigma = [];
mu = [];
for k = 1:K
   sigma{k} = cov(X);
   mu{k} = X(k,:);
end

%I Initlize the prior probability for each data point. We are assuming that the probability is equal
% prior = [.5 .5];
prior = (ones(K,1) * (1 / K));

% delete files. This is done to make sure that the files are created fresh everytime we run the algorithm.
delete("out/em/Z_prior.csv");
delete("out/em/param.dat");

for(i = 1 : 1000)
	i
	w = eStep(X, N, K, mu, sigma, prior, D);
	mu_ = mu;
	[prior, mu, sigma] = mStep(w, N, K, mu, sigma, X, prior);

	% writing to files
	csvwrite("out/em/Z_prior.csv", i, "-append");
	csvwrite("out/em/Z_prior.csv", prior', "-append");

	if(converged(mu, mu_, 0.00001))
		f = fopen('out/em/param.dat', 'wt');
		fprintf(f, "mu:\n");
		for j = 1 :  size(mu, 2)
			fprintf(f, '%s ', mat2str(mu{j}));
			fprintf(f, "\n" )
		end
		fprintf(f, "\n\nsigma:\n");
		for j = 1 : size(sigma, 2)
			for k = 1 : size(sigma{j},1)
				fprintf(f, '%s', mat2str(sigma{j}(k,:)));
				fprintf(f, '\n')
			end
			fprintf(f, '\n');
		end
		fprintf(f, "\n\nw:\n")
		for j = 1 : size(w, 1)
			fprintf(f, '%s', mat2str(w(j, :)));
			fprintf(f, '\n');
		end
		fclose(f);
		break;
	end
end
