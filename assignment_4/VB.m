% load packages
pkg load statistics;

clear; close all; clc;

% -------------------
% e-step!
% -------------------
function gamma = eStep(X, N, K, D, W, mu, a, b, v)
	sum_ = ones(K, 1);
	for k = 1 : K
		for i = 1 : D
			sum_(k) = sum_(k) + psi((v(k) + 1 - i) / 2); % used in 10.65 from Bishop
		end
	end
	logrho = ones(N,K);
	gamma = ones(N,K);
	for k = 1 : K
	    for n = 1 : N
	        mu_ = D / b(k) + v(k) * (X(n, :)' - mu(:, k))'*W(:, :, k) * (X(n , :)' - mu(:, k)); % 10.64 from Bishop
	        delta_ = sum_(k) + D * log(2) + log(det(W(:, :, k))); % 10.65 from Bishop
	        phi_ = psi(a(k)) - psi(sum(a)); % 10.66 from Bishop.
	        logrho(n, k) = phi_ + 0.5 * delta_ - (D * log(2 * pi)) / 2 - 0.5 * mu_; % Used in 10.46 From Bishop

	    end
	end
	rho = exp(logrho);
	for n = 1 : N
	    sum_ = 0;
	    for k = 1 : K
			% Update sum_
	        sum_ = sum_ + rho(n, k);
	    end
	    for k = 1 : K
			% Update gamma
	        gamma(n, k) = rho(n, k) / sum_;
	    end
	end
endfunction

% -------------------
% M-step!
% -------------------
function [ratio, cov_ , W, mu, a, b, v] = mStep(X, N, K, D, W, W0inv, mu, m0, gamma, a0, b0, v0)
	Winv = zeros(D, D, K);
	nk = sum(gamma); % 10.51 from Bishop
	Sx = zeros(D, K);
	for k = 1 : K
        Sx(:, k) = sum(repmat(gamma(:, k), [1, D]) .* X); % 10.52 from Bishop
    end
	Sxx = zeros(D, D, K);
    for k = 1 : K
        m = zeros(D, D);
        for n = 1 : N
            m = m + gamma(n, k) * X(n, :)' * X(n, :); % 10.62 from Bishop
        end
        Sxx(:, :, k) = m;
    end

	% Update parameters
	a = a0 + nk'; % 10.58 from Bishop
	b = b0 * ones(K,1) + nk'; % 10.60 from Bishop
	for k = 1 : K
	    mu(:, k) = (b0 * mu(:, k) + Sx(:, k)) / (b0 + nk(k)); % 10.61 from Bishop
	end
	v = v0 + nk'; % 10.63 From Bishop
	for k = 1 : K
	    Winv(:, :, k) = W0inv + b0 * (m0 * m0') + Sxx(:, :, k) - b(k) * (mu(:, k)*mu(:, k)');
	    W(:, :, k) = inv(Winv(:, :, k));
	end
	%to estimate the lower bound we also need the cov matrices
	cov_ = zeros(D, D, K);

	for k = 1 : K
	    cov_(:, :, k) = Sxx(:, :, k) / nk(k) - mu(:, k) * mu(:, k)';
	end
	ratio = nk / sum(nk);

endfunction

% -------------------
% Helpers!
% -------------------
function converged = converged(mu, mu_, epsilon)
	converged = all(all(abs(mu-mu_) <= epsilon));
end

% -------------------
% initialization
% -------------------
X = csvread('x.csv');

% Set initial parameters
N = size(X, 1); % Total number of observations
D = size(X, 2); % Dimensions
K = 4; % Number of classes i want to sort the data into.


% Initialize mu
mu = zeros(D, K);
m0 = zeros(D, K);
for k = 1 : K
   m0(:, k) = X(k, :);
end
mu = m0;

% Initialize hyperparameters
a0 = 0.1 * ones(K, 1);
a = a0;
b0 = (1)*ones(1, 1);
b = b0;
v0 = (D + 1) * ones(K, 1);
v = v0;

% Initialize rest
W0inv = eye(D);
W = zeros(D, D, K);

for k = 1 : K
    W(:, :, k) = eye(D, D);
end

gamma = zeros(N, K);

% Run one maximization step
[ratio, cov_ , W, mu, a, b, v] = mStep(X, N, K, D, W, W0inv, mu, m0, gamma, a0, b0, v0);

% delete files. This is done to make sure that the files are created fresh everytime we run the algorithm.

delete("out/vb/Z_gamma.csv");
delete("out/vb/param.dat.csv");

for i = 1 : 1000
	i
	gamma = eStep(X, N, K, D, W, mu, a, b, v);
	mu_ = mu;
	[ratio, cov_ , W, mu, a, b, v] = mStep(X, N, K, D, W, W0inv, mu, m0, gamma, a0, b0, v0);
	csvwrite("out/vb/Z_gamma.csv", i, "-append");
	csvwrite("out/vb/Z_gamma.csv", gamma', "-append");

	if(converged(mu, mu_, 0.00001))
		f = fopen('out/vb/param.dat', 'wt');
		fprintf(f, "mu:\n");
		for j = 1 :  size(mu, 1)
			fprintf(f, '%s', mat2str(mu(j,:)))
			fprintf(f, "\n" )
		end
		fprintf(f, "\n\na:\n");
		fprintf(f, '%s', mat2str(a));
		fprintf(f, "\n\nb:\n")
		fprintf(f, '%s', mat2str(b));
		fprintf(f, "\n\nv:\n")
		fprintf(f, '%s', mat2str(v));
		fclose(f);
		break;
	end
end
