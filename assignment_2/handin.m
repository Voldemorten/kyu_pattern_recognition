% load packages
pkg load statistics; %pkg load

clear; close all;

% constants
m_1 = [3;1];
m_2 = [1;3];
sig_1 = [1 2; 2 5];
sig_2 = sig_1;
n = 100;
c = 2;

% debug
show_plots = 0;

% Exercise 1 - Initialize data
R1 = mvnrnd(m_1,sig_1,n);
R2 = mvnrnd(m_2,sig_2,n);

if show_plots
	figure; plot(R1(:,1),R1(:,2),'r+'); hold on; plot(R2(:,1),R2(:,2),'bo');
	axis([-10 10 -10 10], "square");
	printf("Showing original plot. Press any key to draw P1 and P2");
	input("","s");
end

% Exercise 2 - PCA)
% Step 1: Merge data;
R = [R1;R2];
% step 2: Calculate covariance matrix
cov_R = cov(R);
% step 3: Calculate eigenvector and eigenvalues
[eig_vec, eig_val] = eig(cov_R);
% step 4: Eigenvalues are not ordered. Sort them
[eig_val, i] = sort(diag(eig_val), 'descend');
eig_vec = eig_vec(:,i);
eig_vec_lda = eig_vec;

%scale to scale points such that our P-lines get longer.
scale = 10;

if show_plots
	% Draw P1, P2
	line([-eig_vec(1,1) * scale eig_vec(1,1) * scale],[-eig_vec(2,1) * scale eig_vec(2,1) * scale], "linestyle", "-", "color", "r");
	line([-eig_vec(1,2) * scale eig_vec(1,2) * scale],[-eig_vec(2,2) * scale eig_vec(2,2) * scale], "linestyle", "-", "color", "b");
	legend("R1","R2","P1","P2");

	printf("Press any key to move data by substracting mean\n");
	input("","s");

	% Move data by mean
	R_new = R-mean(R);
	figure; hold on;
	plot(R_new(1:n,1),R_new(1:n,2),'r+'); plot(R_new(n+1:n*2,1),R_new(n+1:n*2,2),'bo');
	axis([-10 10 -10 10], "square");

	line([-eig_vec(1,1) * scale eig_vec(1,1) * scale],[-eig_vec(2,1) * scale eig_vec(2,1) * scale], "linestyle", "-", "color", "r");
	line([-eig_vec(1,2) * scale eig_vec(1,2) * scale],[-eig_vec(2,2) * scale eig_vec(2,2) * scale], "linestyle", "-", "color", "b");

	legend("R1","R2","P1","P2");
endif

% Exercise 3 - LDA
% TODO: UNDERSTAND AND REWRITE!
printf("Press any key to continue with LDA");
input("","s");
mu_total = mean(R)
mu = [ mean(R1); mean(R2) ]
Sw = (R - mu(c,:))'*(R - mu(c,:))
Sb = (ones(c,1) * mu_total - mu)' * (ones(c,1) * mu_total - mu)
[eig_vec, eig_val] = eig(Sw\Sb)
[eig_val, i] = sort(diag(eig_val), 'descend');
eig_vec = eig_vec(:,i);
R_new = R-mean(R);

if show_plots
	figure; hold on;
	plot(R_new(1:n,1),R_new(1:n,2),'r+'); plot(R_new(n+1:n*2,1),R_new(n+1:n*2,2),'bo');
	axis([-10 10 -10 10], "square");

	line([-eig_vec(1,1) * scale eig_vec(1,1) * scale],[-eig_vec(2,1) * scale eig_vec(2,1) * scale], "linestyle", "-", "color", "r");
	legend("R1","R2","P1");
end
