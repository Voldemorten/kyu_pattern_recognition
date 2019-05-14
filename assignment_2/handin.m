% load packages
pkg load statistics; %pkg load

clear; close all;

% constants
m_1 = [3;1];
m_2 = [1;3];
sig_1 = [1 2; 2 5];
sig_2 = sig_1;
n = 10;

% debug
show_plots = 1;

% Exercise 1
R1 = mvnrnd(m_1,sig_1,n);
R2 = mvnrnd(m_2,sig_2,n);

if show_plots
	figure; plot(R1(:,1),R1(:,2),'r+'); hold on; plot(R2(:,1),R2(:,2),'bo'); legend("R1","R2");
	axis([-2 8 -2 8], "square");
	printf("Showing original plot. Press any key to draw P1 and P2");
	input("","s");
end
% Exercise 2 (PCA)
% Step 1: Merge data;
R = [R1;R2];
% step 2: Calculate covariance matrix
cov_R = cov(R);
% step 3: Calculate eigenvector and eigenvalues
[eig_vec, eig_val] = eig(cov_R);
% step 4: Eigenvalues are not ordered. Sort them
[eig_val, i] = sort(diag(eig_val), 'descend');
eig_vec = eig_vec(:,i);

%scale to scale points such that our P-lines get longer.
scale = 10;

if show_plots
	% Draw eigenvector
	line([-eig_vec(1,1) * scale eig_vec(1,1) * scale],[-eig_vec(2,1) * scale eig_vec(2,1) * scale], "linestyle", "-", "color", "b");
	line([-eig_vec(1,2) * scale eig_vec(1,2) * scale],[-eig_vec(2,2) * scale eig_vec(2,2) * scale], "linestyle", "-", "color", "b");

	printf("Press to move data by substracting mean");
	input("","s");
	% Move data by mean
	R = R-mean(R);
	figure; hold on;
	plot(R(1:n,1),R(1:n,2),'r+'); plot(R(n+1:n*2,1),R(n+1:n*2,2),'bo');
	legend("P1","P2","R1","R2"); axis("square");
	xlim([-5 5]);
	ylim([-5 5]);
	axis("square");

	line([-eig_vec(1,1) * scale eig_vec(1,1) * scale],[-eig_vec(2,1) * scale eig_vec(2,1) * scale], "linestyle", "-", "color", "b");
	line([-eig_vec(1,2) * scale eig_vec(1,2) * scale],[-eig_vec(2,2) * scale eig_vec(2,2) * scale], "linestyle", "-", "color", "b");
endif
