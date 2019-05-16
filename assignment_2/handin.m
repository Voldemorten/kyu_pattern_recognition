% load packages
pkg load statistics;

clear; close all;

% constants
m_1 = [3;1];
m_2 = [1;3];
sig_1 = [1 2; 2 5];
sig_2 = sig_1;
n = 5;
c = 2;

% debug
show_plots = 1;

% Exercise 1 - Initialize data
R1 = mvnrnd(m_1,sig_1,n);
R2 = mvnrnd(m_2,sig_2,n);

%demo:
R1 = [2 3;3 4;4 5;5 6;5 7];
R2 = [2 1;3 2;4 2;4 3;6 4;7 6];

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

	% Projecting data on PC1 (eigenvector 1)
	z = R_new*eig_vec

	% and reconstruct it
	p = z*eig_vec;
	p = bsxfun(@plus, p, mean(R))
	R_new = p;

	% By courtesy of https://stackoverflow.com/questions/14530944/octave-creating-two-histograms-with-color-blending, to work around octaves missing facealpha / transparency function.
	figure;
	hold on;
	[y1, x1] = hist(R_new(1:n,1)*eig_vec(1,1)',20);
	[y2, x2] = hist(R_new(n+1:n*2,1)*eig_vec(1,1)',20);
	[ys1 xs1] = stairs(y1, x1);
	[ys2 xs2] = stairs(y2, x2);
	xs1 = [xs1(1); xs1; xs1(end)];  xs2 = [xs2(1); xs2; xs2(end)];
	ys1 = [0; ys1; 0];  ys2 = [0; ys2; 0];
	clf
	hold on;
	h1=fill(xs1,ys1,"red");
	h2=fill(xs2,ys2,"blue");
	set(h1,'facealpha',0.5);
	set(h2,'facealpha',0.5);
	hold off;
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

z = R_new*eig_vec(:,1)
% and reconstruct it
p = z*eig_vec(:,1)'
p = bsxfun(@plus, p, mean(R))
R_new = p;
figure;
hold on;
[y1, x1] = hist(R_new(1:n,:)*eig_vec(1,:)',20);
[y2, x2] = hist(R_new(n+1:n*2,:)*eig_vec(1,:)',20);
[ys1 xs1] = stairs(y1, x1);
[ys2 xs2] = stairs(y2, x2);
xs1 = [xs1(1); xs1; xs1(end)];  xs2 = [xs2(1); xs2; xs2(end)];
ys1 = [0; ys1; 0];  ys2 = [0; ys2; 0];
clf
hold on;
h1=fill(xs1,ys1,"red");
h2=fill(xs2,ys2,"green");
set(h1,'facealpha',0.5);
set(h2,'facealpha',0.5);
hold off;
