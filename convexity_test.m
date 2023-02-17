## -*- Mode: octave -*-
pkg load statistics
x = linspace(0,1,100);


## x: points to update, U: utility function, distr: probability
function V = propagate_belief(x, U, distr)
  V = U;
  for t=1:length(x)
	P = distr(x(t), x);
	V(t) = P*U';
  endfor
endfunction

## It has to be the case that the expected value of y is equal to x
function P = uniform_martingale(x, y, epsilon=0.2)
  epsilon = min(x, min(epsilon, 1 - x));
  P = (abs(x - y) <= epsilon);
  P/=sum(P);

endfunction

## It has to be the case that the expected value of y is equal to x
function P = beta_martingale(x, y, n=2)
  alpha = n * x;
  beta = n - alpha;
  P = betapdf(y, alpha, beta);
  P/=sum(P);
endfunction

function V = propagate_discrete_belief(x, U, P)
  V = U;
  for t=1:length(x)
	prior = x(t);
	V(t) = 0;
	for z =1:2
	  p_z = P(1,z) * prior + P(2,z) * (1 - prior);
	  posterior = prior * P(1,z) / p_z;
	  x_post = argmin(abs(posterior - x));
	  V(t) += p_z * U(x_post);
	endfor
	
  endfor
endfunction


function V = propagate_gaussian_belief(x, U, params)
  V = U;
  for t=1:length(x)
	prior = x(t);
	V(t) = 0;
	Z_norm = 0;
	for k =1:length(x)
	  z = x(k);
	  p_z = normpdf(z, params(1), 1)* prior + normpdf(z, params(2), 1)* (1 - prior);
	  posterior = prior * normpdf(z, params(1), 1) / p_z;
	  x_post = argmin(abs(posterior - x));
	  V(t) += p_z * U(x_post);
	  Z_norm += p_z;
	endfor
	V(t) /= Z_norm;
  endfor
endfunction
  


function V = propagate_gaussian_belief_sampling(x, U, params)
  V = U;
  for t=1:length(x)
	prior = x(t);
	V(t) = 0;
	n_samples = 100;
	for n =1:n_samples
	  model = 1 + (rand < prior);
	  z = normrnd(params(model), 1);
	  p_z = normpdf(z, params(1), 1)* prior + normpdf(z, params(2), 1)* (1 - prior);
	  posterior = prior * normpdf(z, params(1), 1) / p_z;
	  x_post = argmin(abs(posterior - x));
	  V(t) += U(x_post) / n_samples;
	endfor
	
  endfor
endfunction


figure(1)
U = max(x, 1 - x);
hold off
plot(x, U)
hold on
P = [0.49 0.51;
	 0.51, 0.49];

for k=1:10
  U = propagate_discrete_belief(x, U, P);
  plot(x, U)
endfor
print("symmetric_propagation_0_49.pdf");

figure(2)
U = max(x, 1 - x);
hold off
plot(x, U)
hold on
P = [0.1, 0.9;
	 0.9, 0.1];

for k=1:10
  U = propagate_discrete_belief(x, U, P);
  plot(x, U)
endfor
print("symmetric_propagation_0_1.pdf");


figure(3)
U = max(x, 1 - x);
hold off
plot(x, U)
hold on
P = [0.4, 0.6;
	 0.1, 0.9];

for k=1:10
  U = propagate_discrete_belief(x, U, P);
  plot(x, U)
endfor
print("asymmetric_propagation_0_1_0_4.pdf");


### Hm ###

figure(4)
U = max(x, 1 - x);
hold off
plot(x, U)
hold on
normal_models = [0.0, 1.0];
for k=1:4
  U = propagate_gaussian_belief(x, U, normal_models);
  plot(x, U);
  drawnow();
  printf("Iter %d\n", k);
  fflush(stdout);
endfor
print("gaussian_propagation_0_45.pdf");

