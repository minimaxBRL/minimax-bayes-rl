## -*- Mode: octave -*-

n_models = 3; # number of models
# utility of the optimal policy each model relative to the other models, constructed so that the diagonal values are maximal
U = rand(n_models, n_models);
for k =1:n_models
  U(k,k) = 1
endfor


## R(i,j) is the regret of policy i relative to model j
R = zeros(n_models, n_models)
for policy=1:n_models
  for model=1:n_models
	R(policy, model) = U(model, model) - U(policy, model);
  endfor
endfor

omega = linspace(0,1); # interpolation

src = 1;
dst = 2;
src_belief = zeros(n_models,1); src_belief(src) = 1;
dst_belief = zeros(n_models,1); dst_belief(dst) = 1;

U_plt = zeros(length(omega), 1);
hold off;
for k = 1:n_models
  for i = 1:length(omega)
	belief = omega(i) * src_belief + (1 - omega(i)) * dst_belief;
	U_plt(i) = U(k,:) * belief;
  endfor
  plot(U_plt)
  hold on;
endfor

# Now let us assume that half of the time is spent learning which one is the right belief, and then exploiting it. Then the belief is going to be

