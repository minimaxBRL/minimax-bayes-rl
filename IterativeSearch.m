## -*- Mode: octave -*-

1;

## return the value function associated with the MDPs here
function [V_opt, V_m] = getValueFunction(state)
  V_opt = exp(sum((state.xi - state.center).^2)) ;
  grad = 2 * (state.xi - state.center) * exp(sum((state.xi - state.center).^2));
  #V_opt = sum((state.xi - state.center).^2) ;
  # grad = 2 * (state.xi - state.center);
  V_m = zeros(state.n_MDPs, 1);
  for i=1:state.n_MDPs
    p = zeros(state.n_MDPs, 1);
    p(i) = 1;
    V_m(i) = (p - state.xi)' * grad + V_opt;
  endfor
endfunction

state.n_MDPs = 3;
state.center = dirichlet_rnd(ones(state.n_MDPs, 1))

## first, test that the value functions are correct.
if (0)
  hold off;
  for xi=0.0:.1:1.0
	state.xi = [xi, 1 - xi]';
	[V_opt, V_m] = getValueFunction(state);
	plot(xi, V_opt, '1o')
	hold on
	plot([1, 0], V_m, '2');       # since the 1 means the probability of
                                # the first mdp is 1, we have to put
                                # this on the right hand side
  endfor
  sleep(1);
endif

## now, test the algorithm11

#printf("random first xi\n");
state.xi = ones(state.n_MDPs, 1) / state.n_MDPs;
##state.xi
hold off;
strategies = [];
values = [];





dominated = [];
for iter = 1:32

  [V_opt, V_m] = getValueFunction(state);
  V_opt_list(iter) = V_opt;
  strategies = [strategies V_m];
  values = [values V_opt];

  n_strategies = columns(strategies);

  if (n_strategies >= state.n_MDPs)

	permutations = nchoosek([1:n_strategies], state.n_MDPs);

	## since we always choose the optimal strategy for a point, no
	## strategy is dominated, so there is no need to track non-dominated
	## strategies.
	
	n_permutations = rows(permutations);
##	printf("Iter %d, matrices %d\n", iter, n_permutations);
	fflush(stdout);

	min_x_value = inf;
	new_value = false;

	for k=1:n_permutations
	  A = zeros(state.n_MDPs, state.n_MDPs);
	  A(1,:) = ones(1, state.n_MDPs);
	  b = zeros(state.n_MDPs, 1);
	  b(1) = 1;
	  p_k = permutations(k, 1);
	  for j=2:state.n_MDPs; 
		p_j = permutations(k, j);
		A(j,:) = strategies(:, p_k)' - strategies(:, p_j)';
		b(j) = values(p_j) - values(p_k);
	  endfor
	  if (rank(A) == state.n_MDPs) 
		x = inv(A) * b;
		##x = A \ b;
		other_values = x' * strategies;
		x_value = max(other_values);
		x_critical = x' * strategies(:, p_k); # + values(p_k);
		## new point is not dominated ...
		if (x_value < min_x_value && min(x) >= 0 && x_critical >= x_value)
		  min_x = x;
		  min_x_value = x_value;
		  new_value = true;
		endif
	  endif
	endfor
	if (new_value)
	  state.xi = min_x;
	  new_V_opt= getValueFunction(state);
	  distance = norm(min_x - state.center);
	  printf("%f (%f) %f\n",
			 min_x_value,
			 distance,
			 new_V_opt - min_x_value);
	  fflush(stdout);
	else
	  state.xi = dirichlet_rnd(ones(state.n_MDPs, 1));
	endif
  else
	state.xi = dirichlet_rnd(ones(state.n_MDPs, 1));
  endif

endfor
