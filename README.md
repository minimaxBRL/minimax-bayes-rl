# Minimax-Bayes Reinforcement Learning 

## **Infinite MDP**:

### Plotting existing results:

    python minimax.py --plot
    
    python minimax.py --plot --num_rollouts 10 --num_mdps 100 --seed 100 --mmpri params/maximin_prior.npy --mmpol params/minimax_policy.npy --unpol params/br_uniform_policy.npy --brpol params/br_maximin_policy.npy
    
### Training new policies / priors:

New policies and priors will end up in a directory results/<{both, policy, prior}_adaptive>/...

    python minimax.py --both
    
    python minimax.py --prior
    
    python minimax.py --policy
    
    python minimax.py --both --seed 100 --train_prior params/maximin_prior.npy --train_policy params/minimax_policy.npy
