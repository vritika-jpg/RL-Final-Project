# Reinforcement Learning for Inventory Replenishment under Demand Uncertainty and Working Capital Constraints

## Problem & "Why RL?"
We study a multi-period inventory replenishment problem in which a firm must decide how much inventory to reorder each period under uncertain demand. The goal is to maximize long-run business performance by balancing sales revenue against stockout risk, holding cost, ordering cost, and cash tied up in inventory.
This problem requires Reinforcement Learning because it is a sequential decision-making problem under uncertainty, not a one-time prediction problem. A replenishment decision made today changes future inventory levels, future stockout exposure, future holding costs, and future cash availability. Therefore, the quality of an action cannot be judged only by its immediate effect; it must be evaluated based on its impact on the full future trajectory of the system.

## MDP Formulation

State Space:

We will use a finite, discrete state space suitable for tabular RL. A state at time (t) will be defined as:

  Inventory level: low / medium / high
  
  Demand condition: low / normal / high
  
  Cash availability: tight / normal / ample
  
Thus, the state summarizes the operational and financial information needed to make a replenishment decision.

Action Space:

  At each period, the agent chooses a discrete reorder quantity:
    
    - 0 units
    
    - Small order
    
    - Medium order
    
    - Large order

This keeps the action space finite and executable within a tabular framework.

Reward Function:

The reward at each period will be defined as:

Reward=Sales Revenue−Ordering Cost−Holding Cost−Stockout Penalty

We may also include a small penalty for excessive inventory to reflect working capital inefficiency. This reward structure encourages the agent to maintain profitable inventory levels while avoiding both understocking and overstocking.

## Environment and Data Strategy

We plan to build a custom simulator rather than rely on a closed-form model. In each period, stochastic demand will be generated based on the current demand condition, inventory will be updated based on realized sales and replenishment, and cash availability will evolve depending on the ordering decision and realized revenue.
This approach is appropriate because the environment is naturally sequential and uncertain, and the simulator allows us to generate many episodes for learning. It also matches the course emphasis on model-free RL methods when transition probabilities are not assumed to be explicitly known.

## Baselines and Evaluation

  We will compare the trained RL agent against several baseline policies:
  
    - No reorder policy
    
    - Fixed reorder policy (always order the same quantity)
    
    - Order-up-to heuristic (replenish to a fixed target level)
    
    - Random actions policy with realistic action constraints

The RL agent will be evaluated using:

  - Average total profit per episode
  
  - Stockout frequency / service level
  
  - Average holding cost
  
  - Average ending inventory / cash efficiency

We expect the RL agent to outperform simple heuristic and random policies by learning a state-dependent replenishment strategy that better balances profitability, risk, and inventory efficiency over time.


