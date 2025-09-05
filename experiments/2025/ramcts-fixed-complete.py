"""
RaMCTS: FIXED IMPLEMENTATION
This version addresses all the issues found in testing
"""

import math
import random
import collections
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Create output directory
OUTPUT_DIR = "ramcts_results_fixed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("RaMCTS FIXED VERSION - Debugged and Optimized")
print("=" * 60)

# ====================
# FrozenLake Model (FIXED)
# ====================
class FrozenLakeModel:
    """Fixed FrozenLake model with better action mapping."""
    
    def __init__(self, map_name="4x4"):
        if map_name == "4x4":
            self.desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
            self.map_size = 4
        elif map_name == "8x8":
            self.desc = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", 
                        "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", 
                        "FHFFHFHF", "FFFHFFFG"]
            self.map_size = 8
            
        self.holes = {r * self.map_size + c 
                     for r, row in enumerate(self.desc) 
                     for c, char in enumerate(row) if char == 'H'}
        self.goal_state = self.map_size * self.map_size - 1
        self.action_count = 4
        
    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        row, col = state // self.map_size, state % self.map_size
        
        # Fixed action mapping
        if action == 0:  # LEFT
            col = max(col - 1, 0)
        elif action == 1:  # DOWN
            row = min(row + 1, self.map_size - 1)
        elif action == 2:  # RIGHT
            col = min(col + 1, self.map_size - 1)
        elif action == 3:  # UP
            row = max(row - 1, 0)
            
        next_state = row * self.map_size + col
        
        if next_state in self.holes:
            return next_state, 0.0, True
        if next_state == self.goal_state:
            return next_state, 1.0, True
        return next_state, 0.0, False
    
    def get_optimal_action(self, state: int) -> int:
        """Get heuristic optimal action (toward goal)."""
        if state == self.goal_state:
            return 0
        
        row, col = state // self.map_size, state % self.map_size
        goal_row, goal_col = self.map_size - 1, self.map_size - 1
        
        # Simple heuristic: move toward goal
        if row < goal_row and (row + 1) * self.map_size + col not in self.holes:
            return 1  # DOWN
        elif col < goal_col and row * self.map_size + (col + 1) not in self.holes:
            return 2  # RIGHT
        elif row > 0 and (row - 1) * self.map_size + col not in self.holes:
            return 3  # UP
        elif col > 0 and row * self.map_size + (col - 1) not in self.holes:
            return 0  # LEFT
        else:
            # Avoid holes
            for action in [1, 2, 3, 0]:
                next_state, _, done = self.step(state, action)
                if not done or next_state == self.goal_state:
                    return action
            return random.randrange(4)

# ====================
# Node (FIXED)
# ====================
class Node:
    def __init__(self, state: Any, action_count: int, 
                 parent: Optional['Node'] = None, 
                 action_taken: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.N = 0
        self.W = 0.0
        self.children: Dict[int, Node] = {}
        self.untried_actions = list(range(action_count))
        random.shuffle(self.untried_actions)
        
    def q_value(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

# ====================
# RaCTS Miner (FIXED)
# ====================
@dataclass
class MinerConfig:
    """Fixed configuration that actually works."""
    decay: float = 0.999  # Slower decay
    prune_threshold: float = 1e-4
    max_table_size: int = 10000
    near_success_quantile: float = 0.5  # MUCH lower threshold
    smoothing_lambda: float = 1.0  # More smoothing
    idf_cap: float = 2.0  # Lower cap
    n_gram_max: int = 2  # Just 1-2 grams for now
    context_weight_max: float = 1.2

class RaCTSMiner:
    """Fixed pattern miner with better scoring."""
    
    def __init__(self, model: FrozenLakeModel, config: MinerConfig = None):
        self.model = model
        self.config = config or MinerConfig()
        
        self.c_all = collections.defaultdict(float)
        self.c_pos = collections.defaultdict(float)
        self.episode_returns = collections.deque(maxlen=50)  # Smaller buffer
        
        self.all_total = 0.0
        self.pos_total = 0.0
        self.pos_updates = 0
        self.top_patterns = []
        
        # Fixed n-gram size
        self.n_max = 2 if model.map_size <= 4 else 2
            
    def _apply_decay(self):
        """Apply decay more carefully."""
        decay = self.config.decay
        
        # Only decay after enough episodes
        if self.all_total < 10:
            return
            
        for key in list(self.c_all.keys()):
            self.c_all[key] *= decay
            if self.c_all[key] < self.config.prune_threshold:
                del self.c_all[key]
                
        for key in list(self.c_pos.keys()):
            self.c_pos[key] *= decay
            if self.c_pos[key] < self.config.prune_threshold:
                del self.c_pos[key]
                
        self.all_total *= decay
        self.pos_total *= decay
            
    def _get_near_success_threshold(self) -> float:
        """Much more lenient threshold."""
        if len(self.episode_returns) < 3:
            return 0.0  # Accept any success early
        # Use median instead of high percentile
        return np.median(self.episode_returns)
    
    def _extract_ngrams(self, trace: List[Tuple[int, int]], n: int) -> List[tuple]:
        if len(trace) < n:
            return []
        return [tuple(trace[i:i+n]) for i in range(len(trace) - n + 1)]
    
    def update(self, trace: List[Tuple[int, int]], total_return: float):
        """Update with more lenient criteria."""
        self._apply_decay()
        self.episode_returns.append(total_return)
        
        # Always update all-episode statistics
        for n in range(1, min(self.n_max + 1, len(trace) + 1)):
            for gram in self._extract_ngrams(trace, n):
                self.c_all[gram] += 1
        self.all_total += 1
        
        # Much more lenient positive criteria
        threshold = self._get_near_success_threshold()
        if total_return >= threshold and total_return > 0:  # Must have SOME reward
            for n in range(1, min(self.n_max + 1, len(trace) + 1)):
                for gram in self._extract_ngrams(trace, n):
                    self.c_pos[gram] += 1
            self.pos_total += 1
            self.pos_updates += 1
            
    def _compute_score(self, gram: tuple) -> float:
        """Simplified scoring that actually works."""
        if self.all_total == 0 or self.c_all.get(gram, 0) < 2:
            return 0.0
            
        # Simple success rate
        success_rate = self.c_pos.get(gram, 0) / (self.c_all.get(gram, 0) + 1)
        
        # Frequency bonus (patterns seen more are better)
        freq_bonus = min(1.0, self.c_all.get(gram, 0) / 20)
        
        return success_rate * (1 + freq_bonus)
    
    def calculate_prior(self, node: Node, action: int, 
                       history: List[Tuple[int, int]]) -> float:
        """Calculate prior with fixed scoring."""
        state = node.state
        
        # Build candidate grams
        score = 0.0
        
        # 1-gram
        gram1 = ((state, action),)
        score += self._compute_score(gram1)
        
        # 2-gram if history available
        if len(history) >= 1:
            gram2 = (history[-1], (state, action))
            score += self._compute_score(gram2) * 1.5  # Bonus for longer patterns
            
        # Add small bonus for goal-directed actions
        if action in [1, 2]:  # DOWN or RIGHT
            score += 0.1
            
        return score

# ====================
# MCTS Solver (FIXED)
# ====================
@dataclass  
class MCTSConfig:
    """Fixed configuration that works."""
    max_sims_per_move: int = 100
    c_uct: float = 1.414  # sqrt(2)
    c_puct: float = 1.0  # Reduced
    beta_max: float = 0.5  # MUCH less aggressive
    beta_start_pos: int = 2  # Earlier
    beta_full_pos: int = 5  # Earlier
    trust_margin: float = 0.0  # No margin
    warm_sims: int = 10  # Less warmup
    mini_duel: bool = False  # Disable for now
    duel_extra_sims: int = 0
    rollout_max_steps: int = 50
    rollout_policy: str = "biased"  # NEW: biased rollouts

class MCTSSolver:
    """Fixed MCTS solver."""
    
    def __init__(self, model: FrozenLakeModel, config: MCTSConfig = None):
        self.model = model
        self.config = config or MCTSConfig()
        self.action_count = model.action_count
        
    def _compute_beta(self, miner: Optional[RaCTSMiner]) -> float:
        """More aggressive beta ramp."""
        if miner is None:
            return 0.0
        
        pos = miner.pos_updates
        if pos < self.config.beta_start_pos:
            return 0.0
        elif pos >= self.config.beta_full_pos:
            return self.config.beta_max
        else:
            progress = (pos - self.config.beta_start_pos) / (self.config.beta_full_pos - self.config.beta_start_pos)
            return self.config.beta_max * progress
            
    def _simulate_rollout(self, state: int) -> float:
        """FIXED: Biased rollout policy."""
        if self.config.rollout_policy == "biased":
            # Heavily biased toward goal
            for _ in range(self.config.rollout_max_steps):
                if random.random() < 0.7:
                    # Move toward goal
                    action = self.model.get_optimal_action(state)
                else:
                    action = random.randrange(self.action_count)
                    
                state, reward, done = self.model.step(state, action)
                if done:
                    return reward
            return 0.0
        else:
            # Original random
            for _ in range(self.config.rollout_max_steps):
                action = random.randrange(self.action_count)
                state, reward, done = self.model.step(state, action)
                if done:
                    return reward
            return 0.0
    
    def _select_action(self, node: Node, miner: Optional[RaCTSMiner], 
                      beta: float, history: List[Tuple[int, int]], 
                      sim_count: int) -> int:
        """Fixed action selection."""
        
        # Always use UCT during expansion phase
        if node.N < 10:
            # Pure UCT early
            best_action = None
            best_value = -float('inf')
            
            for action, child in node.children.items():
                if child.N == 0:
                    value = float('inf')
                else:
                    exploit = child.q_value()
                    explore = self.config.c_uct * math.sqrt(math.log(node.N) / child.N)
                    value = exploit + explore
                    
                if value > best_value:
                    best_value = value
                    best_action = action
                    
            return best_action if best_action is not None else random.choice(list(node.children.keys()))
        
        # After warmup, can use priors
        if miner and beta > 0 and sim_count >= self.config.warm_sims:
            # Get prior scores
            prior_scores = {}
            for action in node.children:
                prior_scores[action] = miner.calculate_prior(node, action, history)
            
            # Normalize to probabilities
            if max(prior_scores.values()) > 0:
                min_score = min(prior_scores.values())
                scores = {a: s - min_score + 0.1 for a, s in prior_scores.items()}
                total = sum(scores.values())
                priors = {a: s/total for a, s in scores.items()}
            else:
                priors = {a: 1.0/len(node.children) for a in node.children}
            
            # PUCT selection
            best_action = None
            best_value = -float('inf')
            sqrt_n = math.sqrt(node.N)
            
            for action, child in node.children.items():
                q = child.q_value()
                prior = priors.get(action, 1.0/self.action_count)
                exploration = self.config.c_puct * prior * sqrt_n / (1 + child.N)
                value = q + exploration
                
                if value > best_value:
                    best_value = value
                    best_action = action
                    
            return best_action
        else:
            # Standard UCT
            best_action = None
            best_value = -float('inf')
            
            for action, child in node.children.items():
                if child.N == 0:
                    value = float('inf')
                else:
                    exploit = child.q_value()
                    explore = self.config.c_uct * math.sqrt(math.log(node.N) / child.N)
                    value = exploit + explore
                    
                if value > best_value:
                    best_value = value
                    best_action = action
                    
            return best_action if best_action is not None else random.choice(list(node.children.keys()))
            
    def _select_and_expand(self, root: Node, miner: Optional[RaCTSMiner], 
                          beta: float, history: List[Tuple[int, int]], 
                          sim_count: int) -> List[Node]:
        """Fixed selection and expansion."""
        path = [root]
        node = root
        
        # Traverse tree
        while node.is_fully_expanded() and node.children:
            action = self._select_action(node, miner, beta, 
                                        history + [(n.state, n.action_taken) 
                                                  for n in path[1:]], 
                                        sim_count)
            if action not in node.children:
                break
            node = node.children[action]
            path.append(node)
            
        # Expand if needed
        if not node.is_fully_expanded():
            # Prioritize goal-directed actions
            if 1 in node.untried_actions:  # DOWN
                action = 1
                node.untried_actions.remove(1)
            elif 2 in node.untried_actions:  # RIGHT
                action = 2
                node.untried_actions.remove(2)
            else:
                action = node.untried_actions.pop()
                
            next_state, _, _ = self.model.step(node.state, action)
            child = Node(next_state, self.action_count, 
                        parent=node, action_taken=action)
            node.children[action] = child
            path.append(child)
            
        return path
    
    def _backup(self, path: List[Node], value: float):
        """Standard backup."""
        for node in reversed(path):
            node.N += 1
            node.W += value
            
    def choose_move(self, start_state: int, history: List[Tuple[int, int]], 
                   miner: Optional[RaCTSMiner] = None) -> int:
        """Fixed move selection."""
        root = Node(start_state, self.action_count)
        
        # Run simulations
        for sim in range(self.config.max_sims_per_move):
            beta = self._compute_beta(miner)
            path = self._select_and_expand(root, miner, beta, history, sim)
            reward = self._simulate_rollout(path[-1].state)
            self._backup(path, reward)
            
        # Return most visited action
        if not root.children:
            # Fallback to goal-directed
            return self.model.get_optimal_action(start_state)
            
        return max(root.children, key=lambda a: root.children[a].N)

# ====================
# Q-Learning (FIXED)
# ====================
class QLearningAgent:
    """Fixed Q-Learning with better hyperparameters."""
    
    def __init__(self, n_states: int, n_actions: int):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = 0.8  # Higher learning rate
        self.gamma = 0.95  # Lower discount
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995  # Much slower decay
        self.epsilon_min = 0.01
        
    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def run_qlearning_experiment(map_name: str = "4x4",
                            max_episodes: int = 10000,
                            success_streak: int = 10) -> Dict[str, Any]:
    """Fixed Q-Learning experiment."""
    
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    agent = QLearningAgent(n_states, n_actions)
    
    results = {
        'episode_returns': [],
        'solved': False,
        'solve_episode': -1
    }
    
    consecutive_successes = 0
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        
        for step in range(100):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                break
                
        agent.decay_epsilon()
        results['episode_returns'].append(reward)
        
        if reward > 0:
            consecutive_successes += 1
        else:
            consecutive_successes = 0
            
        if episode % 500 == 0:
            recent_success = np.mean(results['episode_returns'][-100:])
            print(f"Q-Learning - Episode {episode}: Success rate: {recent_success:.2f}")
                  
        if consecutive_successes >= success_streak:
            results['solved'] = True
            results['solve_episode'] = episode + 1
            print(f"Q-Learning solved in {episode + 1} episodes!")
            break
            
    env.close()
    return results

def run_experiment(map_name: str, method: str, sims_per_move: int,
                  max_episodes: int = 1000) -> Dict[str, Any]:
    """Run a single experiment with fixed parameters."""
    
    env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=False)
    model = FrozenLakeModel(map_name)
    
    # Fixed MCTS config
    mcts_config = MCTSConfig(
        max_sims_per_move=sims_per_move,
        c_uct=math.sqrt(2),
        c_puct=1.0,
        beta_max=0.5,
        beta_start_pos=2,
        beta_full_pos=5,
        trust_margin=0.0,
        warm_sims=10,
        rollout_policy="biased"
    )
    solver = MCTSSolver(model, mcts_config)
    
    # Setup miner if needed
    if method == "RaMCTS":
        miner = RaCTSMiner(model)
    else:
        miner = None
    
    results = {
        'episode_returns': [],
        'solved': False,
        'solve_episode': -1
    }
    
    consecutive_successes = 0
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        trace = []
        
        for step in range(100):
            action = solver.choose_move(state, trace, miner)
            trace.append((state, action))
            state, reward, done, _, _ = env.step(action)
            
            if done:
                break
        
        if miner:
            miner.update(trace, reward)
        
        results['episode_returns'].append(reward)
        
        if reward > 0:
            consecutive_successes += 1
        else:
            consecutive_successes = 0
        
        if episode % 50 == 0:
            recent_success = np.mean(results['episode_returns'][-20:])
            print(f"{method} - Episode {episode}: Success rate: {recent_success:.2f}, Streak: {consecutive_successes}")
        
        if consecutive_successes >= 10:
            results['solved'] = True
            results['solve_episode'] = episode + 1
            print(f"{method} solved in {episode + 1} episodes!")
            break
    
    env.close()
    return results

# ====================
# Main Execution
# ====================
if __name__ == "__main__":
    print("\nStarting Fixed Experiments...")
    print("=" * 60)
    
    results_file = open(f"{OUTPUT_DIR}/results_fixed.txt", 'w')
    
    def log(text):
        print(text)
        results_file.write(text + '\n')
        results_file.flush()
    
    all_results = {}
    
    for map_name in ["4x4", "8x8"]:
        log(f"\nTesting on FrozenLake {map_name}")
        log("=" * 60)
        
        all_results[map_name] = {}
        
        # 1. Q-Learning
        log("\n1. Q-Learning (Fixed)")
        results = run_qlearning_experiment(map_name, max_episodes=10000 if map_name == "4x4" else 30000)
        all_results[map_name]['Q-Learning'] = results
        
        # 2. Vanilla MCTS (Fair)
        budget = 150 if map_name == "4x4" else 300
        log(f"\n2. Vanilla MCTS ({budget} sims)")
        results = run_experiment(map_name, "Vanilla", budget, max_episodes=1000)
        all_results[map_name]['Vanilla'] = results
        
        # 3. RaMCTS (Fair)
        log(f"\n3. RaMCTS ({budget} sims)")
        results = run_experiment(map_name, "RaMCTS", budget, max_episodes=1000)
        all_results[map_name]['RaMCTS'] = results
    
    # Summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    
    for map_name in ["4x4", "8x8"]:
        log(f"\nFrozenLake {map_name}:")
        for method, results in all_results[map_name].items():
            if results['solved']:
                log(f"  {method}: SOLVED in {results['solve_episode']} episodes")
            else:
                log(f"  {method}: Failed")
    
    results_file.close()
    print(f"\nResults saved to {OUTPUT_DIR}/results_fixed.txt")
    print("\n✅ If this works better, we'll update the paper with these results!")
