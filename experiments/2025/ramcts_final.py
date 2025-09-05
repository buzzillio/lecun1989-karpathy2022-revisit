"""
RaMCTS: Pattern-Mined Monte Carlo Tree Search
Final implementation for paper submission
Clean version without visualizations - outputs structured logs for analysis
"""

import math
import random
import collections
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import gymnasium as gym
import json
from datetime import datetime

# ============================================================
# CORE COMPONENTS
# ============================================================

class FrozenLakeModel:
    """Deterministic FrozenLake environment model for planning"""
    
    def __init__(self, map_name="4x4"):
        self.map_name = map_name
        if map_name == "4x4":
            self.desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
            self.map_size = 4
        elif map_name == "8x8":
            self.desc = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", 
                        "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
            self.map_size = 8
        else:
            raise ValueError(f"Unknown map: {map_name}")
        
        # Precompute holes and goal
        self.holes = set()
        self.goal_state = None
        for r, row in enumerate(self.desc):
            for c, cell in enumerate(row):
                state = r * self.map_size + c
                if cell == 'H':
                    self.holes.add(state)
                elif cell == 'G':
                    self.goal_state = state
        
        self.action_count = 4
        self.actions = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    
    def step(self, state: int, action: int) -> Tuple[int, float, bool]:
        """Deterministic transition"""
        row, col = state // self.map_size, state % self.map_size
        
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
        elif next_state == self.goal_state:
            return next_state, 1.0, True
        else:
            return next_state, 0.0, False


class Node:
    """MCTS tree node"""
    
    def __init__(self, state: Any, action_count: int, parent=None, action_taken=None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: Dict[int, Node] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.untried_actions = list(range(action_count))
        random.shuffle(self.untried_actions)
    
    def q_value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0


class RaCTSMiner:
    """Pattern mining module for RaMCTS"""
    
    def __init__(self, model: FrozenLakeModel, decay: float = 0.997):
        self.model = model
        self.decay = decay
        self.min_count = 1e-3
        
        # Pattern statistics
        self.c_all = collections.Counter()  # All episodes
        self.c_pos = collections.Counter()  # Near-success episodes
        
        # Episode tracking
        self.episode_returns = []
        self.pos_updates = 0
        self.total_episodes = 0
        
        # Context weighting
        self.goal_state = model.goal_state
        self.map_size = model.map_size
    
    def _get_ngrams(self, trace: List[Tuple[int, int]], n: int) -> List[Tuple]:
        """Extract n-grams from trace"""
        if len(trace) < n:
            return []
        return [tuple(trace[i:i+n]) for i in range(len(trace) - n + 1)]
    
    def _decay_counts(self):
        """Apply exponential decay to counts"""
        for counter in [self.c_all, self.c_pos]:
            for key in list(counter.keys()):
                counter[key] *= self.decay
                if counter[key] < self.min_count:
                    del counter[key]
    
    def update(self, trace: List[Tuple[int, int]], reward: float):
        """Update pattern statistics from episode"""
        self._decay_counts()
        self.total_episodes += 1
        self.episode_returns.append(reward)
        
        # Keep only recent returns for threshold calculation
        if len(self.episode_returns) > 100:
            self.episode_returns = self.episode_returns[-100:]
        
        # Extract all n-grams (1 to 3)
        all_grams = []
        for n in range(1, min(4, len(trace) + 1)):
            all_grams.extend(self._get_ngrams(trace, n))
        
        # Update counts
        for gram in all_grams:
            self.c_all[gram] += 1
        
        # Determine if near-success (top 15% of returns)
        if len(self.episode_returns) >= 5:
            threshold = np.percentile(self.episode_returns, 85)
            if reward >= threshold and reward > 0:
                self.pos_updates += 1
                for gram in all_grams:
                    self.c_pos[gram] += 1
    
    def calculate_prior(self, state: int, action: int, 
                       history: List[Tuple[int, int]]) -> float:
        """Calculate prior score for state-action pair"""
        # Build candidate n-grams ending with (state, action)
        candidates = [((state, action),)]  # 1-gram
        
        if history:
            # 2-gram
            candidates.append((history[-1], (state, action)))
            if len(history) >= 2:
                # 3-gram
                candidates.append((history[-2], history[-1], (state, action)))
        
        # Calculate scores
        total_score = 0.0
        for gram in candidates:
            if gram in self.c_pos:
                # NPMI × IDF scoring
                pos_count = self.c_pos[gram]
                all_count = max(self.c_all[gram], 1.0)
                
                # Calculate NPMI components
                p_pos = sum(self.c_pos.values()) / max(sum(self.c_all.values()), 1.0)
                p_gram = all_count / max(sum(self.c_all.values()), 1.0)
                p_joint = pos_count / max(sum(self.c_all.values()), 1.0)
                
                if p_joint > 0 and p_pos > 0 and p_gram > 0:
                    pmi = math.log(p_joint / (p_pos * p_gram + 1e-10) + 1e-10)
                    npmi = pmi / (-math.log(p_joint + 1e-10) + 1e-10)
                    
                    # IDF weighting
                    idf = math.log(self.total_episodes / (all_count + 1.0))
                    
                    # Context weight (distance to goal)
                    context_weight = self._context_weight(state)
                    
                    score = max(0, npmi) * idf * context_weight
                    total_score += score
        
        return total_score
    
    def _context_weight(self, state: int) -> float:
        """Weight based on proximity to goal"""
        if self.goal_state is None:
            return 1.0
        
        # Manhattan distance to goal
        s_row, s_col = state // self.map_size, state % self.map_size
        g_row, g_col = self.goal_state // self.map_size, self.goal_state % self.map_size
        distance = abs(s_row - g_row) + abs(s_col - g_col)
        
        # Closer to goal = higher weight
        max_distance = 2 * self.map_size
        weight = 1.0 + (1.0 - distance / max_distance)
        return weight
    
    def get_beta(self) -> float:
        """Calculate beta for prior strength (ramp-up)"""
        if self.pos_updates < 3:
            return 0.0
        elif self.pos_updates < 10:
            progress = (self.pos_updates - 3) / 7.0
            return 2.0 * progress
        else:
            return 2.0
    
    def get_top_patterns(self, k: int = 10) -> List[Tuple[Tuple, float]]:
        """Get top-k patterns by score"""
        pattern_scores = []
        
        for gram in self.c_pos:
            pos_count = self.c_pos[gram]
            all_count = max(self.c_all[gram], 1.0)
            
            p_pos = sum(self.c_pos.values()) / max(sum(self.c_all.values()), 1.0)
            p_gram = all_count / max(sum(self.c_all.values()), 1.0)
            p_joint = pos_count / max(sum(self.c_all.values()), 1.0)
            
            if p_joint > 0 and p_pos > 0 and p_gram > 0:
                pmi = math.log(p_joint / (p_pos * p_gram + 1e-10) + 1e-10)
                npmi = pmi / (-math.log(p_joint + 1e-10) + 1e-10)
                idf = math.log(self.total_episodes / (all_count + 1.0))
                score = max(0, npmi) * idf
                pattern_scores.append((gram, score))
        
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        return pattern_scores[:k]


class MCTSSolver:
    """Monte Carlo Tree Search with pattern-based priors"""
    
    def __init__(self, model: FrozenLakeModel, simulations_per_move: int = 100,
                 c_uct: float = 1.4, c_puct: float = 1.25, 
                 trust_margin: float = 0.05, warm_sims: int = 50):
        self.model = model
        self.simulations_per_move = simulations_per_move
        self.c_uct = c_uct
        self.c_puct = c_puct
        self.trust_margin = trust_margin
        self.warm_sims = warm_sims
    
    def _select(self, node: Node, miner: Optional[RaCTSMiner], 
                history: List[Tuple[int, int]], beta: float) -> Node:
        """Select best child using UCT or PUCT"""
        if not node.children:
            return node
        
        # Calculate priors if miner available
        priors = {}
        if miner and beta > 0:
            for action in node.children:
                priors[action] = miner.calculate_prior(node.state, action, history)
            
            # Softmax normalization
            if priors:
                max_prior = max(priors.values())
                exp_priors = {a: math.exp(beta * (p - max_prior)) 
                             for a, p in priors.items()}
                sum_exp = sum(exp_priors.values())
                priors = {a: exp_p / sum_exp for a, exp_p in exp_priors.items()}
        
        best_value = -float('inf')
        best_action = None
        uct_best_value = -float('inf')
        uct_best_action = None
        
        sqrt_parent = math.sqrt(node.visit_count)
        log_parent = math.log(node.visit_count + 1)
        
        for action, child in node.children.items():
            # UCT value (always calculated)
            if child.visit_count == 0:
                uct_value = float('inf')
            else:
                exploitation = child.q_value()
                exploration = self.c_uct * math.sqrt(log_parent / child.visit_count)
                uct_value = exploitation + exploration
            
            if uct_value > uct_best_value:
                uct_best_value = uct_value
                uct_best_action = action
            
            # PUCT value (if priors available)
            if miner and beta > 0:
                if child.visit_count == 0:
                    puct_value = float('inf')
                else:
                    exploitation = child.q_value()
                    prior = priors.get(action, 1.0 / len(node.children))
                    exploration = self.c_puct * prior * sqrt_parent / (1 + child.visit_count)
                    puct_value = exploitation + exploration
                
                if puct_value > best_value:
                    best_value = puct_value
                    best_action = action
            else:
                # No priors, use UCT
                if uct_value > best_value:
                    best_value = uct_value
                    best_action = action
        
        # Trust margin: only use PUCT if clearly better
        if miner and beta > 0 and node.visit_count >= self.warm_sims:
            if best_value >= uct_best_value - self.trust_margin:
                return node.children[best_action]
        
        return node.children[uct_best_action]
    
    def _expand(self, node: Node, model: FrozenLakeModel) -> Node:
        """Expand a node by trying an untried action"""
        if not node.untried_actions:
            return node
        
        action = node.untried_actions.pop()
        next_state, _, _ = model.step(node.state, action)
        child = Node(next_state, model.action_count, parent=node, action_taken=action)
        node.children[action] = child
        return child
    
    def _simulate(self, state: int, model: FrozenLakeModel) -> float:
        """Random rollout from state"""
        current_state = state
        for _ in range(100):  # Max rollout length
            action = random.randrange(model.action_count)
            next_state, reward, done = model.step(current_state, action)
            if done:
                return reward
            current_state = next_state
        return 0.0
    
    def _backup(self, node: Node, value: float):
        """Propagate value up the tree"""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
    
    def choose_move(self, state: int, miner: Optional[RaCTSMiner] = None,
                   history: List[Tuple[int, int]] = None) -> int:
        """Choose best action using MCTS"""
        if history is None:
            history = []
        
        root = Node(state, self.model.action_count)
        beta = miner.get_beta() if miner else 0.0
        
        for sim in range(self.simulations_per_move):
            node = root
            sim_history = history.copy()
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = self._select(node, miner, sim_history, beta)
                if node.parent and node.action_taken is not None:
                    sim_history.append((node.parent.state, node.action_taken))
            
            # Expansion
            if not node.is_fully_expanded():
                node = self._expand(node, self.model)
            
            # Simulation
            value = self._simulate(node.state, self.model)
            
            # Backup
            self._backup(node, value)
        
        # Choose most visited action
        if not root.children:
            return random.randrange(self.model.action_count)
        
        return max(root.children.keys(), key=lambda a: root.children[a].visit_count)


# ============================================================
# BASELINE ALGORITHMS
# ============================================================

def run_q_learning(env_name: str, episodes: int = 5000, target_streak: int = 10) -> dict:
    """Q-Learning baseline"""
    env = gym.make('FrozenLake-v1', map_name=env_name, is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    Q = np.zeros((n_states, n_actions))
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    results = {
        'rewards': [],
        'streak': 0,
        'solved_at': None
    }
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _, _ = env.step(action)
            
            # Q-Learning update
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            
            state = next_state
            total_reward += reward
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        results['rewards'].append(total_reward)
        
        # Check streak
        if total_reward > 0:
            results['streak'] += 1
            if results['streak'] >= target_streak and results['solved_at'] is None:
                results['solved_at'] = episode + 1
        else:
            results['streak'] = 0
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(results['rewards'][-100:])
            print(f"  Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, "
                  f"Epsilon = {epsilon:.3f}, Streak = {results['streak']}")
    
    env.close()
    return results


def run_vanilla_mcts(env_name: str, simulations: int, episodes: int = 500, 
                     target_streak: int = 10) -> dict:
    """Vanilla MCTS baseline"""
    model = FrozenLakeModel(env_name)
    env = gym.make('FrozenLake-v1', map_name=env_name, is_slippery=False)
    solver = MCTSSolver(model, simulations_per_move=simulations)
    
    results = {
        'rewards': [],
        'streak': 0,
        'solved_at': None
    }
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = solver.choose_move(state, miner=None)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        
        results['rewards'].append(total_reward)
        
        # Check streak
        if total_reward > 0:
            results['streak'] += 1
            if results['streak'] >= target_streak and results['solved_at'] is None:
                results['solved_at'] = episode + 1
        else:
            results['streak'] = 0
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(results['rewards'][-10:])
            print(f"  Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, "
                  f"Streak = {results['streak']}")
    
    env.close()
    return results


def run_ramcts(env_name: str, simulations: int, episodes: int = 500, 
               target_streak: int = 10) -> dict:
    """RaMCTS with pattern mining"""
    model = FrozenLakeModel(env_name)
    env = gym.make('FrozenLake-v1', map_name=env_name, is_slippery=False)
    solver = MCTSSolver(model, simulations_per_move=simulations)
    miner = RaCTSMiner(model)
    
    results = {
        'rewards': [],
        'streak': 0,
        'solved_at': None,
        'patterns': [],
        'beta_values': []
    }
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        trace = []
        history = []
        
        while not done:
            action = solver.choose_move(state, miner=miner, history=history)
            trace.append((state, action))
            history.append((state, action))
            if len(history) > 3:
                history = history[-3:]
            
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        
        # Update miner
        miner.update(trace, total_reward)
        
        results['rewards'].append(total_reward)
        results['beta_values'].append(miner.get_beta())
        
        # Check streak
        if total_reward > 0:
            results['streak'] += 1
            if results['streak'] >= target_streak and results['solved_at'] is None:
                results['solved_at'] = episode + 1
                print(f"  → Solved at episode {episode + 1}!")
                # Store final patterns before breaking
                top_patterns = miner.get_top_patterns(5)
                results['patterns'].append({
                    'episode': episode + 1,
                    'patterns': [(str(p), float(s)) for p, s in top_patterns]
                })
                break  # EARLY STOPPING!
        else:
            results['streak'] = 0
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(results['rewards'][-10:])
            print(f"  Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, "
                  f"Beta = {miner.get_beta():.2f}, Streak = {results['streak']}")
            
            # Store top patterns
            if (episode + 1) % 50 == 0:
                top_patterns = miner.get_top_patterns(5)
                results['patterns'].append({
                    'episode': episode + 1,
                    'patterns': [(str(p), float(s)) for p, s in top_patterns]
                })
    
    env.close()
    return results


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():
    """Run complete benchmark"""
    
    print("=" * 60)
    print("RaMCTS BENCHMARK - FINAL VERSION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    configs = {
        '4x4': {
            'q_episodes': 5000,
            'vanilla_sims': 400,
            'ramcts_sims': 80,
            'max_episodes': 500
        },
        '8x8': {
            'q_episodes': 30000,
            'vanilla_sims': 1200,
            'ramcts_sims': 150,
            'max_episodes': 500
        }
    }
    
    all_results = {}
    
    for map_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"TESTING ON FROZENLAKE {map_name}")
        print(f"{'='*60}\n")
        
        results = {}
        
        # Q-Learning
        print(f"[1/3] Running Q-Learning ({config['q_episodes']} episodes max)...")
        q_results = run_q_learning(map_name, config['q_episodes'])
        results['q_learning'] = q_results
        print(f"  → {'SOLVED' if q_results['solved_at'] else 'FAILED'}")
        if q_results['solved_at']:
            print(f"     Solved in {q_results['solved_at']} episodes")
        print()
        
        # Vanilla MCTS
        print(f"[2/3] Running Vanilla MCTS ({config['vanilla_sims']} sims/move)...")
        vanilla_results = run_vanilla_mcts(map_name, config['vanilla_sims'], 
                                          config['max_episodes'])
        results['vanilla'] = vanilla_results
        print(f"  → {'SOLVED' if vanilla_results['solved_at'] else 'FAILED'}")
        if vanilla_results['solved_at']:
            print(f"     Solved in {vanilla_results['solved_at']} episodes")
            print(f"     Final success rate: {np.mean(vanilla_results['rewards'][-20:]):.2%}")
        print()
        
        # RaMCTS
        print(f"[3/3] Running RaMCTS ({config['ramcts_sims']} sims/move)...")
        ramcts_results = run_ramcts(map_name, config['ramcts_sims'], 
                                   config['max_episodes'])
        results['ramcts'] = ramcts_results
        print(f"  → {'SOLVED' if ramcts_results['solved_at'] else 'FAILED'}")
        if ramcts_results['solved_at']:
            print(f"     Solved in {ramcts_results['solved_at']} episodes")
            print(f"     Final success rate: {np.mean(ramcts_results['rewards'][-20:]):.2%}")
        print()
        
        # Print discovered patterns
        if ramcts_results['patterns']:
            print("  Top Discovered Patterns:")
            latest_patterns = ramcts_results['patterns'][-1]['patterns']
            for i, (pattern, score) in enumerate(latest_patterns[:5], 1):
                print(f"    {i}. Score {score:.3f}: {pattern}")
        
        all_results[map_name] = results
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for map_name in ['4x4', '8x8']:
        print(f"\nFrozenLake {map_name}:")
        
        methods = [
            ('Q-Learning', 'q_learning'),
            ('Vanilla', 'vanilla'),
            ('RaMCTS', 'ramcts')
        ]
        
        for method_name, method_key in methods:
            result = all_results[map_name][method_key]
            if result['solved_at']:
                print(f"  {method_name}: SOLVED in {result['solved_at']} episodes")
            else:
                print(f"  {method_name}: Failed")
    
    # Output JSON for external visualization
    print("\n" + "=" * 60)
    print("JSON OUTPUT FOR VISUALIZATION")
    print("=" * 60)
    
    # Prepare data for JSON export
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for map_name, map_results in all_results.items():
        json_data['results'][map_name] = {
            'q_learning': {
                'solved_at': map_results['q_learning']['solved_at'],
                'final_streak': map_results['q_learning']['streak'],
                'rewards': map_results['q_learning']['rewards'][-100:]  # Last 100
            },
            'vanilla': {
                'solved_at': map_results['vanilla']['solved_at'],
                'final_streak': map_results['vanilla']['streak'],
                'rewards': map_results['vanilla']['rewards']
            },
            'ramcts': {
                'solved_at': map_results['ramcts']['solved_at'],
                'final_streak': map_results['ramcts']['streak'],
                'rewards': map_results['ramcts']['rewards'],
                'beta_values': map_results['ramcts']['beta_values'],
                'patterns': map_results['ramcts']['patterns']
            }
        }
    
    print(json.dumps(json_data, indent=2))
    
    return all_results


if __name__ == "__main__":
    results = main()
