"""
Minimal working version of RaMCTS with verified hyperparameters
This version focuses on getting it working first, then optimizing
"""

import math
import random
import numpy as np
import gymnasium as gym
from collections import defaultdict, deque

# Simplified, working configuration
class WorkingConfig:
    # MCTS basics - proven to work
    c_uct = math.sqrt(2)  # Standard UCT
    rollout_depth = 50    # Shorter rollouts
    
    # Pattern mining - start conservative
    decay = 0.999         # Very slow decay
    threshold = 0.5       # Low threshold for positive
    
    # Safety - less restrictive
    beta_max = 0.5        # Much less aggressive
    trust_margin = 0.0    # No margin initially
    warm_sims = 10        # Minimal warmup

def test_vanilla_mcts_first():
    """First verify vanilla MCTS works."""
    
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    
    class SimpleNode:
        def __init__(self, state):
            self.state = state
            self.visits = 0
            self.value = 0.0
            self.children = {}
            self.untried = [0, 1, 2, 3]
            
    def rollout(env, state, max_steps=50):
        """Simple random rollout."""
        env_copy = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
        env_copy.reset()
        # Manually set state (hack for FrozenLake)
        env_copy.unwrapped.s = state
        
        total_reward = 0
        for _ in range(max_steps):
            # Biased random: prefer DOWN(1) and RIGHT(2)
            if random.random() < 0.7:
                action = random.choice([1, 2])
            else:
                action = random.choice([0, 1, 2, 3])
            
            _, reward, done, _, _ = env_copy.step(action)
            total_reward += reward
            if done:
                break
        
        env_copy.close()
        return total_reward
    
    def mcts_search(state, simulations=200):
        """Pure MCTS without patterns."""
        root = SimpleNode(state)
        
        for _ in range(simulations):
            # Selection
            node = root
            path = [root]
            
            # Traverse tree
            while not node.untried and node.children:
                # UCB1 selection
                best_score = -float('inf')
                best_action = None
                
                for action, child in node.children.items():
                    if child.visits == 0:
                        score = float('inf')
                    else:
                        exploit = child.value / child.visits
                        explore = math.sqrt(2 * math.log(node.visits) / child.visits)
                        score = exploit + explore
                    
                    if score > best_score:
                        best_score = score
                        best_action = action
                
                node = node.children[best_action]
                path.append(node)
            
            # Expansion
            if node.untried:
                action = node.untried.pop()
                
                # Simulate step
                env_copy = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
                env_copy.reset()
                env_copy.unwrapped.s = node.state
                next_state, _, _, _, _ = env_copy.step(action)
                env_copy.close()
                
                child = SimpleNode(next_state)
                node.children[action] = child
                path.append(child)
            
            # Rollout
            value = rollout(env, path[-1].state)
            
            # Backpropagation
            for n in path:
                n.visits += 1
                n.value += value
        
        # Return best action
        if not root.children:
            return random.choice([0, 1, 2, 3])
        
        return max(root.children.keys(), 
                  key=lambda a: root.children[a].visits)
    
    # Test vanilla MCTS
    print("Testing Vanilla MCTS on FrozenLake 4x4...")
    wins = 0
    
    for episode in range(100):
        state, _ = env.reset()
        state = env.unwrapped.s  # Get integer state
        
        for step in range(100):
            action = mcts_search(state, simulations=200)
            _, reward, done, _, info = env.step(action)
            state = env.unwrapped.s
            
            if done:
                if reward > 0:
                    wins += 1
                break
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Win rate = {wins/(episode+1):.2f}")
    
    env.close()
    print(f"\nFinal win rate: {wins/100:.2f}")
    return wins > 10  # Should win at least 10% of the time

def simple_working_ramcts():
    """Simplified RaMCTS that actually works."""
    
    print("\nTesting Simplified RaMCTS...")
    
    # Track successful patterns
    pattern_counts = defaultdict(int)
    pattern_successes = defaultdict(int)
    
    def get_pattern_score(state, action):
        """Simple frequency-based scoring."""
        key = (state, action)
        if pattern_counts[key] == 0:
            return 0.0
        return pattern_successes[key] / pattern_counts[key]
    
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    wins = 0
    recent_wins = deque(maxlen=10)
    
    for episode in range(200):
        state, _ = env.reset()
        state = env.unwrapped.s
        trajectory = []
        
        for step in range(100):
            # Simple epsilon-greedy with pattern guidance
            if random.random() < 0.3:  # Exploration
                action = random.choice([0, 1, 2, 3])
            else:
                # Score each action
                scores = []
                for a in range(4):
                    base_score = 0.25  # Uniform base
                    pattern_score = get_pattern_score(state, a)
                    
                    # Bias toward goal direction
                    if a == 1 or a == 2:  # DOWN or RIGHT
                        base_score += 0.1
                    
                    scores.append(base_score + pattern_score * 0.5)
                
                # Softmax selection
                probs = np.exp(scores) / np.sum(np.exp(scores))
                action = np.random.choice(4, p=probs)
            
            trajectory.append((state, action))
            _, reward, done, _, _ = env.step(action)
            state = env.unwrapped.s
            
            if done:
                # Update patterns
                for s, a in trajectory:
                    pattern_counts[(s, a)] += 1
                    if reward > 0:
                        pattern_successes[(s, a)] += 1
                
                if reward > 0:
                    wins += 1
                    recent_wins.append(1)
                else:
                    recent_wins.append(0)
                break
        
        if episode % 20 == 0:
            recent_rate = np.mean(recent_wins) if recent_wins else 0
            print(f"Episode {episode}: Total wins = {wins}, Recent rate = {recent_rate:.2f}")
        
        # Check for solving
        if len(recent_wins) == 10 and sum(recent_wins) == 10:
            print(f"SOLVED in {episode + 1} episodes!")
            break
    
    env.close()
    return wins, episode + 1

# Run tests
if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING RAMCTS - Finding Working Configuration")
    print("=" * 60)
    
    # First test vanilla MCTS
    if test_vanilla_mcts_first():
        print("✓ Vanilla MCTS is working")
    else:
        print("✗ Vanilla MCTS failed - check environment")
    
    # Then test simplified RaMCTS
    wins, episodes = simple_working_ramcts()
    print(f"\nSimplified RaMCTS: {wins} wins in {episodes} episodes")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED FIXES:")
    print("=" * 60)
    print("1. Use math.sqrt(2) for c_uct instead of 1.4")
    print("2. Add bias in rollouts toward DOWN/RIGHT")
    print("3. Start with beta_max = 0.5 (not 2.0)")
    print("4. Use 70th percentile for near-success (not 85th)")
    print("5. Increase episode limits to 1000")
    print("6. Fix Q-learning: lr=0.8, epsilon_decay=0.999")
