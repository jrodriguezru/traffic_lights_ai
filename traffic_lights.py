import random
import copy
import util
from states import TFState
from qlearning_agents import QLearningAgent, TrafficApproximateQAgent


def run_simulation(model_type='qlearning', episodes=10, steps_per_episode=50):
    print(f"Starting simulation with model: {model_type}")
    
    # Initialize agent based on model type
    if model_type == 'qlearning':
        # Standard Q-Learning (can have some default epsilon)
        agent = QLearningAgent(alpha=0.2, epsilon=0.05, gamma=0.8)
    elif model_type == 'qlearning_epsilon':
        # Q-Learning with higher exploration
        agent = QLearningAgent(alpha=0.2, epsilon=0.3, gamma=0.8)
    elif model_type == 'approximate':
        # Approximate Q-Learning
        # Use a much smaller alpha because features (number of cars) can be large,
        # leading to large Q-values and potential divergence (NaNs).
        agent = TrafficApproximateQAgent(alpha=0.001, epsilon=0.05, gamma=0.8)
    else:
        print(f"Unknown model type: {model_type}")
        return

    for episode in range(episodes):
        # Initialize state
        # Random initial cars
        state = TFState('RED', 'GREEN', random.randint(0, 5), random.randint(0, 5))
        
        total_reward = 0
        
        for step in range(steps_per_episode):
            # 1. Get action from agent
            action = agent.getAction(state)
            
            # 2. Store current state for update (deepcopy because updateState modifies in place)
            prev_state = copy.deepcopy(state)
            
            # 3. Execute action (transition)
            state.updateState(action)
            next_state = state # state is now updated
            
            # 4. Calculate reward
            reward = next_state.getReward()
            total_reward += reward
            
            # 5. Update agent
            agent.update(prev_state, action, next_state, reward)
            
            # Optional: Print step info
            print(f"Ep {episode} Step {step}: Action={action}, Reward={reward}, State={state}")
            
        print(f"Episode {episode + 1}/{episodes} finished. Total Reward: {total_reward}")

if __name__ == '__main__':
    # Example usage
    print("--- Q-Learning ---")
    run_simulation('qlearning', episodes=5)
    
    print("\n--- Q-Learning with Epsilon ---")
    run_simulation('qlearning_epsilon', episodes=5)
    
    print("\n--- Approximate Q-Learning ---")
    run_simulation('approximate', episodes=5)
