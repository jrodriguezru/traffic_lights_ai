import random
import copy
import util
from ui import TrafficLightUI
import tkinter as tk
from states import TFState
from qlearning_agents import QLearningAgent, TrafficApproximateQAgent

def plot_results(history, switch_counts):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required for plotting. Please install it: pip install matplotlib")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    for i, data in enumerate(history):
        ax1.plot(data['ns'], label=f'Episode {i+1}')
        ax2.plot(data['ew'], label=f'Episode {i+1}')
        
    ax1.set_title('North-South Cars Waiting per Tick')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Cars')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('East-West Cars Waiting per Tick')
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Cars')
    ax2.legend()
    ax2.grid(True)

    # Bar chart for switches
    episodes = range(1, len(switch_counts) + 1)
    ax3.bar(episodes, switch_counts, color='skyblue')
    ax3.set_title('Number of Switches per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Switches')
    ax3.set_xticks(episodes)
    ax3.grid(axis='y')
    
    plt.tight_layout()
    plt.show()

def run_simulation(model_type='qlearning', episodes=10, steps_per_episode=50, reward_type='initial', use_gui=False):
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

    if use_gui:
        root = tk.Tk()
        ui = TrafficLightUI(root)
        
        # State variables for the GUI loop
        sim_state = {
            'episode': 0,
            'step': 0,
            'state': TFState('RED', 'GREEN', random.randint(0, 5), random.randint(0, 5), reward_type),
            'total_reward': 0,
            'history': [],
            'current_data': {'ns': [], 'ew': []},
            'switch_history': [],
            'current_switches': 0
        }
        
        def step_simulation():
            if sim_state['episode'] >= episodes:
                print("All episodes finished.")
                root.destroy()
                plot_results(sim_state['history'], sim_state['switch_history'])
                return

            if sim_state['step'] >= steps_per_episode:
                print(f"Episode {sim_state['episode'] + 1}/{episodes} finished. Total Reward: {sim_state['total_reward']}")
                
                # Save episode data
                sim_state['history'].append(sim_state['current_data'])
                sim_state['current_data'] = {'ns': [], 'ew': []}
                
                # Save switch data
                sim_state['switch_history'].append(sim_state['current_switches'])
                sim_state['current_switches'] = 0
                
                sim_state['episode'] += 1
                sim_state['step'] = 0
                sim_state['state'] = TFState('RED', 'GREEN', random.randint(0, 5), random.randint(0, 5), reward_type)
                
                sim_state['total_reward'] = 0
                root.after(10, step_simulation)
                return

            # Logic
            state = sim_state['state']
            
            # Record data
            sim_state['current_data']['ns'].append(state.num_cars_waiting_ns)
            sim_state['current_data']['ew'].append(state.num_cars_waiting_ew)
            
            action = agent.getAction(state)
            if action == 'SWITCH':
                sim_state['current_switches'] += 1
                
            prev_state = copy.deepcopy(state)
            state.updateState(action)
            next_state = state
            reward = next_state.getReward()
            sim_state['total_reward'] += reward
            agent.update(prev_state, action, next_state, reward)
            
            ui.update(state)
            sim_state['step'] += 1
            
            # Delay (100ms)
            root.after(100, step_simulation)

        root.after(100, step_simulation)
        root.mainloop()
        return

    for episode in range(episodes):
        # Initialize state
        # Random initial cars
        state = TFState('RED', 'GREEN', random.randint(0, 5), random.randint(0, 5), reward_type)
        
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
    # print("--- Q-Learning ---")
    # run_simulation('qlearning', episodes=5, steps_per_episode=100, reward_type='penalty', use_gui=True)
    
    # print("\n--- Q-Learning with Epsilon ---")
    # run_simulation('qlearning_epsilon', episodes=5, steps_per_episode=100, reward_type='balanced', use_gui=True)

    print("\n--- Approximate Q-Learning with GUI ---")
    run_simulation('approximate', episodes=5, steps_per_episode=100, reward_type='balanced', use_gui=True)
