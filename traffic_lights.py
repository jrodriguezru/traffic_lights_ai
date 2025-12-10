import random
import copy
import util
from ui import TrafficLightUI
import tkinter as tk
from states import TFState
from qlearning_agents import QLearningAgent, TrafficApproximateQAgent

def plot_results(history, switch_counts):
    """
    The function `plot_results` generates plots for North-South and East-West cars waiting per tick, and
    a bar chart showing the number of switches per episode.
    
    :param history: The `history` parameter in the `plot_results` function is a list of dictionaries
    where each dictionary represents data for a specific episode. Each dictionary contains keys 'ns' and
    'ew' which represent the number of cars waiting in the North-South direction and East-West direction
    respectively at each time step
    :param switch_counts: The `switch_counts` parameter in the `plot_results` function is a list that
    contains the number of switches made in each episode of a simulation. The function uses this data to
    create a bar chart showing the number of switches per episode. Each bar in the chart represents the
    number of switches made in
    :return: The `plot_results` function is returning a visualization of the training history and switch
    counts. It generates line plots for the average number of cars waiting in the North-South and
    East-West directions per tick, with data smoothed by averaging every `window_size` ticks. It also
    includes a bar chart showing the number of switches made per episode. The function displays these
    plots using Matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is required for plotting. Please install it: pip install matplotlib")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    window_size = 100

    for i, data in enumerate(history):
        # Smooth data by averaging every window_size ticks
        ns_raw = data['ns']
        ew_raw = data['ew']
        
        ns_smoothed = [sum(ns_raw[j:j+window_size])/len(ns_raw[j:j+window_size]) for j in range(0, len(ns_raw), window_size)]
        ew_smoothed = [sum(ew_raw[j:j+window_size])/len(ew_raw[j:j+window_size]) for j in range(0, len(ew_raw), window_size)]
        
        # Generate x-axis values corresponding to the start of each window
        x_axis = range(0, len(ns_raw), window_size)

        label = f'Episode {i+1}'
        if i == len(history) - 1:
            label = 'Test Episode'

        ax1.plot(x_axis, ns_smoothed, label=label)
        ax2.plot(x_axis, ew_smoothed, label=label)
        
    # Plot for NS waiting cars by tick (windowed average)
    ax1.set_title(f'North-South Cars Waiting per Tick (Avg every {window_size} ticks)')
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Cars')
    ax1.legend()
    ax1.grid(True)
    
    # Plot for EW waiting cars by tick (windowed average)
    ax2.set_title(f'East-West Cars Waiting per Tick (Avg every {window_size} ticks)')
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
    
    tick_labels = [str(e) for e in episodes]
    if len(tick_labels) > 1:
        tick_labels[-1] = 'Test'
    
    ax3.set_xticks(episodes)
    ax3.set_xticklabels(tick_labels)
    ax3.grid(axis='y')
    
    plt.tight_layout()
    plt.show()

def run_simulation(model_type='qlearning', episodes=10, steps_per_episode=50, reward_type='initial', use_gui=False):
    """
    This function `run_simulation` runs a traffic simulation using different reinforcement
    learning models and can display the simulation in a GUI or non-GUI mode.
    
    :param model_type: The `model_type` parameter in the `run_simulation` function specifies the type of
    reinforcement learning model to use for the simulation. It can take on different values:, defaults
    to qlearning (optional)

    :param episodes: The `episodes` parameter in the `run_simulation` function specifies the number of
    episodes to run during the simulation. Each episode represents a complete cycle of interactions
    between the agent and the environment. The agent learns from these interactions and updates its
    strategy based on the rewards received, defaults to 10 (optional)
    
    :param steps_per_episode: The `steps_per_episode` parameter specifies the maximum number of steps
    (time ticks) that the simulation will run for each episode. It determines how long each episode will
    last in terms of time steps. This parameter is used to control the duration of each episode in the
    simulation, defaults to 50 (optional)
    
    :param reward_type: The `reward_type` parameter in the `run_simulation` function determines the type
    of reward calculation used in the simulation. It can have different values based on the specific
    implementation of the simulation. In the provided code, the `reward_type` is used to initialize the
    state with a specific reward type in, defaults to initial (optional)
    
    :param use_gui: The `use_gui` parameter in the `run_simulation` function determines whether the
    simulation will be run with a graphical user interface (GUI) or not. If `use_gui` is set to `True`,
    the simulation will be displayed and interacted with using a GUI interface. The parameters `use_gui, 
    defaults to False (optional)
    
    :return: The `run_simulation` function does not explicitly return any value. It either runs the
    simulation with the specified parameters or prints out information during the simulation process.
    The function is designed to perform a simulation based on the provided inputs and does not have a
    specific return value.
    """
    print(f"Starting simulation with model: {model_type}")
    
    # Initialize agent based on model type
    if model_type == 'qlearning':
        # Standard Q-Learning (can have some default epsilon)
        agent = QLearningAgent(alpha=0.2, epsilon=0.05, gamma=0.8, ticks_per_episode=steps_per_episode)
    elif model_type == 'qlearning_epsilon':
        # Q-Learning with higher exploration
        agent = QLearningAgent(alpha=0.2, epsilon=0.3, gamma=0.8)
    elif model_type == 'approximate':
        # Approximate Q-Learning
        # Using a much smaller alpha because features (number of cars) can be large,
        # leading to large Q-values and potential divergence (NaNs).
        agent = TrafficApproximateQAgent(alpha=0.001, epsilon=0.05, gamma=0.8, ticks_per_episode=steps_per_episode)
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
            """
            Runs the simulation step by step in the GUI.
            """
            if sim_state['episode'] > episodes:
                # If no more episodes
                print("All episodes finished.")
                root.destroy()
                plot_results(sim_state['history'], sim_state['switch_history'])
                return

            if sim_state['step'] >= steps_per_episode:
                # End of episode
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
                
                if sim_state['episode'] == episodes:
                    # If last training episode, now set for testing
                    print("Starting Test Episode (No Learning, No Exploration)")
                    agent.epsilon = 0.0
                    agent.alpha = 0.0

                root.after(10, step_simulation)
                return

            # Logic
            state = sim_state['state']
            
            # Record data
            sim_state['current_data']['ns'].append(state.num_cars_waiting_ns)
            sim_state['current_data']['ew'].append(state.num_cars_waiting_ew)
            
            # Get action from agent
            action = agent.getAction(state)
            if action == 'SWITCH':
                sim_state['current_switches'] += 1
                
            # Updates state, gets reward, and updates agent
            prev_state = copy.deepcopy(state)
            state.updateState(action)
            next_state = state
            reward = next_state.getReward()
            sim_state['total_reward'] += reward
            agent.update(prev_state, action, next_state, reward)
            
            ui.update(state)
            sim_state['step'] += 1
            
            # Delay (100ms)
            root.after(10, step_simulation)

        root.after(100, step_simulation)
        root.mainloop()
        return

    # NON GUI MODE
    # Variables to store history for plotting
    history = []
    switch_history = []

    for episode in range(episodes + 1):
        if episode == episodes:
            # Last training episode, now set for testing
            print("Starting Test Episode (No Learning, No Exploration)")
            agent.epsilon = 0.0
            agent.alpha = 0.0

        # Initialize state
        # Random initial cars
        state = TFState('RED', 'GREEN', random.randint(0, 5), random.randint(0, 5), reward_type)
        
        total_reward = 0
        current_data = {'ns': [], 'ew': []}
        current_switches = 0
        
        for step in range(steps_per_episode):
            # Record data
            current_data['ns'].append(state.num_cars_waiting_ns)
            current_data['ew'].append(state.num_cars_waiting_ew)

            # 1. Get action from agent
            action = agent.getAction(state)
            
            if action == 'SWITCH':
                current_switches += 1
            
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
            # print(f"Ep {episode} Step {step}: Action={action}, Reward={reward}, State={state}")
            
        history.append(current_data)
        switch_history.append(current_switches)
        
        if episode < episodes:
            print(f"Episode {episode + 1}/{episodes} finished. Total Waiting Cars: {total_reward}")
        else:
            print(f"Test Episode finished. Total Waiting Cars: {total_reward}")
    

    plot_results(history, switch_history)

if __name__ == '__main__':
    # Example usage
    # print("--- Q-Learning ---")
    # run_simulation('qlearning', episodes=5, steps_per_episode=100, reward_type='penalty', use_gui=True)
    
    print("\n--- Q-Learning with Epsilon ---")
    run_simulation('qlearning_epsilon', episodes=5, steps_per_episode=4320, reward_type='initial', use_gui=False)

    # print("\n--- Approximate Q-Learning with GUI ---")
    # run_simulation('approximate', episodes=5, steps_per_episode=100, reward_type='balanced', use_gui=True)
