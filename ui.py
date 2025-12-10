import tkinter as tk
import copy

class TrafficLightUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Light Simulation")
        
        # Canvas for drawing the intersection
        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack()
        
        # Labels for stats
        self.stats_frame = tk.Frame(root)
        self.stats_frame.pack(fill=tk.X)
        
        self.tick_label = tk.Label(self.stats_frame, text="Tick: 0", font=("Arial", 14))
        self.tick_label.pack(side=tk.LEFT, padx=10)
        
        self.ns_cars_label = tk.Label(self.stats_frame, text="NS Cars: 0", font=("Arial", 12))
        self.ns_cars_label.pack(side=tk.LEFT, padx=10)
        
        self.ew_cars_label = tk.Label(self.stats_frame, text="EW Cars: 0", font=("Arial", 12))
        self.ew_cars_label.pack(side=tk.LEFT, padx=10)

        # Draw static road elements
        self.draw_roads()
        
        # Placeholders for dynamic elements
        self.ns_light_id = None
        self.ew_light_id = None
        self.ns_cars_text_id = None
        self.ew_cars_text_id = None

    def draw_roads(self):
        # Vertical Road (NS)
        self.canvas.create_rectangle(150, 0, 250, 400, fill='gray')
        self.canvas.create_line(200, 0, 200, 400, fill='yellow', dash=(10, 10))
        
        # Horizontal Road (EW)
        self.canvas.create_rectangle(0, 150, 400, 250, fill='gray')
        self.canvas.create_line(0, 200, 400, 200, fill='yellow', dash=(10, 10))
        
        # Intersection center
        self.canvas.create_rectangle(150, 150, 250, 250, fill='darkgray')

    def update(self, state):
        # Update Labels
        self.tick_label.config(text=f"Tick: {state.tick}")
        self.ns_cars_label.config(text=f"NS Cars: {state.num_cars_waiting_ns}")
        self.ew_cars_label.config(text=f"EW Cars: {state.num_cars_waiting_ew}")
        
        # Update Lights
        ns_color = state.light_color_ns.lower()
        ew_color = state.light_color_ew.lower()
        
        # NS Light (Top)
        if self.ns_light_id: self.canvas.delete(self.ns_light_id)
        self.ns_light_id = self.canvas.create_oval(180, 100, 220, 140, fill=ns_color, outline='black', width=2)
        
        # EW Light (Left)
        if self.ew_light_id: self.canvas.delete(self.ew_light_id)
        self.ew_light_id = self.canvas.create_oval(100, 180, 140, 220, fill=ew_color, outline='black', width=2)

        # Visualize Cars (Simple text count on road)
        if self.ns_cars_text_id: self.canvas.delete(self.ns_cars_text_id)
        self.ns_cars_text_id = self.canvas.create_text(200, 50, text=f"{state.num_cars_waiting_ns}\nCars", fill='white', font=("Arial", 12, "bold"))

        if self.ew_cars_text_id: self.canvas.delete(self.ew_cars_text_id)
        self.ew_cars_text_id = self.canvas.create_text(50, 200, text=f"{state.num_cars_waiting_ew}\nCars", fill='white', font=("Arial", 12, "bold"))
        
        self.root.update()

def run_gui_simulation(agent, steps=100, delay=1):
    root = tk.Tk()
    ui = TrafficLightUI(root)
    
    # Import here to avoid circular dependency if placed at top
    from states import TFState
    
    state = TFState('RED', 'GREEN', 5, 5, reward_type='balanced')
    
    def step_simulation(current_step):
        if current_step >= steps:
            print("Simulation finished.")
            return

        action = agent.getAction(state)
        prev_state = copy.deepcopy(state)
        state.updateState(action)
                
        ui.update(state)
        
        # Schedule next step
        root.after(int(delay * 1000), lambda: step_simulation(current_step + 1))

    # Start simulation loop
    root.after(100, lambda: step_simulation(0))
    root.mainloop()

if __name__ == "__main__":
    # For testing UI standalone. Not training
    from qlearning_agents import QLearningAgent
    agent = QLearningAgent(alpha=0.1, epsilon=0.1, gamma=0.8)
    run_gui_simulation(agent)
