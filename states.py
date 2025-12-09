import math
import random

class TFState:
    """
    A TFState represents the state of a traffic light at an intersection.
    It includes information about the current light color and the number of cars waiting.
    """

    def __init__(self, light_color_ns, light_color_ew, num_cars_waiting_ns, num_cars_waiting_ew, reward_type='initial'):
        self.light_color_ns = light_color_ns  # e.g., 'RED', 'GREEN', 'YELLOW'
        self.light_color_ew = light_color_ew  # e.g., 'RED', 'GREEN', 'YELLOW'
        self.num_cars_waiting_ns = num_cars_waiting_ns  # integer count of cars waiting going north-south or south-north
        self.num_cars_waiting_ew = num_cars_waiting_ew  # integer count of cars waiting going east-west or west-east
        self.tick = 0
        self.reward_type = reward_type
        self.ticks_since_last_switch = 0
        self.last_action_penalty = 0
        # print(f"Initialized TFState: {self}")

    def __eq__(self, other):
        return (self.light_color_ns == other.light_color_ns and
                self.light_color_ew == other.light_color_ew and
                self.num_cars_waiting_ns == other.num_cars_waiting_ns and
                self.num_cars_waiting_ew == other.num_cars_waiting_ew and
                self.tick == other.tick and
                self.reward_type == other.reward_type and
                self.ticks_since_last_switch == other.ticks_since_last_switch)

    def __hash__(self):
        return hash((self.light_color_ns, self.light_color_ew, self.num_cars_waiting_ns, self.num_cars_waiting_ew, self.tick, self.reward_type, self.ticks_since_last_switch))
    def __str__(self):
        return f"TFState(light_color_ns={self.light_color_ns}, light_color_ew={self.light_color_ew}, num_cars_waiting_ns={self.num_cars_waiting_ns}, num_cars_waiting_ew={self.num_cars_waiting_ew}, tick={self.tick}, reward_type={self.reward_type}, last_switch={self.ticks_since_last_switch})"
    
    def getLegalActions(self):
        """
        Returns the legal actions for this state.
        Possible actions could be 'SWITCH', or 'STAY'.
        """
        return ['SWITCH', 'STAY']
    
    def updateState(self, action):
        """
        Updates the state based on the action taken.
        """
        self.tick += 1

        # Logic for switching penalty
        penalty_threshold = 5
        penalty_amount = 50
        
        if action == 'SWITCH':
            if self.ticks_since_last_switch < penalty_threshold:
                self.last_action_penalty = penalty_amount
            else:
                self.last_action_penalty = 0
            self.ticks_since_last_switch = 0

            if self.light_color_ns == 'GREEN':
                self.light_color_ns = 'RED'
                self.light_color_ew = 'GREEN'
            else:
                self.light_color_ns = 'GREEN'
                self.light_color_ew = 'RED'
        else:
            self.ticks_since_last_switch += 1
            self.last_action_penalty = 0
        
        # Sine wave arrival logic
        period = 50  # Adjust as needed
        amplitude = 2
        base = 3
        epsilon = 1.0 # Noise magnitude

        # Calculate base arrival rate with sine wave
        arrival_rate = base + amplitude * math.sin(2 * math.pi * self.tick / period)
        
        # Add noise
        noise = random.uniform(-epsilon, epsilon)
        
        # Total cars to add (ensure non-negative)
        new_cars = int(max(0, round(arrival_rate + noise)))
        
        # Distribute new cars (e.g., evenly)
        self.num_cars_waiting_ns += new_cars // 2
        self.num_cars_waiting_ew += new_cars - (new_cars // 2)

        # Handle departures (cars leaving if light is green)
        if self.light_color_ns == 'GREEN' and self.num_cars_waiting_ns > 0:
            self.num_cars_waiting_ns = max(0, self.num_cars_waiting_ns - 3)
        if self.light_color_ew == 'GREEN' and self.num_cars_waiting_ew > 0:
            self.num_cars_waiting_ew = max(0, self.num_cars_waiting_ew - 3)

    def getReward(self):
        """
        Returns the reward for the current state.
        """
        if self.reward_type == 'initial':
            return self.getRewardInitial()
        elif self.reward_type == 'squared':
            return self.getRewardSquared()
        elif self.reward_type == 'balanced':
            return self.getRewardBalanced()
        elif self.reward_type == 'penalty':
            return self.getRewardSwitchingPenalty()
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def getRewardBalanced(self):
        """
        Penalize imbalance between directions.
        """
        total_cars = self.num_cars_waiting_ns + self.num_cars_waiting_ew
        imbalance = abs(self.num_cars_waiting_ns - self.num_cars_waiting_ew)
        # Penalize total cars AND the difference (weighted by 2)
        return - (total_cars + 2 * imbalance)

    def getRewardSwitchingPenalty(self):
        """
        Penalize switching too fast.
        """
        total_cars = self.num_cars_waiting_ns + self.num_cars_waiting_ew
        return - (total_cars + self.last_action_penalty)

    def getRewardSquared(self):
        """
        Returns the reward for the current state.
        A more complex reward could be negative of squared total cars waiting.
        """
        return - (self.num_cars_waiting_ns ** 2 + self.num_cars_waiting_ew ** 2)

    def getRewardInitial(self):
        """
        Returns the reward for the current state.
        A simple reward could be negative of total cars waiting.
        """
        return - (self.num_cars_waiting_ns + self.num_cars_waiting_ew)