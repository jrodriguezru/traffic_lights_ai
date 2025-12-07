import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class TrafficLightExtractor:
    """
    Feature extractor for Approximate Q-Learning in Traffic Lights.
    """
    def getFeatures(self, state, action):
        features = util.Counter()
        features['bias'] = 1.0
        features['num_cars_ns'] = state.num_cars_waiting_ns
        features['num_cars_ew'] = state.num_cars_waiting_ew
        features['light_ns'] = 1.0 if state.light_color_ns == 'GREEN' else 0.0
        
        if action == 'SWITCH':
             features['action_switch'] = 1.0
        else:
             features['action_stay'] = 1.0
             
        # Interaction features
        features['ns_green_and_cars'] = features['light_ns'] * features['num_cars_ns']
        
        return features