from agent import Agent

class MADDPGController():
    
    def __init__(self, state_size, action_size, seed, hparams):
        self.seed = random.seed(seed)
        self.state_size = state_size 
        self.action_size = action_size 
        self.hparams = hparams 
        
        self.act0 = np.zeros(self.action_size)
        self.act1 = np.zeros(self.action_size)
        self.agents = [Agent(state_size, action_size, seed, hparams, identity) for identity in range(0,2)]
        
        
    def act(self, states):
        
        act0 = self.agents[0].act(states[0])
        act1 = self.agents[1].act(states[1])
        
        
        return [act0, act1]
    
        
    def step(self, states, actions, rewards, next_states, dones, ep):
        for i, agent in enumerate(self.agents):
            agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i], ep)
        
                    
    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    def print_models(self):
        for agent in self.agents:
            agent.print_models()
            
    def save_models(self):
        for agent in self.agents:
            agent.save_models()