import random

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return (1 + (1/N) - 2*B)


class MoneyAgent(Agent):
    """ An agent with fixed initial wealth."""
    def __init__(self, unique_id):
        # Each agent should have a unique_id
        self.unique_id = unique_id 
        self.wealth = 1
    
    def step(self, m):
        """Give money to another agent."""
        if self.wealth > 0:
            # Pick a random agent
            other = random.choice(m.schedule.agents)
            # Give them 1 unit money
            other.wealth += 1
            self.wealth -= 1        

class MoneyModel(Model):
    """A model with some number of agents."""
    def __init__(self, N):
        self.running = True
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.create_agents()
        agent_reporters = {"Wealth": lambda a: a.wealth}
        model_reporters = {"Gini": compute_gini}
        self.dc = DataCollector(model_reporters=model_reporters,
                                agent_reporters=agent_reporters)

    def create_agents(self):
        """Method to create all the agents."""
        for i in range(self.num_agents):
            a = MoneyAgent(i)
            self.schedule.add(a)

    def step(self):
        self.schedule.step()
        self.dc.collect(self)