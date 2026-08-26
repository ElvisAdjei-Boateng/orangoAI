import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OrangoEnv(gym.Env):
    def __init__(self):
        super().__init__()

        #continuous variable states( 0 to 1)

        self.observation_space= spaces.Box(
          low = 0.0,
          high=1.0,
          shape =(7,),
          dtype=np.float32

        )

        #6 advice strategies
        #Ask, inform, plan, small action, Reflect, Summarize
        self.action_space = spaces. Discrete(6)


        self.max_steps=10
        self.current_step =0
        self.state =None
        # self.user = None

    def reset(self, seed=None, options=None):
        super().reset(seed= seed)
        self.current_step=0

        self.state=self.np_random.uniform(
          low=0.0,
          high=1.0,
          size=7
        ).astype(np.float32)

        return self.state , {}

    def step(self, action):
        self.current_step +=1
        old_state = self.state.copy()

        #Temporary transition logic
        #We'll replace this with our user simulator
        self.applyAction(action)

        reward = self.calculateReward(
          old_state,
          self.state,
          action

        )
        terminated = False
        truncated = self.current_step >=self.max_steps

        info={"step":self.current_step, "action":action}

        return (
            self.state, 
            reward, 
            terminated, 
            truncated, 
            info
            )

    def applyAction(self,action):
        #Placeholder for transition logic
        #Until we build the user simulator

        if action ==0:  #Aask
          self.state[1] +=0.05

        elif action ==1: #Inform
          self.state[1]+=0.03

        elif action == 2 : #Plan
          self.state[2]+=0.08
          self.state[6]+=0.05

        elif action == 3: #Small action
          self.state[6]+=0.07
          self.state[5]+=0.03
        
        elif action == 4: #Reflect
          self.state[3]-=0.03
          self.state[2]+=0.04

        elif action == 5: #Summarize
          self.state[1]+=0.04
        
        self.state = np.clip(
          self.state,
          0.0,
          1.0
        )

    def calculateReward(self, old_state, new_state, action):
        goal_progress_change = (new_state[6]-old_state[6])
        clarity_change = (new_state[1]-old_state[1])
        stress_change = (old_state[3]- new_state[3])

        reward = (0.5*goal_progress_change + 0.3*clarity_change + 0.2*stress_change)
       
        return float(reward)





