# from dataclasses import dataclass

# import numpy as np

# @dataclass
# class UserProfile:
#     """Hidden characteristics of a simulated user
#     The RL agent does NOT directly observe these values 
# """
# receptiveness:float
# follow_through:float
# learning_rate:float
# stress_sensitivity:float

# class UserSimulator:
# """Simulated how user react to different advice strategies"""

# def __init__(self, profile:UserProfile):

# def respond(self,state,action, rng):
# """"Given the current observable state and the chosen action, produce the user's next state"""

#     next_state = state.copy()
#     (
#       severity,clarity, goal_clarity, stress, motivation, engagement, goal_progress
#     ) = next_state

#     #Randomness represents the fact that users dont react identically every time

#     noise = rng.normal(0, 0.015)

#     if action==0: #Action 0
#       clarity_change =(0.08*self.profile.learning_rate*self.profile.receptiveness)
#       next_state[1] += clarity_change

#       #Asking good questions can slightly increase engagement

#       next_state[5] +=(0.03*self.profile.receptiveness)

#     elif action==1:
#       clarity_change=(0.06* self.profile.learning_rate*self.profile.receptiveness)

#       next_state[1] += clarity_change
#       #Information can reduce uncertainty but isnt necessarily enough to create action

#       next_state[3]-=(0.02*self.profile.learning_rate)

      
#     elif action == 2:

#       goal_change = (
#                  0.10
#                 * self.profile.follow_through
#                 * self.profile.receptiveness
#             )

#       next_state[6] += goal_change
#       next_state[2] += (
#                 0.06 * self.profile.receptiveness
#             )

#       next_state[5] += (
#                 0.03 * self.profile.receptiveness
#             )


#     elif action == 3:

#       goal_change = (
#                 0.08
#                 * self.profile.follow_through
#             )

#       next_state[6] += goal_change

#       next_state[5] += (
#                 0.05 * self.profile.receptiveness
#             )

#       next_state[3] -= (
#                 0.03 * self.profile.follow_through
#             )


#     elif action == 4:

#             next_state[1] += (
#                 0.05 * self.profile.learning_rate
#             )

#             next_state[3] -= (
#                 0.05
#                 * self.profile.stress_sensitivity
#             )

#             next_state[2] += (
#                 0.04 * self.profile.receptiveness
#             )

#     elif action == 5:

#             next_state[1] += (
#                 0.04 * self.profile.learning_rate
#             )

#             next_state[5] += (
#                 0.02 * self.profile.receptiveness
#             )

#         # -----------------------------------------
#         # Add stochastic variation
#         # -----------------------------------------

#             next_state += noise

#         # Keep every state variable inside [0, 1]
#             next_state = np.clip(
#             next_state,
#             0.0,
#             1.0
#             )

#             return next_state

      
# #Create user profile

# def create_user_profile(user_type, rng):

#     if user_type == "motivated":

#         return UserProfile(
#             receptiveness=0.85,
#             follow_through=0.85,
#             learning_rate=0.80,
#             stress_sensitivity=0.60,
#         )

#     if user_type == "neutral":

#         return UserProfile(
#             receptiveness=0.55,
#             follow_through=0.50,
#             learning_rate=0.50,
#             stress_sensitivity=0.50,
#         )

#     if user_type == "resistant":

#         return UserProfile(
#             receptiveness=0.30,
#             follow_through=0.25,
#             learning_rate=0.30,
#             stress_sensitivity=0.30,
#         )

#     raise ValueError(
#         f"Unknown user type: {user_type}"
#     )
