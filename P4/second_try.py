# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''
    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.eta = 0.001
        self.gamma = 0.8
        self.epsilon = 0.01
        self.alpha = 1
        self.W = npr.uniform(0, 0.5, 19)
        self.game_counter = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epsilon *= 0.99
        if self.game_counter % 5 == 0:
            print(self.W)
        self.game_counter += 1

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        def function(state, action):
            centered = np.mean([state["tree"]["top"] - state["monkey"]["top"], state["tree"]["bot"] - state["monkey"]["bot"]])
            temp_state = 1/100*np.array([state["monkey"]["vel"],
                                         state["monkey"]["vel"] ** 2,
                                         state["tree"]["dist"],
                                         state["tree"]["dist"] ** 2, 
                                         centered,
                                         centered ** 2,
                                         state["monkey"]["vel"] * state["tree"]["dist"],
                                         state["tree"]["dist"] * centered,
                                         state["monkey"]["vel"] * centered])
            return np.append(np.append([1], temp_state), np.zeros(temp_state.shape)) if action == 0 else np.append(np.append([1], np.zeros(temp_state.shape)), temp_state)

        if not self.last_state:
            self.last_state  = state
            self.last_action = 0 if npr.binomial(1, self.epsilon) == 0 else npr.binomial(1, 0.5)
            return self.last_action

        old = np.dot(self.W, function(self.last_state, self.last_action))
        new = np.max([np.dot(self.W, function(state, 0)), np.dot(self.W, function(state, 1))])
        self.W = (1-self.eta)*self.W + self.eta*(self.last_reward + self.gamma*new) - self.eta*self.alpha*np.linalg.norm(self.W) ** 2
        self.last_action = np.argmax([np.dot(self.W, function(state, 0)), np.dot(self.W, function(state, 1))]) if npr.binomial(1, self.epsilon) == 0 else npr.binomial(1, 0.5)
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 300, 1)

	# Save history.
	np.save('hist',np.array(hist))

	print('max: ' + str(max(hist)))

	print('average: ' + str(np.mean(np.array(hist))))
