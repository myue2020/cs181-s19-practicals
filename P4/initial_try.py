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

        self.eta = 0.05
        self.gamma = 0.7
        self.epsilon = 0.01
        self.gravity = 0
        self.qtable = {}


    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epsilon *= 0.995
        # print(self.gravity)
        self.gravity = None
        # print(self.qtable)

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # don't jump at the first frame, determine gravity
        if not self.gravity:
            self.gravity = -state["monkey"]["vel"]
            return 0

        # if state["score"] > 40:
        #     self.epsilon = 0
#self.gravity,
        new_state = (
                     # self.gravity,
					 state["monkey"]["vel"] // 10,
                     state["tree"]["dist"] // 120,
                     np.mean([state["tree"]["top"] - state["monkey"]["top"], state["tree"]["bot"] - state["monkey"]["bot"]]) // 60)

        def q(state):
            if state not in self.qtable:
                self.qtable[state] = np.zeros(2)
            return self.qtable[state]

        if not self.last_state:
            self.last_state  = new_state
            self.last_action = np.argmax(q(self.last_state)) if npr.binomial(1, self.epsilon) == 0 else npr.binomial(1, 0.5)
            return self.last_action

        # Q next line
        self.qtable[self.last_state][self.last_action] = (1 - self.eta)*q(self.last_state)[self.last_action] + self.eta * (self.last_reward + self.gamma*np.max(q(new_state)))
        # SARSA next line
        #self.qtable[self.last_state][self.last_action] = (1 - self.eta)*q(self.last_state)[self.last_action] + self.eta * (self.last_reward + self.gamma*q(new_state)[np.argmax(q(new_state)) if npr.binomial(1, self.epsilon) == 0 else npr.binomial(1, 0.5)])

        self.last_state = new_state
        self.last_action = np.argmax(q(self.last_state)) if npr.binomial(1, self.epsilon) == 0 else npr.binomial(1, 0.5)

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

	n = 200
	# Run games.
	run_games(agent, hist, n, 1)

	# Save history.
	np.save('./scores/q_report_bad',np.array(hist))

	print('max: ' + str(max(hist)))

	print('average: ' + str(np.mean(np.array(hist)[3*n//4:n])))
