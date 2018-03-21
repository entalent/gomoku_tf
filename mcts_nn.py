from __future__ import print_function
import math
import copy
from time import clock
import numpy as np
from mcts import Node
from state import State
from policy_net import PolicyNetConv


class MCTS:
    def __init__(self, c, max_simulate_count, policy_function):
        self.root = Node(parent=None, prior_prob=1.0)
        self.c = c
        self.max_simulate_count = max_simulate_count
        self.policy = self.policy_function
        self.net = PolicyNetConv()

    def policy_function(self, state):
        '''

        :param state:
        :return: [action_probs, value]
        '''
        action_probs, value = self.net.session.run([self.net.action_prob, self.net.value], feed_dict={
            self.net.state_input: [state.get_nn_input()]
        })
        all_empty_pos = state.all_empty_pos()
        probs = []
        action_probs = np.reshape(action_probs, (State.BOARD_SIZE, State.BOARD_SIZE))
        for row in range(action_probs.shape[0]):
            for col in range(action_probs.shape[1]):
                if (row, col) in all_empty_pos:
                    probs.append(action_probs[row, col])
        return zip(all_empty_pos, probs), 0

    def simulate(self, state):
        node = self.root
        while True:
            if node.is_leaf_node():
                break
            move, node = node.select_child(self.c)
            state.place_chess(move[0], move[1], state.get_next_player())

        action_probs, leaf_value = self.policy(state)
        end_state = state.get_end_state()

        if end_state is None:
            node.expand(action_probs)
        else:
            if end_state == State.END_STATE_DRAW:
                leaf_value = 0.0
            elif end_state == state.get_next_player():  # ????
                leaf_value = 1.0
            else:
                leaf_value = -1.0
        node.update_recursive(-leaf_value)

    def get_action_prob(self, state, temperature_param):
        def _softmax(x):    # x -> softmax(x)
            # exp_x = np.exp(x - np.max(x))
            # return exp_x / exp_x.sum(axis=0)
            probs = np.exp(x - np.max(x))
            probs /= np.sum(probs)
            return probs

        for _ in range(self.max_simulate_count):
            _state = copy.deepcopy(state)
            self.simulate(_state)
        action_visitcount = [(action, node.visit_count) for action, node in self.root.children.items()]
        actions, visit_count = zip(*action_visitcount)
        probs = _softmax((1.0 / temperature_param) * np.log(np.array(visit_count) + 1e-9))
        return actions, probs

    def get_move(self, state, add_noise=True, return_action_prob=False, temperature_param=1e-3):
        start = clock()

        actions, probs = self.get_action_prob(state, temperature_param=temperature_param)

        if add_noise:
            move_index = np.random.choice(a=range(len(actions)), p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))) )
            move = actions[move_index]
        else:
            move_index = np.random.choice(a=range(len(actions)), p=probs)
            move = actions[move_index]

        end = clock()
        # print('get_move time: {}, {}'.format(end - start, t_sum))
        if not return_action_prob:
            return move
        else:
            action_probs = np.zeros(shape=(State.BOARD_SIZE, State.BOARD_SIZE), dtype=np.float32)
            for i in range(len(actions)):
               action_probs[actions[i]] = probs[i]
            return move, action_probs

    def move(self, move, ):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            # new_state = copy.deepcopy(self.root.state)
            # new_state.place_chess(move[0], move[1])
            self.root = Node(parent=None, prior_prob=1.0)

