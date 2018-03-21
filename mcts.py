from __future__ import print_function
import math
import copy
from state import State
import numpy as np
from time import clock


def rollout_policy_function(state):
    all_empty_pos = state.all_empty_pos()
    probs = np.random.rand(len(all_empty_pos))
    return zip(all_empty_pos, probs)
    # result = []
    # for i in range(len(probs)):
    #     result.append((all_empty_pos[i], probs[i]))
    # return result

def policy_function(state):
    all_empty_pos = state.all_empty_pos()
    probs = np.random.rand(len(all_empty_pos))
    return zip(all_empty_pos, probs), 0


class Node:
    '''
    defines node in monte carlo tree
    '''
    def __init__(self, parent, prior_prob):
        self.Q = 0  #
        self.P = prior_prob  # prior brobability
        self.u = 0  # visit-count-adjusted prior score u
        self.visit_count = 0
        self.parent = parent    # parent node, =None if the node is root
        self.children = {}  # key: move on board, value: Node

    def expand(self, action_and_prob_list):
        '''
        :param action_and_prob_list: ((row, col), prob)
        :return:
        '''
        for action, prob in action_and_prob_list:
            if action in self.children:
                continue
            self.children[action] = Node(self, prob)

    def select_child(self, c):
        '''

        :param c:
        :return: ((row, col), node)
        '''
        all_child = self.children.items()
        return max(all_child, key=lambda item: item[1].get_value(c))

    def update(self, sub_tree_value):
        self.visit_count += 1
        self.Q += float(sub_tree_value - self.Q) / float(self.visit_count)

    def update_recursive(self, value):
        if self.parent is not None:
            self.parent.update_recursive(-value)
        self.update(value)

    def get_value(self, c):
        '''
        calculate upper confidence bound (UCB)
        :param c: control degree of exploration
        :return:
        '''
        self.u = c * self.P * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.Q + self.u

    def is_leaf_node(self):
        return self.children == {}

    def is_root_node(self):
        return self.parent is None


class MCTS:
    def __init__(self, c, max_simulate_count, policy_function, rollout_policy_function):
        self.root = Node(parent=None, prior_prob=1.0)
        self.c = c
        self.max_simulate_count = max_simulate_count
        self.policy = policy_function
        self.rollout_policy = rollout_policy_function

    def simulate(self, state):
        node = self.root
        # selection
        while True:
            if node.is_leaf_node():
                break
            move, node = node.select_child(self.c)
            state.place_chess(move[0], move[1], state.get_next_player())

        action_probs, _ = self.policy(state)
        end_state = state.get_end_state()

        # expansion
        if end_state is None:   # game is not end
            node.expand(action_probs)
        # simulation
        leaf_value = self.evaluate_rollout(state, self.rollout_policy)
        # backpropagation
        node.update_recursive(-leaf_value)

    def evaluate_rollout(self, state, rollout_policy_function, max_rollout_count=1000):
        current_player = state.get_next_player()
        for i in range(max_rollout_count):
            end_state = state.get_end_state()
            if end_state is not None:
                break
            action_probs = rollout_policy_function(state)
            max_action = max(action_probs, key=lambda item: item[1])[0]
            state.place_chess(max_action[0], max_action[1], state.get_next_player())
        else:
            pass
        if end_state is None or end_state == State.END_STATE_DRAW:
            return 0
        else:
            if current_player == end_state:
                return 1
            else:
                return -1

    def get_move(self, state):
        start = clock()
        for _ in range(self.max_simulate_count):
            # print('sim', _)
            _state = copy.deepcopy(state)
            self.simulate(_state)
        res = max(self.root.children.items(), key=lambda item: item[1].visit_count)[0]
        end = clock()
        # print('get_move time: {}'.format(end - start))
        return res

    def move(self, move, ):
        if move in self.root.children.keys():
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = Node(parent=None, prior_prob=1.0)
