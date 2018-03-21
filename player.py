from __future__ import print_function
from abc import abstractmethod
from state import State
from game import GuiChessBoard, ChessBoard
import random
import threading
import mcts
import mcts_nn


class BasePlayer:
    def __init__(self, role):
        assert (role == State.BLACK or role == State.WHITE)
        self.role = role

    # return (row, col)
    @abstractmethod
    def get_next_step(self, state):
        pass


class HumanPlayer(BasePlayer):
    def __init__(self, role, chessboard):
        BasePlayer.__init__(self, role)
        self.chessboard = chessboard
        self.sem = threading.Semaphore(0)

    def on_place_chess(self, row, col):
        self.pos = (row, col)
        if not self.pos in self.chessboard.state.all_empty_pos():
            return
        self.sem.release()

    def get_next_step(self, state):
        # block and wait for human action
        self.chessboard.on_place_chess = self.on_place_chess
        self.sem.acquire()
        self.chessboard.on_place_chess = None
        return self.pos


class HumanPlayerCommand(BasePlayer):
    def __init__(self, role, chessboard):
        BasePlayer.__init__(self, role)
        self.chessboard = chessboard

    def get_next_step(self, state):
        # block and wait for human action
        s = input('next move:')
        row, col = (int(i) for i in s.split(' '))
        if (row, col) not in self.chessboard.state.all_empty_pos():
            print('invalid move')
        return row, col


class RandomPlayer(BasePlayer):
    def __init__(self, role):
        BasePlayer.__init__(self, role)

    # returns random move
    def get_next_step(self, state):
        return random.sample(state.all_empty_pos(), k=1)[0]


class PlayerMCTS(BasePlayer):
    def __init__(self, role, max_simulate_count=500):
        BasePlayer.__init__(self, role)
        self.max_simulate_count=500
        self.mcts = mcts.MCTS(c = 5, max_simulate_count=max_simulate_count, policy_function=mcts.policy_function, rollout_policy_function=mcts.rollout_policy_function)

    def get_next_step(self, state):
        all_empty_pos = state.all_empty_pos()
        if len(all_empty_pos) == 0:
            return None
        else:
            move = self.mcts.get_move(state)
            self.mcts.move(-1)
            return move

    def __str__(self):
        return 'mcts, max_sim = {}'.format(self.max_simulate_count)


class PlayerMCTS1(BasePlayer):
    def __init__(self, role, max_simulate_count=500):
        BasePlayer.__init__(self, role)
        self.max_simulate_count = max_simulate_count
        self.mcts = mcts_nn.MCTS(c = 5, max_simulate_count=max_simulate_count, policy_function=mcts_nn.MCTS.policy_function)

    def get_next_step(self, state):
        all_empty_pos = state.all_empty_pos()
        if len(all_empty_pos) == 0:
            return None
        else:
            move = self.mcts.get_move(state)
            self.mcts.move(-1)
            return move

    def get_next_step_and_action_prob(self, state, add_noise, temperature_param):
        if len(state.all_empty_pos()) == 0:
            return None
        else:
            move, action_probs = self.mcts.get_move(state, return_action_prob=True, add_noise=add_noise, temperature_param=temperature_param)
            self.mcts.move(-1)
            return move, action_probs

    def __str__(self):
        return 'mcts with nn, max_sim = {}'.format(self.max_simulate_count)
