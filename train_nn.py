from __future__ import print_function
import numpy as np
import mcts_nn
from state import State
from game import Game, ChessBoard, GuiChessBoard
from player import PlayerMCTS, PlayerMCTS1
from collections import deque
from util import Log
import random

player = PlayerMCTS1(role=State.BLACK)
self_play_data = deque(maxlen=10000)
train_batch_size = 500
evaluate_interval = 20


# 数据增广
def data_augment(self_play_data):
    aug_data = []
    for d in self_play_data:
        # print(d[0].shape, d[1].shape, d[2].shape)
        state = d[0]
        prob = d[1]
        v = d[2]

        state_f = np.zeros(dtype=np.float32, shape=state.shape)
        for i in range(state.shape[2]):
            state_f[:, :, i] = np.fliplr(state[:, :, i])
        prob_f = np.fliplr(prob)

        for i in range(1, 5):
            state1 = np.rot90(state, i, axes=(0, 1))
            prob1 = np.rot90(prob, i)
            aug_data.append((state1, prob1, v))

            state2 = np.rot90(state_f, i, axes=(0, 1))
            prob2 = np.rot90(prob_f, i)
            aug_data.append((state2, prob2, v))

    return aug_data


def self_play():
    global self_play_data
    global player
    player1 = player
    player2 = player

    game = Game(chessboard=ChessBoard(), player_defensive=player1, player_offensive=player2)
    winnner, play_data = game.self_play(show_steps=False)
    play_data = data_augment(play_data)
    self_play_data.extend(play_data)


def evaluate(eval_plays=10, max_simulate_count=500):
    global player
    Log.log('max_sim: {}'.format(max_simulate_count))
    win_count = 0
    for _ in range(eval_plays):
        chessboard = ChessBoard()
        player1 = player  # 1: +nn
        player2 = PlayerMCTS(role=State.WHITE, max_simulate_count=max_simulate_count)
        game = Game(chessboard=chessboard, player_offensive=player1, player_defensive=player2)
        winner = game.play(show_steps=False)
        Log.log('state: {}'.format(game.chessboard.state), print_to_stdout=False)
        if winner == State.END_STATE_BLACK:
            win_count += 1
    win_ratio = float(win_count) / eval_plays
    Log.log('win_ratio: {}'.format(win_ratio))
    return float(win_count) / eval_plays    # 胜率


max_sim = 500
best_win_ratio = 0.0
train_step = 1
lr_multiplier = 1.0


def train(global_step):
    kl_targ = 0.025
    def _kl(old_probs, new_probs):
        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
        return kl

    global train_batch_size
    global self_play_data
    global player
    global max_sim
    global best_win_ratio
    global train_step
    global lr_multiplier

    self_play()

    if len(self_play_data) >= train_batch_size:
        train_data = random.sample(self_play_data, k=train_batch_size)
        state_batch = [d[0] for d in train_data]
        action_prob_batch = [d[1].reshape([d[1].size]) for d in train_data]
        winner_batch = np.array([d[2] for d in train_data])
        winner_batch = winner_batch.reshape((winner_batch.size, 1))
        nn = player.mcts.net
        old_probs, old_v = nn.session.run([nn.action_prob, nn.value], feed_dict={
            nn.state_input: state_batch
        })
        for _ in range(5):
            __, loss, lr = nn.session.run([nn.train_op, nn.loss, nn.learning_rate], feed_dict={
                nn.state_input: state_batch,
                nn.action_prob_input: action_prob_batch,
                nn.value_input: winner_batch,
                nn.train_step: train_step + 1,
                # nn.learning_rate: 1e-5
                nn.learning_rate: lr_multiplier * 0.005
            })
            train_step += 1

            new_probs, new_v = nn.session.run([nn.action_prob, nn.value], feed_dict={
                nn.state_input: state_batch
            })
            kl = _kl(old_probs, new_probs)

            print('lr_mul={:.5f} lr={:.5f} kl={:.5f} loss={:.5f}'.format(lr_multiplier, float(lr), kl, loss))
            Log.log(loss, end=', ', print_to_stdout=False)
            if kl > kl_targ * 4:
                break
        Log.log('lr={:.7f}'.format(float(lr)), print_to_stdout=False)
        # adjust learning rate
        if kl > kl_targ * 2 and lr_multiplier > 0.1:
            lr_multiplier /= 1.5
        elif kl < kl_targ / 2 and lr_multiplier < 10:
            lr_multiplier *= 1.5

        Log.log('')
        if global_step % evaluate_interval == 0:
            win_ratio = evaluate(max_simulate_count=max_sim)
            print('best win_ratio: {}'.format(best_win_ratio))
            if win_ratio > best_win_ratio:
                best_win_ratio = win_ratio
                Log.log('new network')
                player.mcts.net.save_network(global_step)
                if win_ratio >= 0.9 and max_sim < 4000:
                    max_sim += 100
                    best_win_ratio = 0.0
            Log.flush()
    else:
        Log.log(len(self_play_data))


if __name__ == '__main__':
    for i in range(2500):
        train(i)