from __future__ import print_function
import tkinter
import player
from state import State
import threading
import time
import cProfile
import game
import os
import platform

# python 3 only
assert(platform.python_version_tuple()[0] == '3')

gui_chess_board = None
players = [None, None]


def init_player(player_kind, index, role):
    global gui_chess_board
    global players
    if player_kind == 0:    # human
        players[index] = player.HumanPlayer(role=role, chessboard=gui_chess_board)
    elif player_kind == 1:  # mcts
        if State.BOARD_SIZE == 6:
            players[index] = player.PlayerMCTS(role=role, max_simulate_count=3000)
        elif State.BOARD_SIZE == 9:
            players[index] = player.PlayerMCTS(role=role, max_simulate_count=4000)
    elif player_kind == 2:  # mcts_nn
        if State.BOARD_SIZE == 6:
            p = player.PlayerMCTS1(role=role)
            p.mcts.net.load_network(os.path.join('models', 'policy_net-6'))
            players[index] = p
        elif State.BOARD_SIZE == 9:
            p = player.PlayerMCTS1(role=role)
            p.mcts.net.load_network(os.path.join('models', 'policy_net-9'))
            players[index] = p


def play_thread_gui():
    global gui_chess_board
    global players
    print('play thread')
    p1 = players[0]
    p2 = players[1]
    g = game.Game(gui_chess_board, p1, p2)
    g.play(show_steps=True)


def play_thread_command():
    print('play thread')
    chess_board = game.ChessBoard()

    # p1 = player.RandomPlayer(role=State.BLACK)
    # p2 = player.RandomPlayer(role=State.WHITE)

    p1 = player.PlayerMCTS(role=State.BLACK)
    p2 = player.RandomPlayer(role=State.WHITE)

    players = {
        State.BLACK: p1,
        State.WHITE: p2
    }
    current_player_role = State.BLACK
    while (not chess_board.state.get_end_state()):
        print(chess_board.state.seq_count, end=' ')
        step1 = players[current_player_role].get_next_step(chess_board.state)
        # gui_chess_board.state.place_chess(step1[0], step1[1], who=current_player_role)
        chess_board.place_chess(step1[0], step1[1], current_player_role)
        # output state
        # chess_board.update()
        if current_player_role == State.BLACK:
            current_player_role = State.WHITE
        else:
            current_player_role = State.BLACK
    winner = chess_board.state.get_end_state()
    print('state: ', chess_board.state)
    print('winner: {}'.format(winner))
    print('play thread end')
    return winner


def splash():
    def selector(group, index):
        def on_radiobutton_select():
            another_group = group ^ 1
            if index == 2:
                rbs[another_group][2]['state'] = tkinter.DISABLED
            else:
                rbs[another_group][2]['state'] = tkinter.NORMAL

        return on_radiobutton_select

    def quit():
        splash_window.destroy()
        print('quit')

    splash_window = tkinter.Tk()
    splash_window.title("选择游戏模式...")

    variables = []
    rbs = []
    for group in range(2):
        label_frame = tkinter.LabelFrame(splash_window, text="player {}".format(group + 1))
        btns = []
        player_kind = '人类玩家', '朴素mcts', '使用神经网络'
        variables.append(tkinter.IntVar())
        v1 = variables[group]
        v1.set(0)
        for i, p in enumerate(player_kind):
            rb = tkinter.Radiobutton(label_frame, variable=v1, text=p, value=i, command=selector(group, i))
            rb.pack()
            btns.append(rb)
        rbs.append(btns)
        label_frame.grid(row=0, column=group, padx=(20, 20), pady=(20, 20))

    v2 = tkinter.IntVar()
    v2.set(0)
    label_frame = tkinter.LabelFrame(splash_window, text="棋盘")
    tkinter.Radiobutton(label_frame, variable=v2, text="6*6 四子棋", value=0).pack()
    tkinter.Radiobutton(label_frame, variable=v2, text="9*9 五子棋", value=1).pack()
    label_frame.grid(row=0, column=2, padx=(20, 20), pady=(20, 20))
    btn_close = tkinter.Button(master=splash_window, text='确定', command=quit)
    btn_close.grid(row=1, column=1)

    splash_window.mainloop()

    print('player 1: {}\nplayer 2: {}'.format(player_kind[variables[0].get()], player_kind[variables[1].get()]))
    return variables[0].get(), variables[1].get(), v2.get()


def main():
    global gui_chess_board
    i1, i2, board = splash()
    if board == 0:
        State.BOARD_SIZE = 6
        State.MAX_CONTINOUS = 4
    elif board == 1:
        State.BOARD_SIZE = 9
        State.MAX_CONTINOUS = 5

    main_window = tkinter.Tk()
    gui_chess_board = game.GuiChessBoard(main_window)
    gui_chess_board.pack()
    gui_chess_board.update()

    init_player(i1, 0, State.BLACK)
    init_player(i2, 1, State.WHITE)

    t = threading.Thread(target=play_thread_gui)
    t.setDaemon(True)
    t.start()

    main_window.mainloop()
    print('mainloop end')


if __name__ == '__main__':
    main()