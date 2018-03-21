from __future__ import print_function
import tkinter
import numpy as np
from state import State
from util import Log
from time import clock

grid_size_px = 30
chess_radius_px = 10
chessboard_size_x = State.BOARD_SIZE
chessboard_size_y = State.BOARD_SIZE

offset_x_px = 10
offset_y_px = 10

chess_color_offensive = 'black' #先手
chess_color_defensive = 'white' #后手
text_color_offensive = '#00FFFF'
text_color_defensive = 'red'


def get_canvas_size():
    global offset_x_px
    global offset_y_px
    global grid_size_px
    width = offset_y_px + grid_size_px * State.BOARD_SIZE + offset_y_px
    height = offset_x_px + grid_size_px * State.BOARD_SIZE + offset_x_px
    return (width, height)


class ChessBoardCanvas(tkinter.Canvas):
    def __init__(self, master=None, height=0, width=0):
        tkinter.Canvas.__init__(self, master, height=height, width=width)
        self.count = 0
        self.draw(None)

    # row, col starts from 0
    def draw_chess(self, row, col, role, seq=None):
        if role == State.EMPTY:
            return
        x = col * grid_size_px + (grid_size_px / 2) + offset_x_px
        y = row * grid_size_px + (grid_size_px / 2) + offset_y_px
        color = chess_color_offensive if role == State.BLACK else chess_color_defensive
        text_color = text_color_offensive if role == State.BLACK else text_color_defensive
        self.create_oval(x - chess_radius_px, y - chess_radius_px, x + chess_radius_px, y + chess_radius_px, fill=color)
        if seq is not None:
            self.create_text(x, y, text=str(seq), fill=text_color)

    def draw_valid_grid(self, row, col):
        x = col * grid_size_px + (grid_size_px / 2) + offset_x_px
        y = row * grid_size_px + (grid_size_px / 2) + offset_y_px
        self.create_oval(x - 10, y - 10, x + 10, y + 10, fill='green')

    def draw(self, state):
        global chessboard_size_x
        global chessboard_size_y
        chessboard_size_x = State.BOARD_SIZE
        chessboard_size_y = State.BOARD_SIZE
        self.delete('all')
        # horizontal line
        for i in range(chessboard_size_y + 1):
            tp = [0, i * grid_size_px, chessboard_size_x * grid_size_px, i * grid_size_px]
            tp[0] += offset_x_px
            tp[1] += offset_y_px
            tp[2] += offset_x_px
            tp[3] += offset_y_px
            self.create_line(tp)
        # vertical line
        for i in range(chessboard_size_x + 1):
            tp = [i * grid_size_px, 0, i * grid_size_px, chessboard_size_y * grid_size_px]
            tp[0] += offset_x_px
            tp[1] += offset_y_px
            tp[2] += offset_x_px
            tp[3] += offset_y_px
            self.create_line(tp)

        if state is not None:
            for row in range(state.data.shape[0]):
                for col in range(state.data.shape[1]):
                    s = state.data[row][col]
                    seq = state.chess_seq[row][col]
                    self.draw_chess(row, col, role=s, seq=seq)
            for (row, col) in state.all_empty_pos():
                # self.draw_valid_grid(row, col)
                pass


class GuiChessBoard(tkinter.Frame):
    def __init__(self, master=None):
        # init gui
        tkinter.Frame.__init__(self, master)
        self.init_ui()
        # init state
        self.state = State()
        self.canvas.draw(self.state)

    def init_ui(self):
        self.label_frame = tkinter.LabelFrame(self, text="棋盘")
        height, width = get_canvas_size()
        self.canvas = ChessBoardCanvas(master=self.label_frame, height=height, width=width)
        self.canvas.bind('<Button-1>', self.on_click)
        self.label_frame.pack()
        self.canvas.pack()

        self.btn_close = tkinter.Button(master=self, text='关闭', command=self.exit)
        self.btn_close.pack()

    def exit(self):
        print('exit game...')
        exit(0)

    def on_click(self, event):
        board_x = event.x - offset_x_px
        board_y = event.y - offset_y_px
        row = int(board_y / grid_size_px)
        col = int(board_x / grid_size_px)
        # print('on click {} {}'.format(row, col))
        if self.on_place_chess is not None:
            self.on_place_chess(row, col)

    def on_place_chess(self, row, col):
        pass

    def place_chess(self, row, col, role):
        if not (row, col) in self.state.all_empty_pos():
            return
        self.state.place_chess(row, col, role)
        self.canvas.draw(self.state)

    def show(self, state, seq):
        s = State()
        s.data = np.array(state)
        s.chess_seq = np.array(seq)
        self.canvas.draw(s)


class ChessBoard:
    def __init__(self):
        self.state = State()

    def place_chess(self, row, col, role):
        self.state.place_chess(row, col, role)

    def update(self):
        print(self.state)


class Game:
    def __init__(self, chessboard, player_offensive, player_defensive):
        '''
        :param chessboard:
        :param player_offensive: 先手
        :param player_defensive: 后手
        '''
        self.players = {
            State.BLACK: player_offensive,
            State.WHITE: player_defensive
        }
        self.chessboard = chessboard

    def play(self, show_steps=True):
        current_player = State.BLACK
        start = clock()
        while self.chessboard.state.get_end_state() is None:
            move = self.players[current_player].get_next_step(self.chessboard.state)
            self.chessboard.place_chess(move[0], move[1], current_player)
            if show_steps:
                self.chessboard.update()
            current_player = State.BLACK if current_player == State.WHITE else State.WHITE
        winner = self.chessboard.state.get_end_state()
        end = clock()
        Log.log('winner: {}, used {:.3f}s'.format(winner, (end - start)))
        return winner

    def self_play(self, show_steps=True):
        states = []
        mcts_probs = []
        current_players = []
        current_player = State.BLACK
        start = clock()
        while self.chessboard.state.get_end_state() is None:
            # temperature parameter set to 1.0 when self-play
            move, action_probs = self.players[current_player].get_next_step_and_action_prob(self.chessboard.state, add_noise=True, temperature_param=1.0)
            # action_probs = action_probs.reshape([action_probs.size])
            # Log.log('move: {}'.format(move), end='\n', print_to_stdout=True)
            Log.log('move: {}'.format(move), end=' ', print_to_stdout=False)

            states.append(self.chessboard.state.get_nn_input())
            mcts_probs.append(action_probs)
            current_player = self.chessboard.state.get_next_player()
            current_players.append(current_player)

            self.chessboard.place_chess(move[0], move[1], current_player)

            if show_steps:
                self.chessboard.update()
            current_player = State.BLACK if current_player == State.WHITE else State.WHITE
        winner = self.chessboard.state.get_end_state()
        end = clock()
        Log.log('winner: {}, used {:.3f}s'.format(winner, (end - start)))
        winner_label = np.zeros(len(current_players))
        if winner != State.END_STATE_DRAW:
            winner_label[np.array(current_players) == winner] = 1.0
            winner_label[np.array(current_players) != winner] = -1.0
        for p in self.players.values():
            p.mcts.move(-1)
        return winner, zip(states, mcts_probs, winner_label)


if __name__ == '__main__':
    window = tkinter.Tk()
    gui_chess_board = GuiChessBoard(window)
    gui_chess_board.pack()
    window.mainloop()