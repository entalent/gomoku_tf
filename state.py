from __future__ import print_function
import numpy as np

# the state representation for gomoku （five-in-a-row or N-in-a-row)
class GomokuState:
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    BOARD_SIZE = 9
    MAX_CONTINOUS = 5

    END_STATE_BLACK = BLACK
    END_STATE_WHITE = WHITE
    END_STATE_DRAW = 0

    def __init__(self):
        self.data = np.zeros(shape=(GomokuState.BOARD_SIZE, GomokuState.BOARD_SIZE), dtype=int)
        self.chess_seq = np.zeros(shape=(GomokuState.BOARD_SIZE, GomokuState.BOARD_SIZE), dtype=int)
        self.seq_count = 1
        self._empty_pos = [(row, col) for row in range(self.data.shape[0]) for col in range(self.data.shape[1])]

    def get_next_player(self):
        return GomokuState.BLACK if self.seq_count % 2 == 1 else GomokuState.WHITE

    def check_is_valid_step(self, row, col, who):
        # assert (who == State.BLACK or who == State.WHITE)
        assert (who == self.get_next_player())
        assert (0 <= row < GomokuState.BOARD_SIZE and 0 <= col < GomokuState.BOARD_SIZE)
        assert (self.data[row][col] == GomokuState.EMPTY and self.chess_seq[row][col] == 0)

    def place_chess(self, row, col, who):
        self.check_is_valid_step(row, col, who)
        self.data[row][col] = who
        self.chess_seq[row][col] = self.seq_count
        self.seq_count += 1
        self.last_row, self.last_col = row, col
        self._empty_pos.remove((row, col))

    def _row(self, i):
        return self.data[i]

    def _col(self, i):
        return self.data[:, i]

    def _diag(self, i, j):
        return np.diag(self.data, j - i)

    def _diag_counter(self, i, j):
        return np.diag(np.fliplr(self.data), self.data.shape[1] - j - i - 1)

    def find_continous(self):
        if self.seq_count < GomokuState.MAX_CONTINOUS:
            return None

        def find_subseq(seq, target):
            count = 0
            for i in range(len(seq)):
                if seq[i] == target:
                    count += 1
                else:
                    count = 0
                if count >= GomokuState.MAX_CONTINOUS:
                    return True
            return False

        i = self.last_row
        j = self.last_col
        target = self.data[i][j]
        l = []
        l.append(self._row(i))
        l.append(self._col(j))
        l.append(self._diag(i, j))
        l.append(self._diag_counter(i, j))
        for seq in l:
            if find_subseq(seq, target):
                return target
        if self.seq_count == self.data.size:
            return GomokuState.END_STATE_DRAW
        return None

    def find_continous_(self):
        def __find_continous(seq):
            cont_pattern_black = np.array([GomokuState.BLACK] * GomokuState.MAX_CONTINOUS)
            cont_pattern_white = np.array([GomokuState.WHITE] * GomokuState.MAX_CONTINOUS)

            if len(seq) < GomokuState.MAX_CONTINOUS:
                return None
            for i in range(0, len(seq) - GomokuState.MAX_CONTINOUS + 1):
                sub_seq = seq[i: i + GomokuState.MAX_CONTINOUS]
                if (cont_pattern_black == sub_seq).all():
                    return GomokuState.BLACK
                elif (cont_pattern_white == sub_seq).all():
                    return GomokuState.WHITE
            return None

        # row
        for i in range(self.data.shape[0]):
            row = self.data[i]
            c = __find_continous(row)
            if c is not None:
                return c

        # col
        for j in range(self.data.shape[1]):
            col = self.data[:, j]
            c = __find_continous(col)
            if c is not None:
                return c

        # diag
        data_flip = np.fliplr(self.data)
        for k in range(-(self.data.shape[0] - 1), self.data.shape[1]):
            m_diag = np.diag(self.data, k=k)
            c_diag = np.diag(data_flip, k=k)
            c1 = __find_continous(m_diag)
            c2 = __find_continous(c_diag)
            if c1 is not None:
                return c1
            if c2 is not None:
                return c2
        return None

    # 黑棋赢：返回1
    # 白棋赢：返回2
    # 平局：返回0
    # 局面不是最终局面：返回None
    def get_end_state(self):
        c = self.find_continous()
        if c:
            return c    # has winner
        if self.seq_count == GomokuState.BOARD_SIZE * GomokuState.BOARD_SIZE:
            return GomokuState.END_STATE_DRAW
        return None

    def all_empty_pos(self):
        return self._empty_pos

    def get_nn_input(self):
        '''

        :return: state, shape = (height, width, channels)
        '''
        # nn_input = np.zeros(shape=(State.BOARD_SIZE, State.BOARD_SIZE, 2), dtype=np.float32)
        # for row in range(self.data.shape[0]):
        #     for col in range(self.data.shape[1]):
        #         if self.data[row][col] == State.BLACK:
        #             nn_input[row][col][0] = 1.0
        #         elif self.data[row][col] == State.WHITE:
        #             nn_input[row][col][1] = 1.0
        # return nn_input

        channel_black = np.zeros(shape=(GomokuState.BOARD_SIZE, GomokuState.BOARD_SIZE), dtype=np.float32)
        channel_white = np.zeros(shape=(GomokuState.BOARD_SIZE, GomokuState.BOARD_SIZE), dtype=np.float32)
        channel_last_chess =  np.zeros(shape=(GomokuState.BOARD_SIZE, GomokuState.BOARD_SIZE), dtype=np.float32)
        channel_offensive_indicator = np.zeros(shape=(GomokuState.BOARD_SIZE, GomokuState.BOARD_SIZE), dtype=np.float32)

        # nn_input = np.zeros(shape=(State.BOARD_SIZE, State.BOARD_SIZE, 4), dtype=np.float32)
        nn_input = np.zeros(shape=(GomokuState.BOARD_SIZE, GomokuState.BOARD_SIZE, 3), dtype=np.float32)
        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                if self.data[row][col] == GomokuState.BLACK:
                    channel_black[row][col] = 1.0
                elif self.data[row][col] == GomokuState.WHITE:
                    channel_white[row][col] = 1.0
        if self.seq_count > 1:
            channel_last_chess[self.last_row][self.last_col] = 1.0
        if self.seq_count % 2 == 0:
            channel_offensive_indicator = 1.0 #最后一步棋是不是先手
        nn_input[:, :, 0] = channel_black
        nn_input[:, :, 1] = channel_white
        nn_input[:, :, 2] = channel_last_chess
        # nn_input[:, :, 3] = channel_offensive_indicator
        return nn_input

    def __str__(self):
        s = 'state:\n{}\nseq:\n{}'.format(str(self.data), str(self.chess_seq))
        return s

# the state representation for reversi
class ReversiState:
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    BOARD_SIZE = 6

    END_STATE_BLACK = BLACK
    END_STATE_WHITE = WHITE
    END_STATE_DRAW = 0

    def __init__(self):
        self.data = np.zeros(shape=(ReversiState.BOARD_SIZE, ReversiState.BOARD_SIZE), dtype=int)
        self.chess_seq = np.zeros(shape=(ReversiState.BOARD_SIZE, ReversiState.BOARD_SIZE), dtype=int)
        self.seq_count = 1
        self.black_count = 0
        self.white_count = 0
        self.data[2][2] = self.data[3][3] = ReversiState.WHITE
        self.data[3][2] = self.data[2][3] = ReversiState.BLACK
        self.seq_count = 5
        self.init_seq_count = 5 #一开始有4个棋子
        self._update_valid_pos(self.get_next_player())

    def get_next_player(self):
        return ReversiState.BLACK if self.seq_count % 2 == 1 else ReversiState.WHITE

    def check_is_valid_step(self, row, col, who):
        # assert (who == State.BLACK or who == State.WHITE)
        assert (who == self.get_next_player())
        assert (0 <= row < ReversiState.BOARD_SIZE and 0 <= col < ReversiState.BOARD_SIZE)
        assert (self.data[row][col] == ReversiState.EMPTY and self.chess_seq[row][col] == 0)

    def _update_state(self, row, col, who):
        self._check_single_pos(row, col, who, update=True)
        self.data[row][col] = who
        # unique, counts = np.unique(self.data, return_counts=True)
        self.black_count = (self.data == ReversiState.BLACK).sum()
        self.white_count = (self.data == ReversiState.WHITE).sum()

    # (row, col)位置是否可以放who颜色的棋子
    def _check_single_pos(self, row, col, who, update=False):
        if self.data[row][col] != State.EMPTY:
            return False
        rev_state = ReversiState.WHITE if who == ReversiState.BLACK else ReversiState.BLACK
        d = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        def _check_neighbor(i):
            rev_list = []
            dx, dy = d[i]
            _r, _c = row + dx, col + dy
            while (0 <= _r < self.data.shape[0]) and (0 <= _c < self.data.shape[1]):
                if self.data[_r][_c] == rev_state:
                    rev_list.append((_r, _c))
                elif self.data[_r][_c] == State.EMPTY:
                    return None
                elif self.data[_r][_c] == who:
                    return rev_list
                _r += dx
                _c += dy
            return None

        flag = False
        if update:
            for i in range(len(d)):
                rev_list = _check_neighbor(i)
                if rev_list is not None and len(rev_list) > 0:
                    flag = True
                    for (row, col) in rev_list:
                        self.data[row][col] = who
            return flag
        else:
            for i in range(len(d)):
                rev_list = _check_neighbor(i)
                if rev_list is not None and len(rev_list) > 0:
                    return True
            return False


    # 更新所有可以下棋的位置
    def _update_valid_pos(self, who):
        self._empty_pos = []
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if self._check_single_pos(i, j, who):
                    self._empty_pos.append((i, j))

    def place_chess(self, row, col, who):
        try:
            self.check_is_valid_step(row, col, who)
        except AssertionError:
            return
        if not (row, col) in self._empty_pos:
            return
        # self.data[row][col] = who
        self.chess_seq[row][col] = self.seq_count
        self.last_row, self.last_col = row, col
        self.seq_count += 1
        self.last_row, self.last_col = row, col
        self._update_state(row, col, who)
        self._update_valid_pos(self.get_next_player())

    def all_empty_pos(self):
        return self._empty_pos

    def get_nn_input(self):
        channel_black = np.zeros(shape=(ReversiState.BOARD_SIZE, ReversiState.BOARD_SIZE), dtype=np.float32)
        channel_white = np.zeros(shape=(ReversiState.BOARD_SIZE, ReversiState.BOARD_SIZE), dtype=np.float32)
        channel_last_chess = np.zeros(shape=(ReversiState.BOARD_SIZE, ReversiState.BOARD_SIZE), dtype=np.float32)
        channel_offensive_indicator = np.zeros(shape=(ReversiState.BOARD_SIZE, ReversiState.BOARD_SIZE),
                                               dtype=np.float32)

        # nn_input = np.zeros(shape=(State.BOARD_SIZE, State.BOARD_SIZE, 4), dtype=np.float32)
        nn_input = np.zeros(shape=(ReversiState.BOARD_SIZE, ReversiState.BOARD_SIZE, 3), dtype=np.float32)
        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1]):
                if self.data[row][col] == ReversiState.BLACK:
                    channel_black[row][col] = 1.0
                elif self.data[row][col] == ReversiState.WHITE:
                    channel_white[row][col] = 1.0
        if self.seq_count > self.init_seq_count:
            channel_last_chess[self.last_row][self.last_col] = 1.0
        if self.seq_count % 2 == 0:
            channel_offensive_indicator = 1.0  # 最后一步棋是不是先手
        nn_input[:, :, 0] = channel_black
        nn_input[:, :, 1] = channel_white
        nn_input[:, :, 2] = channel_last_chess
        # nn_input[:, :, 3] = channel_offensive_indicator
        return nn_input

    # 黑棋赢：返回1
    # 白棋赢：返回2
    # 平局：返回0
    # 局面不是最终局面：返回None
    def get_end_state(self):
        if not len(self._empty_pos) == 0:
            return None
        if self.black_count == self.white_count:
            return ReversiState.END_STATE_DRAW
        elif self.black_count > self.white_count:
            return ReversiState.END_STATE_BLACK
        else:
            return ReversiState.END_STATE_WHITE




State = GomokuState