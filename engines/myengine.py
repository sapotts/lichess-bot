import random
import chess
from chess.engine import PlayResult
import engine_wrapper
from engine_wrapper import MinimalEngine
import torch
from torch import nn
import numpy as np


class Engine(MinimalEngine):
    pass


class MainEngine(Engine):
    def search(self, board: chess.Board, time_limit: chess.engine.Limit, ponder: bool, draw_offered: bool,
               root_moves: engine_wrapper.MOVE) -> chess.engine.PlayResult:
        pass


def board_to_input(board: chess.Board) -> torch.Tensor:
    ret = torch.zeros((1, 385), dtype=torch.float32, device='cuda')
    ret[0][0] = 1 if board.turn else 0
    state = board.board_fen()
    rows = state.split("/")
    current: int = 0
    p_off = 1
    r_off = 65
    n_off = 129
    b_off = 193
    q_off = 257
    k_off = 321
    for i in range(8):
        for j in range(len(rows[i])):
            char = rows[i][j]
            if char.isdigit():
                current += int(char)
            else:
                color = 1
                if char.islower():
                    color = -1
                if char.lower() == "p":
                    ret[0][current + p_off] = color
                if char.lower() == "r":
                    ret[0][current + r_off] = color
                if char.lower() == "q":
                    ret[0][current + q_off] = color
                if char.lower() == "k":
                    ret[0][current + k_off] = color
                if char.lower() == "n":
                    ret[0][current + n_off] = color
                if char.lower() == "b":
                    ret[0][current + b_off] = color
                current += 1
    return ret


# chess.Move(from_square=chess.parse_square())
def pick_move(out: torch.Tensor, board: chess.Board) -> chess.Move:
    pass
    # TODO
    # output NN is 64 + 64 for starting move location to ending move location
    # iterate through legal moves and check what they are rated by NN
    # pick highest rated legal move



class NeuralNet(nn.Module):
    def __init__(self, expl_rate=0.0):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 200)
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(200, 200) for i in range(4)])
        self.lo = nn.Linear(200, 128)

    def forward(self, x):
        out = self.l1(x)
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out = self.lo(out)
        return out


def move_decision(exploration_rate, board, model):
    if np.random.rand() < exploration_rate:
        return PlayResult(random.choice(list(board.legal_moves)), None)
    else:
        inp = board_to_input(board)
        move = model(inp)
        for mv in board.legal_moves:
            print(mv, type(mv))
            print(mv.uci())
        #m = chess.Move()
        move_choice = pick_move(move, board)
        #return move_choice


if __name__ == "__main__":
    # inputs will be 64 * 6 for pieces on the board
    # output will be a chess move verified by chess.Board.legal_moves
    input_size = 385
    device = torch.device("cuda")
    output_size = 80
    lr = 0.001
    games = 1
    exploration_rate = 0
    exploration_rate_decay = 0.99999975
    model1 = NeuralNet()
    model2 = NeuralNet()
    model1.to(device=device)
    model2.to(device=device)
    assert torch.cuda.is_available()
    board = chess.Board()
    for epoch in range(games):
        board.reset_board()
        models = [model1, model2]
        counter = 0
        while not board.is_game_over(claim_draw=True):
            current_model = models[counter % 2]
            #inp = board_to_input(board)
            #print(inp)
            #output = current_model(inp)
            move = move_decision(exploration_rate, board, model1)
            print(move)
            res = board.push_uci(move)
            counter += 1
            print(move)
        print(board.outcome())
