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
    legal_moves: list[chess.Move] = list(board.legal_moves)
    ranking: dict[list[chess.Move]: int] = dict()
    for possible_move in legal_moves:
        starting_ind = possible_move.from_square
        ending_ind = possible_move.to_square
        engine_rating = out[0][(starting_ind * 64) + ending_ind - 63]
        ranking[possible_move] = engine_rating
    top_move = max(ranking, key=ranking.get)
    #print(chess.SQUARE_NAMES)
    return top_move


class NeuralNet(nn.Module):
    def __init__(self, expl_rate=0.0):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 200)
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(200, 200) for i in range(4)])
        self.lo = nn.Linear(200, 4096)

    def forward(self, x):
        out = self.l1(x)
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out = self.lo(out)
        return out


def move_decision(exploration_rate, board, model):
    if np.random.rand() < exploration_rate:
        return random.choice(list(board.legal_moves))
    else:
        inp = board_to_input(board)
        move = model(inp)
        move_choice = pick_move(move, board)
        return move_choice


if __name__ == "__main__":
    # inputs will be 64 * 6 for pieces on the board
    # output will be a chess move verified by chess.Board.legal_moves
    input_size = 385
    device = torch.device("cuda")
    output_size = 80
    lr = 0.0001
    games = 1
    exploration_rate = 1
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
        while not board.is_game_over(claim_draw=False):
            current_model = models[counter % 2]
            move = move_decision(exploration_rate, board, model1)
            exploration_rate *= exploration_rate_decay
            move = move.uci()
            res = board.push_uci(move)
            counter += 1
        print(board.outcome())
        #print(board.is_game_over())
        print(board.fen())


class MainEngine(Engine):
    def __init__(self):
        self.network = NeuralNet()

    def search(self, board: chess.Board, time_limit: chess.engine.Limit, ponder: bool, draw_offered: bool,
               root_moves: engine_wrapper.MOVE) -> chess.engine.PlayResult:
        move = move_decision(0, board, self.network)
        return PlayResult(move, None)
