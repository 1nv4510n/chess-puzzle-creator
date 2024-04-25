from geneticalgorithm import geneticalgorithm as ga
import numpy as np
import chess
import chess.engine

is_valid = False

ENGINE_PATH = "stockfish-windows-x86-64-sse41-popcnt.exe"

board = chess.Board("5Q2/5K1k/8/8/8/8/8/8 w - - 0 1")

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

info = engine.analyse(board, chess.engine.Limit(time=0.1))

def value_to_piece(value) -> chess.Piece:
    if value == 0:
        return None
    elif value <= 6:
        return chess.Piece(value, chess.WHITE)
    else:
        return chess.Piece(value - 6, chess.BLACK)

def array_to_chess_board(arr) -> chess.Board:
    board = chess.Board(fen='8/8/8/8/8/8/8/8 w - - 0 1')
    for i, value in enumerate(arr):
        piece = value_to_piece(value)
        if piece:
            board.set_piece_at(i, piece)
    return board

def f(X):
    board = array_to_chess_board(X.astype(int).tolist())

    # награда за малое количество фигур
    penalty = len(board.piece_map()) * 0.1

    # штраф за невалидную позицию
    if not board.is_valid():
        return 10 + penalty
    
    global is_valid
    is_valid = True
    # анализ очередной позиции движком
    info = engine.analyse(board, chess.engine.Limit(depth=10), multipv=2)

    # штраф за малое количество ходов
    if len(info) < 1:
        return 9 + penalty
    if len(info) < 2:
        return 8 + penalty

    # штраф за матовую позицию черных, либо ее отсутствие
    score = info[0]["score"].white()
    if not score.is_mate() or score.mate() <= 0:
        return 6 + penalty

    # штраф за далекую от мата позицию
    penalty += min(3, abs(score.mate() - 3)) / 3

    # штраф за второй лучший ход
    second_move_score = info[1]["score"].white().score(mate_score=1000)
    if second_move_score > 100:
        penalty += min(10.0, second_move_score / 100)
        
    return penalty

def main():
    print("Chess puzzle generator!")
    varbound = np.array([[0, 12]] * 64)
    algorithm_param = {'max_num_iteration': 5000,
                   'population_size': 20,
                   'mutation_probability': 0.05,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.9,
                   'parents_portion': 0.3,
                   'crossover_type': 'two_point',
                   'max_iteration_without_improv': 2000}
    model = ga(
        function=f, 
        dimension=64, 
        variable_type='int', 
        variable_boundaries=varbound, 
        algorithm_parameters=algorithm_param, 
        convergence_curve=False)
    
    while not is_valid:
        model.run()
    best_board = array_to_chess_board(list(model.best_variable))
    print(f"{'=' * 20}\nChess puzzle generated\n{'=' * 20}\n")
    print(best_board.fen())
    engine.close()

if __name__ == "__main__":
    main()