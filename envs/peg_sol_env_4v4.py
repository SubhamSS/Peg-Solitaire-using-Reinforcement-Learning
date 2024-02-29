import torch
import copy
import random
import numpy as np

class Peg_Board_44:

    def __init__(self):
        g=7
    def initialize_board(self):
        board = np.ones((4, 4))
        board[1, 1] = 0
        return board

    def reward(self,board):
        if self.pegs_left(board) == 2:
            rewards = 100000
        else:
            rewards = -np.power(2,self.pegs_left(board))

        return rewards

    def pegs_left(self,board):
        pegs = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i, j] == 1:
                    pegs = pegs+1
        return pegs
    def action_space(self):
        board = self.initialize_board()
        moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i, j] == 1 or board[i, j] == 0:
                    for d in directions:
                        ni, nj = i + d[0] * 2, j + d[1] * 2
                        if 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] == 0 and board[i + d[0], j + d[1]] == 1:
                            moves.append(((i, j), (ni, nj)))
                        elif 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] == 0 and board[i + d[0], j + d[1]] == 0:
                            moves.append(((i, j), (ni, nj)))
                        elif 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] == 1 and board[i + d[0], j + d[1]] == 1:
                            moves.append(((i, j), (ni, nj)))
                        elif 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] == 1 and board[i + d[0], j + d[1]] == 0:
                            moves.append(((i, j), (ni, nj)))
        return moves

    def valid_moves(self,board):
        moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i, j] == 1:
                    for d in directions:
                        ni, nj = i + d[0] * 2, j + d[1] * 2
                        if 0 <= ni < 4 and 0 <= nj < 4 and board[ni, nj] == 0 and board[i + d[0], j + d[1]] == 1:
                            moves.append(((i, j), (ni, nj)))
        return moves

    def apply_move(self,board,move):
        board_copy = copy.deepcopy(board)
        i, j = move[0]
        ni, nj = move[1]
        board_copy[i, j] = 0
        board_copy[ni, nj] = 1
        board_copy[i + (ni - i) // 2, j + (nj - j) // 2] = 0
        reward = self.reward(board_copy)
        if len(self.valid_moves(board_copy)) == 0:
            done = True
        else:
            done = False
        return board_copy, reward, done

    def reset(self):
        return self.initialize_board()

    def print_board(self,board):
        print(" ", end=" ")
        for i in range(len(board[0] - 1)):
            print(i, end=" ")
        print(" ")
        for i in range(len(board)):
            print(f"{i} ", end="")
            for j in range(len(board[0])):
                if board[i, j] == 0:
                    print("·", end=" ")
                elif board[i, j] == 1:
                    print("x", end=" ")
                else:
                    print(" ", end=" ")
            print(f"{i}")
        print(" ", end=" ")
        for i in range(len(board[0] - 1)):
            print(i, end=" ")
        print(" ")

    # def gen_rand_state(self, pegs):
    #     board = self.initialize_board()
    #     bo_copy = copy.deepcopy(board)
    #     board[1, 1] = 1
    #     randomList = []
    #     # traversing the loop 15 times
    #     while pegs > 0:
    #         # generating a random number in the range 0 to 48
    #         r = random.randint(0, 24)
    #         # checking whether the generated random number is not in the
    #         # randomList
    #         if r not in randomList:
    #             for i in range(len(board)):
    #                 for j in range(len(board[0])):
    #                     if 4 * i + j == r:
    #                         bo_copy[i, j] = 0
    #             # appending the random number to the resultant list, if the condition is true
    #             randomList.append(r)
    #             pegs -= 1
    #     return bo_copy
