from peg_sol_5v5 import Peg_Board_55
import random
from plot_board2 import plot_board

# r = random.randint(0, 48)
# if r in [0,1,5,6,7,8,12,13,35,36,40,41,42,43,47,48]:
#     print(r)
# randomList = []
# peg_num = 10
# while peg_num > 0:
#     # generating a random number in the range 0 to 48
#     r = random.randint(0, 48)
#     if r not in randomList:
#         randomList.append(r)
#         peg_num -= 1




env = Peg_Board_55()
board = env.gen_rand_state(20)
plot_board(board,'image2.png')

# env.print_board(board)
# reward = env.reward(board)
# print(reward)

