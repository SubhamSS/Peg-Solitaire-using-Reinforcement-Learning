# AI-Peg-Solitaire
An AI to solve and assist with the Peg Solitaire game using Reinforcement Learning

# Description

[Peg solitaire](https://en.wikipedia.org/wiki/Peg_solitaire) is a classic single-player board game, played on a board
with holes filled with pegs, where the objective is to remove all but one peg by jumping them over each
other. This project aims to achieve the following:
* Implement a [Deep Q-Network](https://en.wikipedia.org/wiki/Q-learning) to solve different board configurations.
* Implement a methodology to assist a human playing the game. This would work as the following: The bot and the human take turns; the bot moves first followed by the human. The bot’s aim is to make the optimal move considering the human’s possible moves.

The code is written in Python and uses [PyTorch](https://pytorch.org/)

## Environment details

The below environment details a 4 x 4 board. As we change the board to higher dimensions, the State and Actions remain similar (with only changing in dimensions), while Rewards are modeled differently, which will be discussed later. 
* State: An array of 4 x 4 represents the state.
  * `0`: No peg present in the position
  * `1`: Peg is present in the position
* Action: A tuple of size 2 representing the start and end positions
  * Action ((4,2),(2,2)) represents peg from position (4,2) to (2,2) over (3,2)
  * Action space consists of all possible actions (e.g.: 40 for a 4x4 board), but only a few valid at each state
* Reward: Reward is defined as:
  * 100 if the minimum number of pegs is achieved
  * Else, Reward = <math>−2<sup>(no of pegs on board)</sup></math>

## DQN Algorithm

Peg solitaire's discrete actions suit a DQN framework

<b>Objective of DQN</b>: To learn an optimal policy that maximizes the expected discounted sum of
rewards

While running
* 𝑎 ← argmax <math>𝑄(𝑠,𝑎)</math>
* Add <math>s,a,r</math>,<math>s′</math> to memory, where <math>s′</math> = <math>s+a</math>
* If len (memory) > batch size
  * Sample batch of <math>𝑠, 𝑎, 𝑟, s<sup>'</sup></math>
  * <math>𝑄<sub>target</sub></math> ← <math>r+ 𝛾.𝑄′(𝑠′)</math>
  * <math>𝑄<sub>expected</sub></math> ← <math>𝑄(𝑠)</math>
  * <math>ℒ (𝜃)</math> ← ||<math>𝑄<sub>target</sub></math>−<math>𝑄<sub>expected</sub></math>||
  * <math>𝑄′</math>← weights closer to <math>𝑄</math>
  

# 4 x 4 board plots

<img src="Git images/4_4_train.jpg" width="900">

# Higher Dimensional Boards

As we increase the board dimensions, the number of possible board configurations increases exponentially with the number of pegs, making it challenging to explore and learn from all possible states

Further, the reward for a particular move is not immediately
evident ,and the agent may have to make a series of moves to reach a desirable state.
This can make it difficult for the agent to learn an optimal policy, as it needs to consider
the long term consequences of its actions.

Thus, we look for ways to improve the reward model, and modified the rewards to the following:
<img style="float: right;" src="Git images/4_4_train.jpg">
* 5 x 5  Board: Added extra Reward to states which have valid actions:
  * 10<sup>8</sup>if the minimum number of pegs is achieved
  * Else:
    * If state has valid actions: 2 x 2<sup>16−number of pegs on board</sup>
    * Else : 2<sup>16−number of pegs on board</sup>
* Classical Board:
  * New reward term = Modified Reward + <math>n x \sum_{i=1}^n d_i </math>
where n: the number of
empty holes in the board
•
d: the distance
of the hole from the board’s center

# Results

For 5 x 5 board, results after 1000 iterations: 
<img src="Git images/5_5_train.jpg" width="900">

For 7 x 7 board, results after 4000 iterations: 
<img src="Git images/7_7_train.jpg" width="900">

