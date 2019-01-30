import numpy
import random
import math
from decimal import *
from collections import defaultdict


class CliffWorld(object):
    def __init__(self):
        self.startState = [5, 0]
        self.cliffStates = [[5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8]]
        self.cliffReward = -100
        self.goalState = [5, 9]
        self.goalReward = 10
        self.accumulatedReward = 0
        self.transitionCost = -1
        self.currentPosition = self.startState
        self.terminated = False
        self.nr_rows = 6
        self.nr_columns = 10
        self.qValue = [[defaultdict(int) for _ in range(self.nr_columns)] for _ in range(self.nr_rows)]
        self.init_qvalues()
        self.alpha = 0.5

    def init_qvalues(self):
        """
        Our representation uses a 6 x 10 grid of dictionaries,
        each containing the four geographic directions and an allocated q value.
        """
        for row in self.qValue:
            for column in row:
                column["north"] = 0
                column["east"] = 0
                column["south"] = 0
                column["west"] = 0

    def e_greedy(self, position, sloppiness=0.1):
        """
        The epsilon greedy policy is sloppy in a small percentage of moves.
        With a chance of p = sloppiness (default 10%) it chooses randomly instead of greedily.
        :return:
        """
        directions = list(self.qValue[position[0]][position[1]].keys())
        equiprobable_policy = [0.25, 0.25, 0.25, 0.25]
        if random.random() < sloppiness:
            chosen_direction = numpy.random.choice(directions, 1, p=equiprobable_policy)[0]
            return [chosen_direction, self.qValue[position[0]][position[1]][chosen_direction]]
        else:
            return self.get_optimal_q(position)

    def get_optimal_q(self, position):
        current_best = -math.inf
        equals = []
        current_direction = ""
        for direction, value in list(self.qValue[position[0]][position[1]].items()):
            if value > current_best:
                current_best = value
                current_direction = direction
        for direction, value in list(self.qValue[position[0]][position[1]].items()):
            if value == current_best:
                equals.append([direction, value])
        if len(equals) > 1:
            return random.choice(equals)
        else:
            return [current_direction, current_best]

    def move(self, direction):
        if direction == "north":
            if self.currentPosition[0] == 0:
                return
            self.currentPosition = [self.currentPosition[0] - 1, self.currentPosition[1]]

        elif direction == "east":
            if self.currentPosition[1] == 9:
                return
            self.currentPosition = [self.currentPosition[0], self.currentPosition[1] + 1]

        elif direction == "south":
            if self.currentPosition[0] == 5:
                return
            self.currentPosition = [self.currentPosition[0] + 1, self.currentPosition[1]]

        elif direction == "west":
            if self.currentPosition[1] == 0:
                return
            self.currentPosition = [self.currentPosition[0], self.currentPosition[1] - 1]

    def print_optimal_policy_grid(self):
        for row in range(self.nr_rows):
            line_print = "["
            for column in range(self.nr_columns):
                current_best = -math.inf
                direction = ""
                for pair in self.qValue[row][column].items():
                    if current_best < pair[1]:
                        current_best = pair[1]
                        direction = pair[0]
                if current_best == 0:
                    direction = "-"
                line_print += direction + ", "
            print(line_print + "]")
        print("__")

    def print_state_action_values(self):
        for row in range(self.nr_rows):
            for column in range(self.nr_columns):
                print("[" + str(row) + ", " + str(column) + "] -", end=" ")
                for key, value in self.qValue[row][column].items():
                    print(key, ":", round(Decimal(value), 3), end=" ")
                print()

    def play_episode(self, learning_method="Q"):
        """
        Main method for this project.
        Agent uses current map of q values to play one episode in the cliff world until termination.
        During the episode he updates q values according to the learning method provided, either Q-Learning or SARSA
        :param learning_method:
        :return:
        """
        self.currentPosition = self.startState
        e_greedy_move_onwards = self.e_greedy(self.currentPosition)

        while not self.terminated:
            previous_position = [self.currentPosition[0], self.currentPosition[1]]

            if learning_method == "S":
                # updated a' from previous run
                direction = e_greedy_move_onwards[0]
            elif learning_method == "Q":
                direction = self.e_greedy(self.currentPosition)[0]
            else:
                print("Enter 'S' for SARSA or 'Q' for Q-Learning. Now Q has been set as default")
                direction = self.e_greedy(self.currentPosition)[0]

            self.move(direction)
            reward = self.transitionCost
            if self.currentPosition in self.cliffStates:
                reward = self.cliffReward
                self.terminated = True
            elif self.currentPosition == self.goalState:
                reward = self.goalReward
                self.terminated = True
            self.accumulatedReward += reward

            if learning_method == "S":
                e_greedy_move_onwards = self.e_greedy(self.currentPosition)
                self.qValue[previous_position[0]][previous_position[1]][direction] += self.alpha * \
                    (reward + e_greedy_move_onwards[1] - self.qValue[previous_position[0]][previous_position[1]]
                        [direction])
            elif learning_method == "Q":
                self.qValue[previous_position[0]][previous_position[1]][direction] += self.alpha * \
                    (reward + self.get_optimal_q(self.currentPosition)[1] -
                        self.qValue[previous_position[0]][previous_position[1]][direction])

    def policy_learning(self, episodes, learning_method):
        """
        :param
        learning_method: either "Q" for Q-Learning or "S" for SARSA
        episodes: number of iterations in the cliffworld until termination to update state-action values.
        :return:
        """
        for episode in range(episodes):
            self.terminated = False
            self.accumulatedReward = 0
            self.play_episode(learning_method)


example = CliffWorld()
example.policy_learning(1000, "Q")

# Use these two lines for pretty printing.
# example.print_optimal_policy_grid()
# example.print_state_action_values()
