# Import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.signal import convolve2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

# parameter and function setups for env

ROW_COUNT = 6
COLUMN_COUNT = 7

RED = 1
YELLOW = 0
EMPTY = -1

def create_board():
    board = np.full((ROW_COUNT, COLUMN_COUNT), EMPTY)
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == EMPTY

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == EMPTY:
            return r

def get_valid_actions(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

def print_board(board):
    display_board = np.where(board == EMPTY, '.', board)
    display_board = np.where(board == RED, '1', display_board)
    display_board = np.where(board == YELLOW, '0', display_board)
    print(np.flip(display_board, 0))

def winning_move(board, piece):
    # Define the kernel to use for convolution
    kernel = np.array([[1, 1, 1, 1]])

    # Check horizontal
    if (convolve2d(board == piece, kernel, mode='valid') == 4).any():
        return True

    # Check vertical
    if (convolve2d(board == piece, kernel.T, mode='valid') == 4).any():
        return True

    # Check positively sloped diagonals
    if (convolve2d(board == piece, np.eye(4), mode='valid') == 4).any():
        return True

    # Check negatively sloped diagonals
    if (convolve2d(board == piece, np.fliplr(np.eye(4)), mode='valid') == 4).any():
        return True

    return False

num_episodes = 20000
savePath = "/content/drive/MyDrive/COSMOS 2024/Connect Four/"
os.makedirs(savePath, exist_ok=True)
training_interval = 100
decay_interval = 500

# Q-Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(ROW_COUNT * COLUMN_COUNT, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, COLUMN_COUNT)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, epsilon=0.7, gamma=1, learning_rate=0.01):
        self.q_network = QNetwork()
        self.t_network = QNetwork()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.epsilon = epsilon
        self.gamma = gamma

    def get_action(self, state, valid_actions):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                valid_q_values = q_values[0][valid_actions]
                return valid_actions[torch.argmax(valid_q_values).item()]
        else:
            return random.choice(valid_actions)

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])

        q_values = self.q_network(state_tensor)
        next_q_values = self.t_network(next_state_tensor)

        q_value = q_values.gather(1, action_tensor.unsqueeze(1))
        print(q_value)
        next_q_value = next_q_values.max(1)[0].unsqueeze(1)
        expected_q_value = reward_tensor + (1 - done) * self.gamma * next_q_value

        loss = nn.MSELoss()(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_tnet(self):
        self.t_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma
        }, path)

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.gamma = checkpoint['gamma']
        else:
            print(f"No saved model found at {path}")

    def epsilon_decay(self):
        self.epsilon = max(0.01, self.epsilon * 0.9)

# Training loop
def train(num_episodes):
    agent1 = QLearningAgent()
    agent2 = QLearningAgent()

    for episode in range(num_episodes):
        board = create_board()
        game_over = False
        turn = 0

        while not game_over:
            valid_actions = get_valid_actions(board)

            if turn == 0:
                action = agent1.get_action(board, valid_actions)
                piece = RED
            else:
                action = agent2.get_action(board, valid_actions)
                piece = YELLOW

            row = get_next_open_row(board, action)
            drop_piece(board, row, action, piece)

            reward = 0
            if winning_move(board, piece):
                reward = 1 if turn == 0 else -1
                game_over = True
            elif len(get_valid_actions(board)) == 0:
                game_over = True

            next_state = board.copy()

            if turn == 0:
                agent1.update(board, action, reward, next_state, game_over)
            else:
                agent2.update(board, action, -reward, next_state, game_over)

            board = next_state
            turn = 1 - turn

        if episode % training_interval == 0:
            print(f"Episode {episode} completed")
            agent1.update_tnet()
            agent2.update_tnet()

        if episode % decay_interval == 0:
            agent1.epsilon_decay()
            agent2.epsilon_decay()

    return agent1, agent2

# Train the agents
agent1, agent2 = train(num_episodes)
agent1.save(savePath + "agent1.pth")
agent2.save(savePath + "agent2.pth")

# Replace ^^ with the below lines if loading from save file

# agent1, agent2 = QLearningAgent(), QLearningAgent()
# agent1.load(savePath + "agent1.pth")
# agent2.load(savePath + "agent2.pth")

# Function to play a game using the trained model
def play_game(agent1, agent2):
    board = create_board()
    game_over = False
    turn = 0

    while not game_over:
        valid_actions = get_valid_actions(board)

        if turn == 0:
            action = agent1.get_action(board, valid_actions)
            piece = RED
        else:
            action = agent2.get_action(board, valid_actions)
            piece = YELLOW

        row = get_next_open_row(board, action)
        drop_piece(board, row, action, piece)

        if winning_move(board, piece):
            print_board(board)
            print(f"Player {turn + 1} wins!")
            game_over = True
        elif len(get_valid_actions(board)) == 0:
            print_board(board)
            print("It's a draw!")
            game_over = True

        print_board(board)
        turn = 1 - turn

# Play a game with the trained agents
play_game(agent1, agent2)
