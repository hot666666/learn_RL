import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
	    # epsilon의 확률로 무작위 행동, (1 - epsilon)의 확률로 탐욕 행동
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # 탐욕 행동은 결국 최대의 Q를 선택하는 action
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        # 다음 상태에서 최대가 되는 Q 함수의 값(next_q) 계산
        if done:  # 목표 상태에 도달(목표 상태에서의 Q 함수는 항상 0)
            next_q = 0
        else:	  # 그 외 상태에선 다음 상태에서의 최대 Q 함수 값을 계산(이는 탐욕 행동에 대한 Q 함수 값)
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q = max(next_qs)

        target = reward + self.gamma * next_q

        # 현재 상태에서의 Q 함수 값(q)과 T(target)와 오차에 학습률을 곱하여 현재 q 업데이트
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


############################################################################


env = GridWorld()
agent = QLearningAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

# 신경망을 이용한 Q 러닝으로 얻은 Q 함수와 정책
env.render_q(agent.Q)
