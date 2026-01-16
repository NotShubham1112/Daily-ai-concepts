import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = (0, 0)
        self.goal = (size-1, size-1)
        
    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        r, c = self.state
        if action == 0: r = max(0, r-1)
        elif action == 1: r = min(self.size-1, r+1)
        elif action == 2: c = max(0, c-1)
        elif action == 3: c = min(self.size-1, c+1)
        
        self.state = (r, c)
        reward = 10 if self.state == self.goal else -1
        done = (self.state == self.goal)
        return self.state, reward, done

def q_learning_demo():
    env = GridWorld(5)
    # Q-table: (size, size, num_actions)
    Q = np.zeros((5, 5, 4))
    lr = 0.1
    gamma = 0.95
    epsilon = 0.1
    
    for episode in range(100):
        env.state = (0, 0)
        done = False
        while not done:
            s = env.state
            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[s[0], s[1]])
            
            s_next, reward, done = env.step(action)
            
            # Q-update: Q(s,a) = Q(s,a) + lr * (R + gamma * max(Q(s')) - Q(s,a))
            target = reward + gamma * np.max(Q[s_next[0], s_next[1]])
            Q[s[0], s[1], action] += lr * (target - Q[s[0], s[1], action])
            
    print("Learned optimal actions at (0,0):", np.argmax(Q[0, 0]))
    print("Action 1 (Down) or 3 (Right) are expected.")

if __name__ == "__main__":
    q_learning_demo()
