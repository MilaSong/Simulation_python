import numpy as np

class Agent:
    def __init__(self, learn_rate, discount, position):
        self.learn_rate = learn_rate
        self.discount = discount
        self.position = position

class Foobie:
    def __init__(self, reward, position):
        self.reward = reward
        self.position = position

class Qtable:
    def __init__(self, left, right):
        self.left = left
        self.right = right

        
# Objects
goobie = Agent(0.1, 0.95, 1)
big_foob = Foobie(10, 4)
small_foob = Foobie(2, 0)
qtable = Qtable(np.zeros(5), np.zeros(5))

# Decay parameters
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
decay = 0.01


# Functions
def bellman_equation(reward, q, old, new):
    q[old] += goobie.learn_rate * (reward + goobie.discount * max( q[old], q[new]))

    
# Map generation
str = []
[ str.append('+') if i == big_foob.position or  i == small_foob.position else str.append('-') for i in range(5) ]
def_str = str.copy()


# Game loop
n = len(str)
str[goobie.position] = 'G'
for i in range(1000):
    print(' '.join(str))
    str[goobie.position] = def_str[goobie.position]

    exploration_threshold = np.random.uniform(0,1)
    if exploration_threshold > exploration_rate:
        step = 1 if qtable.right[goobie.position] > qtable.left[goobie.position] else 0
    else:
        step = int(np.random.binomial(1, 0.6, 1))
        
    old_pos = goobie.position
    goobie.position = goobie.position * step + step * 1
    
    if goobie.position >= n-1:
        goobie.position = n-1

    if goobie.position == big_foob.position:
        bellman_equation(big_foob.reward, qtable.right, old_pos, goobie.position)
    elif goobie.position == small_foob.position:
        bellman_equation(small_foob.reward, qtable.left, old_pos, goobie.position)
    else:
        if step == 1:
            bellman_equation(big_foob.reward, qtable.right, old_pos, goobie.position)
        else:
            bellman_equation(small_foob.reward, qtable.left, old_pos, goobie.position)
            
    str[goobie.position] = 'G'

    # Decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-decay * i)
    
print(f"Qtable_right: {qtable.right}")
print(f"Qtable_left: {qtable.left}")
