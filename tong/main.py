# Colors for console
from algorithms import RL

def frozen_lake():
  rl = RL('FrozenLake-v0')
  Q = rl.q_learning(10000, 0.9, 0.02)
  policy = rl.get_policy(Q, False)
  rl.print_policy(policy)
  rl.get_score(policy)

def blackjack_state_mapper(s):
  return (1+ s[0]) * (1+ s[1]) + (1 if s[2] else 2)

def blackjack():
  rl = RL('Blackjack-v0', 704, 2, blackjack_state_mapper)
  Q = rl.on_policy_mc(30000, True, 1, 0.01)
  policy = rl.get_policy(Q, False)
  wins = 0
  for i in range(1000):
    _, _, _, win = rl.run_episode(policy, False)
    if win:
      wins +=1
  print((wins/1000*100) if wins > 0 else 0)


blackjack()
# frozen_lake()