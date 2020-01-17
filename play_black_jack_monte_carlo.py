'''
monte-carlo is used when you have system and you donot know the state transition, when you have some states and can perform actions but you dont
know what are the states you are going to end up.
Monte-carlo allows to play those MDP games
we have to deal with the explore exploit dilemma
with some percentage of time episolon you choose random action and with the reset of the time we choose the best know action
off policy means using two policy one policy to perform exploration on the envrionment and the second policy to learn the best state action map 

the rules of the blackjack envrionment:
    
'''
import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    epsilon = 0.05
    GAMMA = 1.0

    agentSumSpace = [i for i in range(4, 22)]
    #the agent allowable sum of scores
    dealerShowCardSpace = [i+1 for i in range(10)]
    #the card the dealer has to show 
    agentAceSpace = [False, True]
    actionSpace = [0, 1] 
    #at each time step the agent can choose stick or hit
    stateSpace = []
    #used to build the target policy
    
    Q = {}
    #is the Q(s,a) q_value function for the state and action pair
    C = {}
    #the relative weight of those trajecotries to occur
    for total in agentSumSpace:
        for card in dealerShowCardSpace:
            for ace in agentAceSpace:
                for action in actionSpace:
                    Q[((total, card, ace), action)] = 0
                    C[((total, card, ace), action)] = 0
                stateSpace.append((total, card, ace))
    #note the Q(s, a) has the following form --> Q[(total_su,, dealers_card, whether you have and ace or not), prob_hit, prob_stick]
    #state space --> [[total_sum,card, whether_ace]]
    #state is list of list each row of the state is list consist of three main features hte totalscore the card showing up whether and ace or not

    targetPolicy = {}
    #is used to get the optimal possble behavoir and so the maximum expected return
    for state in stateSpace:
        values = np.array([Q[(state, a)] for a in actionSpace ])
        #state is list con
        best = np.random.choice(np.where(values==values.max())[0]) 
        
        targetPolicy[state] = actionSpace[best]

    num_of_episodes = 1000000
    for i in range(num_of_episodes):        
        memory = []
        if i % 1000 == 0:
            print('starting episode', i)
        behaviorPolicy = {}
        #is and epsilonilon greedy policy from the target policy
        for state in stateSpace:
            rand = np.random.random()
            if rand < 1 - epsilon:
                behaviorPolicy[state] = [targetPolicy[state]]
                #for 1- epsiloniolon of our time select greedy action from the list of all possible actions
            else:
                behaviorPolicy[state] = actionSpace
        #now we have the behaviorPolicy which an epsilonilon greedy version of the target policy
        observation = env.reset()
        #for each episode reset the envroinment
        done = False
        while not done:
            action = np.random.choice(behaviorPolicy[observation])
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], observation[2], action, reward))
            observation = observation_
        memory.append((observation[0], observation[1], observation[2], action, reward))  
        #observation total , dealer card and whether the player has usable ace
        #fill the replay buffer
        
        
        #at the end of each epsiloniode we want to iterate in the reversed memeory
        G = 0
        W = 1
        last = True
        for playerSum, dealerCard, usableAce, action, reward in reversed(memory):
            #at the end of each epsilonisode we want to learn about our performance so we iterate reversely on our agent replay buffer
            sa = ((playerSum, dealerCard, usableAce), action)
            #state, action tuple is 
            if last:
                last = False
            else:
                C[sa] += W
                Q[sa] += (W / C[sa])*(G-Q[sa])  
                #update using the new sample G and the old sample
                values = np.array([Q[(state, a)] for a in actionSpace ])
                #creat the Q_table
                best = np.random.choice(np.where(values==values.max())[0])        
                targetPolicy[state] = actionSpace[best]
                if action != targetPolicy[state]:
                    break
                    #note q_learning learns from the greedy action if we choose suboptimal action we really break the learning  process
                if len(behaviorPolicy[state]) == 1:
                    #note if the length of the array behavoirPolicyState is one this means you are selecting greedy action!!
                    prob = 1 - epsilon
                else:
                    prob = epsilon / len(behaviorPolicy[state])             
                W *= 1/prob                             
            G = GAMMA*G + reward
        if epsilon - 1e-7 > 0:
            epsilon -= 1e-7
            #decay the epsilonilon
        else:
            epsilon = 0
            
            
    num_of_episodes = 1000
    #then take 1000 episodes and test the game
    rewards = np.zeros(num_of_episodes)
    totalReward = 0
    wins = 0
    losses = 0
    draws = 0
    print('getting ready to test target policy')   
    for i in range(num_of_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = targetPolicy[observation]
            observation_, reward, done, info = env.step(action)            
            observation = observation_
        totalReward += reward
        rewards[i] = totalReward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1
    
    wins /= num_of_episodes
    losses /= num_of_episodes
    draws /= num_of_episodes
    print('win rate', wins, 'loss rate', losses, 'draw rate', draws)
    plt.plot(rewards)
    plt.show()