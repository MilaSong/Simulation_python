import numpy as np
from agent import ACAgent


class Train:
    #best_model_path = "./best_model.checkpoint"
    state_size = 0
    i_episode = 0

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        print('Number of agents:', self.agent_size)
        print('Size of each action:', self.action_size)

        num_episodes = 300
        self.rollout_length = 30
        self.agent = ACAgent(self.state_size, 
                        self.action_size,
                        self.agent_size,
                        rollout_length=self.rollout_length,
                        lr=1e-3,
                        lr_decay=.95,
                        gamma=.95,
                        value_loss_weight = 1,
                        gradient_clip = 5,
                        )
        self.total_rewards = []
        self.avg_scores = []
        self.max_avg_score = -1
        self.max_score = -1
        self.worsen_tolerance = 10  # for early-stopping training if consistently worsen for # episodes

        self.model_path = f"./data/brain{self.brain_id}.checkpoint"
        self.data_path = f"./data/data{self.brain_id}"
        
        #self.agent.load(self.model_path)
        #self.agent.load(self.best_model_path)
        

    def get_action(self, states):
        actions, log_probs, state_values = self.agent.sample_action(states)      # select actions for 20 envs
        return actions, log_probs, state_values

    def before_episodes(self):
        self.experience = []
        self.scores = np.zeros(self.agent_size)                                           # initialize the score
        self.steps_taken = 0

    def train_step(self, states, rewards, not_dones, log_probs, state_values, actions):
        self.steps_taken += 1


        self.experience.append([actions, rewards, log_probs, not_dones, state_values])
        
        if self.steps_taken % self.rollout_length == 0:
            self.agent.update_model(self.experience)
            del self.experience[:]

        self.scores += rewards                                                   # update the scores

    def after_episode(self):

        episode_score = np.mean(self.scores)                                         # compute the mean score for 20 agents

        self.total_rewards.append(episode_score)
        print("Episodic {} Score: {}".format(self.i_episode, episode_score))
        # self.writedata("Episodic {} Score: {}".format(self.i_episode, episode_score))
        self.writedata(f"{episode_score}")
        if self.max_score < episode_score:                                           # saving new best model
            self.max_score = episode_score
            self.agent.save(self.model_path)

        if len(self.total_rewards) >= 100:                       # record avg score for the latest 100 steps
            latest_avg_score = sum(self.total_rewards[(len(self.total_rewards)-100):]) / 100
            print("100 Episodic Average Score: {}".format(latest_avg_score))
            # self.writedata("100 Episodic Average Score: {}".format(latest_avg_score))
            self.avg_scores.append(latest_avg_score)

            if self.max_avg_score <= latest_avg_score:           # record better results
                self.worsen_tolerance = 10                       # re-count tolerance
                self.max_avg_score = latest_avg_score

            else:                                           
                self.worsen_tolerance -= 1                       # count tolerance
                if self.max_avg_score > 10:                      # continue from last best-model
                    print("Loaded from last best model.")
                    # self.writedata("Loaded from last best model.")
                    self.agent.load(self.model_path)

        self.i_episode += 1

    def writedata(self, string):
        with open(self.data_path, 'a') as f:
            f.write(string + "\n")
                  
