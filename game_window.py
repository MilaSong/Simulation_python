import pygame
from random import *
import math
from pygame.locals import *
from game_objects import *
from gameclock import *
import numpy as np
import math
import time
from train import Train

window_width = 1200
window_height = 800

class App:
    graphics = 1
    global window_width
    global window_height
    count = 0
    millis = int(round(time.time() * 1000))
    agents_number = 15
    foob_number = 200
    action_size = 4
    range_x = 8
    range_y = 8
    state_size = range_x * 2 * range_y * 2

    object_list = []

    MAX_SPEED = 60      #pixels per second

    def __init__(self):

        self._display_surf = None
        self.size = self.width, self.height = window_width, window_height
        self.eaten_foob = 0
        self.on_init()
        self.multinit()

    def count_fps(self):
        self.count += 1
        millis_now = int(round(time.time() * 1000))
        if (millis_now - self.millis > 1000):
            self.millis = millis_now
            self.textsurface = self.myfont.render("fps: " + str(self.count), False, (0, 0, 255))
            self.count = 0
        self._display_surf.blit(self.textsurface, (10, 10))

    def generate_objects(self):
        self.object_list = []
        self.dead = []
        self.dones = [0]*self.agents_number
        for i in range(self.agents_number):
            self.object_list.append(Agent.generate_rand_agent(self.width, self.height, i))
        for i in range(self.foob_number):
            self.object_list.append(Foob.generate_rand_agent(self.width, self.height))

        self.seperate_arrays()
        
    def remove_dead_objects(self):
        for i in range(len(self.object_list)):
            if not self.object_list[i].alive:
                if type(self.object_list[i]) == Foob:
                    self.eaten_foob += 1
                    self.object_list[i].position.x = randint(0, self.width)
                    self.object_list[i].position.y = randint(0, self.height)
                    self.object_list[i].alive = True
                    self.object_list[i].color = (0,255,0)
                if type(self.object_list[i]) == Agent:
                    self.dead.append(self.object_list[i])
                    self.dones[self.object_list[i].agentid] = 1


    def update_objects(self, dt):
        for obj1 in self.object_list:
            obj1.update(dt, self.object_list)
            self.check_collisions(obj1)
        self.remove_dead_objects()

    def check_collisions(self, obj1):
        for obj2 in self.object_list:
            if((obj1 is not obj2) and obj1.collides(obj2)):
                obj1.on_collide(obj2)

    def draw_objects(self):
        for obj in self.object_list:
            obj.draw(self._display_surf)

    def keyboard_input(self):
        keys = pygame.key.get_pressed()
        if keys[K_r]:
            self.generate_objects()

    def seperate_arrays(self):
        self.agent_list = []
        self.foob_list = []
        for i in range(len(self.object_list)):
            if self.object_list[i].typeid == 2:
                self.agent_list.append(self.object_list[i])
            else:
                self.foob_list.append(self.object_list[i])

    def printmap(self, grid):
        for i in range(self.range_x*2):
            for j in range(self.range_y*2):
                if int(grid[i][j]) == 2:
                    k = "X"
                elif int(grid[i][j]) == 1:
                    k = "o"
                else:
                    k = "."
                print(f"{k} ", end="")
            print("")
        print("")

    def grid_build(self):
        grid = []
        for i in range(len(self.agent_list)):
            rel_map = np.zeros((self.range_x * 2, self.range_y * 2))
            for j in range(len(self.foob_list)):
                rel_x = int(self.foob_list[j].position.x/10) - int(self.agent_list[i].position.x/10)
                rel_y = int(self.foob_list[j].position.y/10) - int(self.agent_list[i].position.y/10)
                if abs(rel_x) < self.range_x and abs(rel_y) < self.range_y:
                    rel_map[rel_y+self.range_y][rel_x+self.range_x] += 1
            grid.append(rel_map.reshape(-1))
        return np.array(grid)
        
    def set_actions(self, actions):
        for i in range(len(self.object_list)):
            if type(self.object_list[i]) == Agent:
                acts = actions[i].tolist()[0]
                self.object_list[i].set_action(Point(acts[0]-acts[1], acts[2]-acts[3]))

    def get_rewards(self):
        r = []
        for i in range(len(self.object_list)):
            if type(self.object_list[i]) == Agent:
                r.append(self.object_list[i].get_reward())
        return np.array(r)
        
    def on_init(self):
        if self.graphics == 1:
            pygame.init()
            self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.font.init()
            self.myfont = pygame.font.SysFont('Comic Sans MS', 12)
            self.textsurface = self.myfont.render(str(self.count), False, (0, 0, 255))
        self.generate_objects()

    def on_event(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.on_cleanup()

    def on_update(self, dt):
        if self.graphics == 1:
            self.on_event()
            self.keyboard_input()
        self.update_objects(dt)

    def on_render(self, interp):
        if self.graphics == 1:
            self._display_surf.fill((255, 255, 255))
            self.draw_objects()
            self.count_fps()
            pygame.display.update()

    def on_cleanup(self):
        pygame.quit()
        quit()

    def run(self):

        while 1:
            tiks = 0
            self.train_obj.before_episodes()
            self.generate_objects()
            while tiks < 300 and len(self.dead) < 1:
                not_dones = [1-done for done in self.dones]
                grid = self.grid_build()
                actions, log_probs, state_values = self.train_obj.get_action(grid)
                self.set_actions(actions.tolist())

                #self.clock.tick()
                self.on_update(1)
                self.on_render(1)

                rewards = self.get_rewards()
                self.train_obj.train_step(grid, rewards, not_dones, log_probs, state_values, actions)
                tiks += 1

            # ep ended
            self.train_obj.after_episode()
            print(self.eaten_foob)
            self.eaten_foob = 0
            
    brains = []

    def multinit(self):
        for i in range(self.agents_number):
            self.brains.append(Train(state_size = self.state_size, action_size = self.action_size, agent_size = 1, brain_id = i))
        
            
    def run_multiple_brain(self):

        while 1:
            tiks = 0
            for brain in self.brains:
                brain.before_episodes()
            self.generate_objects()
            while tiks < 600 and len(self.dead) < self.agents_number:
                not_dones = [1-done for done in self.dones]
                not_dones_mult = [[x] for x in not_dones]
                grid = self.grid_build()
                grid_mult = [np.array([x]) for x in grid]

                actions = []
                log_probs = []
                state_values = []
                for i, brain in enumerate(self.brains):
                    acts, logs, states = brain.get_action(grid_mult[i])
                    actions.append(acts)
                    log_probs.append(logs)
                    state_values.append(states)
                self.set_actions(actions)

                #self.clock.tick()
                self.on_update(1)
                self.on_render(1)

                rewards = self.get_rewards()
                rewards_mult = [[x] for x in rewards]
                for i, brain in enumerate(self.brains):
                    brain.train_step(grid_mult[i], rewards_mult[i], not_dones_mult[i], log_probs[i], state_values[i], actions[i])
                tiks += 1

            # ep ended
            for brain in self.brains:
                brain.after_episode()
            print(self.eaten_foob)
            self.eaten_foob = 0


if __name__ == "__main__":
    theApp = App()
    theApp.run_multiple_brain()
