from random import *
import math
from helper import *
import pygame


class GameObject:

    width = 10
    height = 10
    radius = 5
    alive = True
    typeid = 0
    
    def __init__(self, speed, pos, color, window_width, window_height, typeid):
        self.speed = speed
        self.position = pos
        self.color = color
        self.window_width = window_width
        self.window_height = window_height
        self.typeid = typeid

    def move(self, dest, dt):
        # target vector distance(length)
        distx = (dest.x-self.position.x)
        disty = (dest.y-self.position.y)
        dist = math.sqrt((distx ** 2) + (disty ** 2))
        # vienetinio vektoriaus koordinates
        if(dist < 1):
            return
        vx = (dest.x-self.position.x)/dist
        vy = (dest.y-self.position.y)/dist
        # jei maziau nei 1 dist, neinam

        mx = vx * dt * self.speed
        my = vy * dt * self.speed
        self.position.x += mx if abs(mx) < abs(distx) else distx
        self.position.y += my if abs(my) < abs(disty) else disty

    def collides(self, obj2):
        xcollides = False
        ycollides = False
        if(obj2.position.x > self.position.x):
            if self.position.x + self.width > obj2.position.x:
                xcollides = True
        else:
            if(self.position.x < obj2.position.x + obj2.width):
                xcollides = True

        if (obj2.position.y > self.position.y):
            if (self.position.y + self.height > obj2.position.y):
                ycollides = True
        else:
            if self.position.y < obj2.position.y + obj2.height:
                ycollides = True

        return xcollides and ycollides

class Agent(GameObject):

    agentid = 0
    
    def __init__(self, movesize, pos, color, window_width, window_height, agentid):
        super().__init__(movesize, pos, color, window_width, window_height, 2)
        self.hunger = 500
        self.agentid = agentid
        self.goal = Point(randint(0, window_width), randint(0, window_height))
        self.previous_hunger = self.hunger

    @staticmethod
    def generate_rand_agent(window_width, window_height, agentid):
        return Agent(500,
                    Point(randint(0, window_width - GameObject.width),
                        randint(0, window_height - GameObject.height)),
                    (0,0,255),
                    window_width,
                     window_height,
                     agentid)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, [self.position.x - self.width/2, self.position.y - self.height/2, self.width, self.height])
        myfont = pygame.font.SysFont('Comic Sans MS', 12)
        textsurface = myfont.render(str(int(self.hunger)), False, (0, 0, 255))
        surface.blit(textsurface, (self.position.x, self.position.y))

    def move(self, dt):
        super().move(self.goal, dt)

    def think_basic(self, olist):
        min = 2 * self.window_width
        for obj in olist:
            if type(obj) is Foob:
                dis = obj.position.distance(self.position)
                if dis < min:
                    min = dis
                    _goal = obj.position

        self.goal = _goal

    def think(self, olist):
        pass
    
    def set_action(self, goal):      
        self.goal = self.position
        self.goal.shift(goal.x*10, goal.y*10)

    def get_reward(self):
        if self.previous_hunger < self.hunger:
            return 1
        elif self.previous_hunger >= self.hunger:
            return -0.1
            
    
    def on_collide(self, who):
        if(type(who) is Foob):
            self.hunger += who.nutrition
            self.color = (255, 0, 0)

    def update(self, dt, object_list):
        self.previous_hunger = self.hunger
        self.think(object_list)
        self.move(dt)
        self.hunger -= dt * 50
        if self.hunger < 0:
            self.alive = False


class Foob(GameObject):
    nutrition = 10

    @staticmethod
    def generate_rand_agent(window_width, window_height):
        return Foob(0,
                    Point(randint(0, window_width - GameObject.width),
                        randint(0, window_height - GameObject.height)),
                    (0,255,0),
                    window_width,
                    window_height, 1)

    def on_collide(self, who):
        if(type(who) is Agent):
            self.color = (255, 0, 0)
            self.destroy()
    
    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.position.x, self.position.y), self.radius)
        pass

    def destroy(self):
        self.alive = False

    def update(self, dt, object_list):
        pass
