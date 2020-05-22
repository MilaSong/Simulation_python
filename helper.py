from math import sqrt


class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

    def distance(self, b):
        return sqrt((self.x-b.x)**2+(self.y-b.y)**2)

    def mult(self, a,b):
        self.x *= a
        self.y *= b
