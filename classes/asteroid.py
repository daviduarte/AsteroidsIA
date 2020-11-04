from utils import colors
import pygame
import math
import numpy as np
import random
import time

class Asteroid:
	def __init__(self, SCREEN, size = 1, color = colors.WHITE, position='random', velocity = 4):#, seed = False):

		self.SCREEN = SCREEN
		self.position = False
		self.color = color
		self.surface = False
		self.shape = False
		self.size = size		# Size 1 = Bigger asteroid, Size 2 = Medium, Size 3 = Small


		self.SCREEN_X = 980
		self.SCREEN_Y = 720

		self.velocity = velocity		#

		"""
		self.seed = False
		if seed is False:
			self.seed = time.time()					
		else:
			self.seed = seed
		"""

		#self.setRandomSeed()

		angle = random.randrange(360)
		if angle == 0 or angle == 90 or angle == 180 or angle == 270:
			angle += 20

		self.direction = math.radians(angle)
		self.direction = [math.sin(self.direction), math.cos(self.direction)]	

		if position == 'random':
			self.initilizeRandomPosition()
		else:
			self.position = position
		self.initShape()			


	"""
	def newSeed(self):
		self.seed = time.time()

	def setRandomSeed(self):
		random.seed(self.seed)
	"""

	"""
	A asteroid can be in any position, except in center
	"""
	def initilizeRandomPosition(self):
		middle_x = int(self.SCREEN_X/2)
		middle_y = int(self.SCREEN_Y/2)

		numbers = []
		x = [random.randrange(0, middle_x-100), random.randrange(middle_x+100, self.SCREEN_X)]
		y = [random.randrange(0, middle_y-100), random.randrange(middle_y+100, self.SCREEN_Y)]

		x = x[random.randrange(2)]
		y = y[random.randrange(2)]

		#random.range()
		#y1 = random.randrange(0, middle_y-150)
		#numbers.append([x1, y1])

		#x2 = random.randrange(middle_x+150, self.SCREEN_X)
		#y2 = random.randrange(middle_y+150, self.SCREEN_Y)
		#numbers.append([x2, y2])
		#print("\n\n\n")
		#print(numbers)
		self.position = np.asarray([x,y])




	def initShape(self):
		pi2 = 2 * 3.14
		n = 5
		radius = 100

		self.shape = [np.asarray([20, 0])*(1/self.size), 
					  np.asarray([60, 0])*(1/self.size), 
					  np.asarray([100, 70])*(1/self.size),
					  np.asarray([50, 80])*(1/self.size), 
					  np.asarray([0, 50])*(1/self.size),
					  np.asarray([20, 0])*(1/self.size)]
		#self.surface = pygame.Surface((self.dimSurface, self.dimSurface), pygame.SRCALPHA)
		self.surface = pygame.Surface((100, 100))
		self.surface.fill((0,255,0))
		self.surface.set_colorkey((0,255,0)) # Like a Chroma Key, but instead green, is black 

		pygame.draw.polygon(self.surface, self.color, self.shape)

	"""
	Move the asteroid with constant acceleration to direction 'self.direction'
	"""
	def move(self):

		w, h = [self.SCREEN_X, self.SCREEN_Y]

		x = self.position[0] + self.direction[0]*self.velocity
		y = self.position[1] + self.direction[1]*self.velocity

		if x < 0:
			x = w
		elif x > w:
			x = 0

		if y < 0:
			y = h
		elif y > h:
			y = 0

		self.position = np.asarray([x,y])

	def update(self):
		self.move()

	def draw(self):
		self.SCREEN.blit(self.surface, self.position)