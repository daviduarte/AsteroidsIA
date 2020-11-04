import pygame
import datetime
from utils import colors
import random
import math
import numpy as np
import time
from tkinter import *
from tkinter import messagebox
from classes.asteroid import Asteroid
from shapely.geometry import LineString
from shapely.geometry import Polygon
import copy

class Ship:
	def __init__(self, position, angle, SCREEN, asteroids, drawNN = None, color = "random"):
		

		self.brain = []					# Ship brain. Contai the Neural Network weights
		self.INPUT_NEURON = 8
		self.HIDDEN_NEURON1 = 16
		self.HIDDEN_NEURON2 = 16
		self.OUTPUT_NEURON = 4
		self.neural_network()

		self.SCREEN_X = 980
		self.SCREEN_Y = 720

		self.status = 'alive'

		# Class to draw the NN in the left menu
		self.drawNN = None
		if drawNN is not None:
			self.drawNN = drawNN

		#HIDDEN_NEURON1 = 4
		self.dimSurface = 50			# A ship is draw in a square of dimSurface X dimSurface pixels
		self.position = position
		self.angle = angle
		self.SCREEN = SCREEN
		self.shipColor = False
		if color == "random":
			self.setRandomColor()
		else:
			self.shipColor = color
		self.shape = [] 				# The ship is made by a set of edges
		self.shapeBase = []

		self.surface = False			# The surface that contains the ship shape
		self.fireArray = []				# Storage each fire position of the ship
		self.shotSurface = False		# The surface that contains the shot draw

		self.asteroids = asteroids		# Pointer to a list of Asteroid Classes
		self.numAsteroids = len(asteroids)			# Number of asteroids. This is used to control the number of new asteroids if the ship destroys all of them
		self.shotSize = [4,10]		
		self.resultingForce	= np.asarray([0,0], dtype="float64")			# Aceleração da nave		

		self.fireTimeOut = 0			# The time between each shot
		self.burstFire = 100		# The ship is allowed to do a burst fire of 4 shots. Then, it is obliged to wait
		self.burstFireCounter = time.time()

		# For calculate the score
		self.shotsFired = 1
		self.shotsHit = 1				#Start with 1 to does not fuck with score formula
		self.timeout = 0				# The fitness takes into account the ship lifespan
		self.score = 0					# Score of this ship. 
		self.shotLock = 0

		# Create the surface for the ship
		self.drawShape()
		# Create the surfaxe for the shots
		self.initShotSurface()		

		self.seed = time.time()		
		#self.setRandomSeed()

	def newSeed(self):
		self.seed = time.time()

	def setRandomSeed(self):
		random.seed(self.seed)

	def initShotSurface(self):
		self.shotSurface = pygame.Surface(self.shotSize)
		self.shotSurface.fill((0, 255, 0))
		self.shotSurface.set_colorkey((0,255,0)) # Like a Chroma Key, but instead green, is black 
		pygame.draw.line(self.shotSurface, colors.WHITE, (0, 0), (0, 10), 2)		

	def initilizeRandomPosition(self, screenWidth, screenHeight):

		self.position = np.asarray([random.randrange(screenWidth), random.randrange(screenHeight)])
		self.angle = random.randrange(360)

	def drawShape(self):
		self.shape = [(25, 0), (50, 50), (0, 50), (25,0)]
		self.shapeBase = copy.deepcopy(self.shape)
		#self.surface = pygame.Surface((self.dimSurface, self.dimSurface), pygame.SRCALPHA)
		self.surface = pygame.Surface((self.dimSurface, self.dimSurface))
		self.surface.fill((0,255,0))
		self.surface.set_colorkey((0,255,0)) # Like a Chroma Key, but instead green, is black 

		pygame.draw.polygon(self.surface, self.shipColor, self.shape)


	def rotate(self, surface, angle, position):
		w, h = surface.get_size()
		box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
		box_rotate = [p.rotate(angle) for p in box]
		min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
		max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
		origin = (position[0] + min_box[0], position[1] - max_box[1])

		rotated_image = pygame.transform.rotate(surface, angle)

		return [rotated_image, origin]
	


	"""
	Rotate some surface over your own axis
	@param surface some surface
	@return a list containg the rotated surface and the position to be drawn
	"""
	def rotateOwnAxis(self, surface, angle, position):
		w, h = surface.get_size()
		box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]		
		box_rotate = [p.rotate(angle) for p in box]
		min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
		max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])		
		pivot = pygame.math.Vector2(w/2, -h/2)
		pivot_rotate = pivot.rotate(angle)
		pivot_move   = pivot_rotate - pivot
		origin = (position[0] + min_box[0] - pivot_move[0], position[1] - max_box[1] + pivot_move[1])

		rotated_surface = pygame.transform.rotate(surface, angle)
		return [rotated_surface, origin]
		
	def ccw(self, A,B,C):
		return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

	# Return true if line segments AB and CD intersect
	def intersect(self, A,B,C,D):
		return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)


	def verifyColision2(self):

		for i, asteroid in enumerate(self.asteroids):
			for j, fire in enumerate(self.fireArray):				
			
				#fire = C
				vet1 = fire[0]	# Shot start point
				vet2 = fire[1]	# Shot vector	

				x = vet1[0] + vet2[0]*1.5
				y = vet1[1] + vet2[1]*1.5

				line1 = LineString([vet1, [x,y]])
				aster = Polygon(asteroid.shape+asteroid.position)

				intersection = aster.intersects(line1)
				intersection2 = aster.contains(line1)


				if intersection == True or intersection2 == True:
					return [i,j]

		return False


	"""
	For each asteroid edge, verify if hits at a shot or a ship
	"""				
	"""
	def verifyColision(self):
		for i, asteroid in enumerate(self.asteroids):
			for j in range(len(asteroid.shape)):

				asteroidAbsolutePosition = asteroid.position
				A = asteroid.shape[j] + asteroidAbsolutePosition
				

				# The last edge, the one that connects the last vertice to the first
				if j == len(asteroid.shape)-1:
					B = asteroid.shape[0] + asteroidAbsolutePosition
				else:
					B = asteroid.shape[j+1] + asteroidAbsolutePosition

				for k in range(len(self.fireArray)):

					fire = self.fireArray[k]
				
					#fire = C
					vet1 = fire[0]
					vet2 = fire[1]
					shot_distance = fire[2]/2
					angle = fire[3]								

					# Init vertice of shot segment. The shot has a lenght of self.shotSize[1]
					x = vet1[0] + vet2[0]*shot_distance
					y = vet1[1] + vet2[1]*shot_distance	

					# Initial shot vertice
					C = [x,y]


					# Calcule the final vertice of shot segment. Due the rotation, this segment can be at anywhere
					# We could draw the shot as a circle, would be much easier :/
					gunPosition = [0, self.shotSize[1]]
					newGunPosition = []
					newGunPosition.append(gunPosition[0] * math.cos(math.radians(self.angle-180)) - gunPosition[1] * math.sin(math.radians(-self.angle-180)))
					newGunPosition.append(gunPosition[0] * math.sin(math.radians(self.angle-180)) + gunPosition[1] * math.cos(math.radians(-self.angle-180)))					

					# Final shot vertice
					D = [C[0] + newGunPosition[0], C[1] + newGunPosition[1]]
				

					if self.intersect(A,B,C,D):
						return [i, k]
		return False
	"""
	"""
	Update the ship position regarding the self.resultingForce,that is modified by self.accelerate
	and self.friction
	"""
	def updatePosition(self):
	
		w, h = [self.SCREEN_X, self.SCREEN_Y]

		x = self.position[0] + self.resultingForce[0]*1
		y = self.position[1] + self.resultingForce[1]*1


		margin = self.dimSurface/2 
		x_ = x + margin
		y_ = y + margin
		if x_ < 0:
			x = w - margin - 1 # Minus 1 to avoid problems with rounding
		elif x_ > w:
			x = 0 + margin + 1 # PLus 1 to avoid problems with rounding

		if y_ < 0:
			y = h - margin - 1 # Minus 1 to avoid problems with rounding
		elif y_ > h:
			y = 0 + margin + 1 # Plus 1 to avoid problems with rounding

		self.position = np.asarray([x, y])



	def updateAngle(self):
		posX = self.position[0]
		posY = self.position[1]

		# translada
		#x -= dimSurface/2
		#y -= dimSurface/2

		newShape = []
		half = self.dimSurface/2
		for point in self.shapeBase:
			# Translada para a origem
			x = point[0] - half
			y = point[1] - half

			newX = x * math.cos(math.radians(-self.angle)) - y * math.sin(math.radians(-self.angle))
			newY = x * math.sin(math.radians(-self.angle)) + y * math.cos(math.radians(-self.angle))

			# Translade back
			newX += half
			newY += half

			newShape.append([newX, newY])
		self.shape = newShape
		

	def calculeScore(self):

		#diff = self.shotsFired - self.shotsHit
		#if diff == 0:
		#	diff = 1

		hitRate = self.shotsHit/self.shotsFired

		
		
		self.score = (self.shotsHit)*10
		self.score *= self.timeout
		self.score *= hitRate*hitRate
		

		# Minha tentativa
		"""
		self.score = (self.shotsHit)*10
		self.score *= hitRate
		self.score += self.timeout
		"""
		
		

		#self.score = self.timeout

		
		print("ShotsHit: " + str(self.shotsHit))
		print("ShotsFired: " + str(self.shotsFired))
		print("Hitrate: " + str(hitRate))
		print("LifeSpan: " + str(self.timeout))
		print("Score: " + str(self.score))
		print("\n")

	def asteroidColision(self):

		line1 = Polygon(self.shape+self.position)
		for i, asteroid in enumerate(self.asteroids):

			line2 = Polygon(asteroid.shape+asteroid.position)			

			intersection = line1.intersects(line2)
			if intersection == True:
				self.status = 'dead'

				# Its dead, so erase all shots
				self.fireArray = []

				# Calcule the final score
				self.calculeScore()

		return 'alive'
						

	"""
	We have to create news asteroids inside Ship class to use the random seed of this ship
	"""
	def newAsteroids(self):
		
		if len(self.asteroids) > 3:
			return 

		# First, put the ship in origin
		self.position = [self.SCREEN_X/2, self.SCREEN_Y/2]

		# Create a set of 2 more asteroids than the previous set
		for i in range(self.numAsteroids + 2):
			asteroid = Asteroid(self.SCREEN)#, seed = seed)
			self.asteroids.append(asteroid)			

		self.numAsteroids += 2



	"""
		This function update only the positions of each element (asteroids, ships, shots, etc).
		The draw is made by draw() function
	"""
	def update(self):

		self.timeout += 1
		#if self.timeout > 200:
		#	print("Killing the ship by timeout")
		#	self.status = 'dead'

		sensores = [
			self.sensor1(),
			self.sensor2(),
			self.sensor3(),
			self.sensor4(),
			self.sensor5(),
			self.sensor6(),
			self.sensor7(),
			self.sensor8()
			#self.sensor9(),
			#self.sensor10(),
			#self.sensor11(),
			#self.sensor12(),
			#self.sensor13(),
			#self.sensor14(),
			#self.sensor15(),
			#self.sensor16(),
		]
		
		output = self.neural_network_inference(sensores)
		
		action = np.argmax(output)
		
		if output[0] > 0.5:
			self.angle += 10
		if output[1] > 0.5:
			self.angle -= 10
		if output[2] > 0.5:
			self.accelerate()
		if output[3] > 0.5:
			self.fire()		
		
		
		
		# Create a set of new asteroids if the ship destroys all the previous ones
		self.newAsteroids()
		# Modify the ship position
		self.updatePosition()
		# Modify the angle
		self.updateAngle()
		# The friction is a constant force that stop the ship in a certain amount of time
		self.friction()
		#Verify if ship hits asteroid. If yes, the self.status is modified to 'dead'
		self.asteroidColision()

		# If a shot hits an asteroid. 
		hasColision = self.verifyColision2()
		if hasColision is not False:

			# Give the ship an additional amount of lifespan
			#self.timeout = 0
			self.shotsHit += 1
			# Allow new shots
			#self.fireTimeOut = 0

			astroidIndex = hasColision[0]
			shotIndex = hasColision[1]

			# Increase the score 
			#self.score += 1

			# Get the size of destroyed asteroid
			destroyedAsteroidSize = self.asteroids[astroidIndex].size

			# If the asteroid is a big one, count the global var so that we can control the creation of a new set of asteroids
			#if destroyedAsteroidSize == 1:
				#self.numAsteroids += 1

			# If the size of destroyed asteroid is small (3), then just delete it
			if destroyedAsteroidSize < 3:

				# Get the position of the asteroid destroyed
				asteroidPosition = self.asteroids[astroidIndex].position
				velocity = self.asteroids[astroidIndex].velocity

				# Divide the asteroid
				seed = random.randrange(10000)
				#print(seed)
				newAsteroid1 = Asteroid(self.SCREEN, size=destroyedAsteroidSize+1, position = np.asarray(asteroidPosition-10), velocity=velocity+1)#, seed = seed)
				#print(seed)
				newAsteroid2 = Asteroid(self.SCREEN, size=destroyedAsteroidSize+1, position = np.asarray(asteroidPosition-10), velocity=velocity+1)#, seed = seed)
				self.asteroids.append(newAsteroid1)
				self.asteroids.append(newAsteroid2)			

			# Del the asteroid
			del self.asteroids[astroidIndex]

			# Del the shot
			del self.fireArray[shotIndex]
			self.shotLock = 0				

		# Calcule the shots	
		w, h = [self.SCREEN_X, self.SCREEN_Y]

		for i, fire in enumerate(self.fireArray):
			
			vet1 = fire[0]	# Shot origin
			vet2 = fire[1]	# Shot vector ()
			shot_distance = fire[2]

			x = vet1[0] + vet2[0]*1.5
			y = vet1[1] + vet2[1]*1.5

			if x < 0:
				x = w
			elif x > w:
				x = 0

			if y < 0:
				y = h
			elif y > h:
				y = 0

			fire[0] = [x,y]

			if shot_distance < 11:
				self.fireArray[i][2] += 1
			else:
				del self.fireArray[i]		
				self.shotLock = 0	


	"""
		This function just draw each element (asteroids, ships, shots, etc).
		The update of their positions is made by update() function
	"""
	def draw(self):
		
		self.drawNN.draw()

		# Rotate the ship
		rotated_ship = self.rotateOwnAxis(self.surface, self.angle, self.position)
		self.SCREEN.blit(rotated_ship[0], rotated_ship[1])

		# Draw the shots	
		for i, fire in enumerate(self.fireArray):

			vet1 = fire[0]
			vet2 = fire[1]
			#shot_distance = fire[2]/2
			angle = fire[3]

			shotSurface = self.rotateOwnAxis(self.shotSurface, angle, vet1)

			x = vet1[0] + vet2[0]*1.5
			y = vet1[1] + vet2[1]*1.5

			self.SCREEN.blit(shotSurface[0], vet1)

		
	def setRandomColor(self):
		# Select one of 5 available colors
		
		colorsAvailable = [colors.RED, colors.YELLOW, colors.BLUE, colors.WHITE, colors.PINK]
		self.shipColor = colorsAvailable[random.randrange(5)]

	def fire(self):
		
		#  Limit again the shot to a burst of 4 shots
		# If is datatime
		#if (time.time() - 0.5) > self.burstFireCounter:
		#	#if (time.time() - 1) > self.burstFire:
		#		self.burstFire = 0
		#		self.burstFireCounter = time.time()


		#  Limit the shots that ship can do. Only one shot per 0.15 sec		
		#if self.burstFire <= 3 and (time.time() - 0.15) > self.fireTimeOut:
		#if (time.time() - 1) > self.fireTimeOut:
		if self.shotLock == 0:

			self.shotsFired += 1

			# Gun position when the ship is rotated by 0 degrees
			#shipPosition = [int(self.position[0]+self.position[0]/2), int(self.position[1]+self.position[1]/2)]
			gunPosition = [25,0]
			newGunPosition = []

			newGunPosition.append(gunPosition[0] * math.cos(math.radians(-self.angle-90)) - gunPosition[1] * math.sin(math.radians(-self.angle-90)))
			newGunPosition.append(gunPosition[0] * math.sin(math.radians(-self.angle-90)) + gunPosition[1] * math.cos(math.radians(-self.angle-90)))
			# Rotate the gun position to find your new position if the ship is rotated

			shipCenter = [newGunPosition[0] + self.position[0]+25, newGunPosition[1] + self.position[1]+25]
			# x = x_0 + at
			# y = y_0 + bt
			# [[x_0, y_0], [a, b], # of shot steps, angle of shot]
			self.fireArray.append([np.asarray(shipCenter), np.asarray(newGunPosition), 0, self.angle])

			#self.burstFire += 1

			#if self.burstFire > 3:
			#	self.burstFireCounter = time.time()

			self.fireTimeOut = time.time()
			self.shotLock = 1

	def accelerate(self):

		newDirection = np.asarray([math.sin(math.radians(self.angle+180)), math.cos(math.radians(self.angle+180))])

		self.resultingForce += newDirection

		# Limit the max velocity. Change just the direction
		norm = np.linalg.norm(self.resultingForce)
		if norm > 10:
			self.resultingForce = (self.resultingForce/norm)*10		


	# Fricction force in x and y axis
	def friction(self):
		norm = np.linalg.norm(self.resultingForce)

		if self.resultingForce[0] > 0:
			self.resultingForce[0] -= 0.2
		else: 
			self.resultingForce[0] += 0.2

		if self.resultingForce[1] > 0:
			self.resultingForce[1] -= 0.2
		else: 
			self.resultingForce[1] += 0.2


		# Avoid ship moving when it was supposed to be stationary
		if self.resultingForce[0] >= -0.2 and self.resultingForce[0] <= 0.2:
			self.resultingForce[0] = 0
		if self.resultingForce[1] >= -0.2 and self.resultingForce[1] <= 0.2:
			self.resultingForce[1] = 0			


	# Given a list of vertices, given an array containing all lines
	def vertices2Lines(self, vertices):


		lineList = []
		for i, point in enumerate(vertices):			
			A = point

			# The last edge, the one that connects the last vertice to the first
			if i == len(vertices)-1:
				B = vertices[0]
			else:
				B = vertices[i+1]
			lineList.append([A,B])


		return np.asarray(lineList)


	"""
	def sigmoid(self, mat, bias):

		result = []	
		for i in range(len(mat)):
		#for(i ; i < mat.length ; i++){
			result.append(1 / (1 + math.exp(-mat[i] + bias)))
		
		return result
	"""


	"""
	*	Receie a matrix, apply reLU in every element of this matrix
	*	@return the matrix applied to reLU
	"""
	def relu(self, mat, bias):

		result = []	
		sum = 0
		for i in range(len(mat)):
		#for(i ; i < mat.length ; i++){
			result.append(np.amax([0, mat[i] + bias ] ))
		
		return result

	def sigmoid(self, mat, bias):

		result = []	
		for i in range(len(mat)):
			try:
				oi = 1 / (1 + math.exp(-mat[i] + bias))
			except Exception as e: 
				print("Exceção no sigmoid")
				print(mat)
				print(bias)
				print(e)
				time.sleep(1000)
			#print(oi)
			result.append(oi)
		
		return result		

	def neural_network_inference(self, input):

		

		layer1 = np.matmul(input, self.brain[0])	
		layer1 = self.relu(layer1, self.brain[3])
		


		layer2 = np.matmul(layer1, self.brain[1])
		layer2 = self.relu(layer2, self.brain[4])
		

		output = np.matmul(layer2, self.brain[2])
		output = self.sigmoid(output, self.brain[5])
			

		# Draw only if we have the pygame initialized. Work aroud detecdetd
		# TODO Pass this config to draw()
		if self.drawNN is not None:
			self.drawNN.setInput(input)
			self.drawNN.setLayer1(layer1)
			self.drawNN.setLayer2(layer2)
			self.drawNN.setOutput(output)	

		return output

	#def initialize_neural_network(l1, l2, l3):
	def initialize_neural_network(self, l1, l2, l3):


		# Init layer l1
		for i in range(self.INPUT_NEURON):
		#for (i = 0 ; i < input_neuron ; i++){
			l1.append([])
			for j in range(self.HIDDEN_NEURON1):
			#for (j = 0 ; j < hidden_neuron ; j++){
				l1[i].append(random.uniform(-1,1))
			
		
		# Init layer l2
		for i in range(self.HIDDEN_NEURON1):
		#for (i = 0 ; i < hidden_neuron ; i++){
			l2.append([])
			for j in range(self.HIDDEN_NEURON2):
			#for (j = 0 ; j < hidden_neuron2 ; j++){
				l2[i].append(random.uniform(-1,1))
			

		# Init layer l3
		for i in range(self.HIDDEN_NEURON2):
		#for (i = 0 ; i < hidden_neuron2 ; i++){
			l3.append([])
			for j in range(self.OUTPUT_NEURON):
			#for (j = 0 ; j < output_neuron ; j++){
				l3[i].append(random.uniform(-1,1))




	def neural_network(self,):

		# Initialize brains with random values
		#for i in range(individuals):
		#for (var i = 0 ; i < individuals ; i++){
		l1 = []
		l2 = []
		l3 = []

		self.initialize_neural_network(l1, l2, l3)
		#self.initialize_neural_network(l1, l2)

		b1 = random.uniform(-1,1)
		b2 = random.uniform(-1,1)
		b3 = random.uniform(-1,1)

		self.brain = []
		self.brain.append(l1)
		self.brain.append(l2)
		self.brain.append(l3)

		self.brain.append(b1)
		self.brain.append(b2)
		self.brain.append(b3)

		#self.brain_set.append(brain)

	def lineEquationByX(self, point, m, X):
		y = m*X - m*point[0] + point[1]
		return int(y)

	def lineEquationByY(self, point, m, Y):
		if m == 0:
			return 0

		x = (Y + m*point[0] - point[1]) / m
		return int(x)


	def processSensor(self, A, B):
		A = [int(A[0]), int(A[1])]
		B = [int(B[0]), int(B[1])]

		lista = [[A, B, 0]]
		distance = 0

		den = B[0] - A[0]
		# Calculate the sensor's slop
		if den == 0:
			m = 0
		else:
			m = (B[1] - A[1])/(B[0] - A[0])

		nearestAsteroidVerticeList = []
		min_ = float("+Inf")
		while len(lista) > 0:
			A, B, dist2ship = lista[0]
			del lista[0]

			# We've to know the distance of each segment of the sensor, if its is broken in the screen edge and teleported to other side of screen.
			# Then, the distance of a asteroid will be the sum of this segment
			#segment_distance_list = []
			min_segment_distance = float("+Inf")
			min_segment = 0			
			if B[1] < 0:
				# If  == 0 and B is up the screen, so the m does not exist.
				newY = 0
				if m == 0:
					newX = A[0]
				else:
					newX = self.lineEquationByY(A, m, newY)

				segment_distance = np.linalg.norm(np.asarray(A) - np.asarray([newX, newY]))

				# Translate the two point that makes the line to other side of screen
				newY += self.SCREEN_Y
				B_ = [B[0], B[1]+self.SCREEN_Y]

				item_list = [[newX, newY], B_, segment_distance]

				# Append this line to the list
				lista.append(item_list)

				#pygame.draw.line(self.SCREEN, colors.LIGHT_PURPLE, [newX, newY], B_, 2)

				"""				
				if segment_distance < min_segment_distance:
					min_segment_distance = segment_distance
				"""


			if B[1] > self.SCREEN_Y:
				#First, find the point that intersect the screen
				newY = self.SCREEN_Y

				# If  == 0 and B is downside the screen, so the m does not exist.
				if m == 0:
					newX = A[0]
				else:
					newX = self.lineEquationByY(A, m, newY)

				segment_distance = np.linalg.norm(np.asarray(A) - np.asarray([newX, newY]))

				# Translate the two point that makes the line to other side of screen
				newY -= self.SCREEN_Y
				B_ = [B[0], B[1]-self.SCREEN_Y]
				
				item_list = [[newX, newY], B_, segment_distance]

				# Append this line to the list
				lista.append(item_list)

				#pygame.draw.line(self.SCREEN, colors.LIGHT_PURPLE, [newX, newY], B_, 2)
				"""
				segment_distance = np.linalg.norm(A - [newX, newY])
				if segment_distance < min_segment_distance:
					min_segment_distance = segment_distance		
				"""

			if B[0] < 0:
				#First, find the point that intersect the screen
				newX = 0

				# If  == 0 and B is rightside the screen, so the m is "lying down".
				if m == 0:
					newY = A[1]
				else:
					newY = self.lineEquationByX(A, m, newX)

				segment_distance = np.linalg.norm(np.asarray(A) - np.asarray([newX, newY]))

				# Translate the two point that makes the line to other side of screen
				newX += self.SCREEN_X
				B_ = [B[0]+self.SCREEN_X, B[1]]

				item_list = [[newX, newY], B_, segment_distance]

				# Append this line to the list
				lista.append(item_list)

				#pygame.draw.line(self.SCREEN, colors.LIGHT_PURPLE, [newX, newY], B_, 2)
				"""
				segment_distance = np.linalg.norm(A - [newX, newY])
				if segment_distance < min_segment_distance:
					min_segment_distance = segment_distance							
				"""

			if B[0] > self.SCREEN_X:
				#First, find the point that intersect the screen
				newX = self.SCREEN_X

				# If  == 0 and B is rightside the screen, so the m is "lying down".
				if m == 0:
					newY = A[1]
				else:
					newY = self.lineEquationByX(A, m, newX)							

				segment_distance = np.linalg.norm(np.asarray(A) - np.asarray([newX, newY]))

				# Translate the two point that makes the line to other side of screen
				newX -= self.SCREEN_X
				B_ = [B[0]-self.SCREEN_X, B[1]]
				
				item_list = [[newX, newY], B_, segment_distance]

				# Append this line to the list
				lista.append(item_list)

				#pygame.draw.line(self.SCREEN, colors.LIGHT_PURPLE, [newX, newY], B_, 2)
				"""
				segment_distance = np.linalg.norm(A - [newX, newY])
				if segment_distance < min_segment_distance:
					min_segment_distance = segment_distance			
				"""

			#segment_distance_list.append(min_segment_distance)
			# I can verify if asteroids exists in the sensor coming out of the edge of screen, cuz the asteroids 
			# exists only in the screen, so fuck this
			for i, asteroid in enumerate(self.asteroids):

				line1 = Polygon(asteroid.shape+asteroid.position)
				line2 = LineString([(A[0], A[1]), (B[0], B[1])])

				# Intersection points, if exist
				intersection = line1.intersection(line2)

				if intersection.geom_type == 'Point':
					points = [intersection.x, intersection.y]
					nearestAsteroidVerticeList.append(points)

				elif intersection.geom_type == 'GeometryCollection':
					[nearestAsteroidVerticeList.append([p.x, p.y]) for p in intersection]

				elif intersection.geom_type == 'MultiPoint':
					[nearestAsteroidVerticeList.append([p.x, p.y]) for p in intersection]

				elif intersection.geom_type == 'LineString':
					# A line intersect a asteroid in ONLY TWO POINTS, even if the line does not cross the entire polygon.
					# If the line ends in the interior of polygon, this is marked as a point
					assert(len(intersection.coords.xy) == 2)

					x1 = intersection.coords.xy[0][0]
					y1 = intersection.coords.xy[1][0]

					x2 = intersection.coords.xy[0][1]
					y2 = intersection.coords.xy[1][1]

					nearestAsteroidVerticeList.append([x1, y1])
					nearestAsteroidVerticeList.append([x2, y2])

				else:
					print(intersection)
					print(intersection.geom_type)
					raise ValueError('Nor Point Neither GeometryCollection.')

				

			if len(nearestAsteroidVerticeList) > 0:
				for i, point in enumerate(nearestAsteroidVerticeList):
					dist = np.linalg.norm(np.asarray(A)-np.asarray(point)) + dist2ship
					if dist < min_:
						min_ = dist


				#distance += min_

				#return min_
			"""
			if min_ == 0:
				return 1
			else:
				return 1/min_
			"""



		if min_ == float("+Inf"):
			return 500
		else:

			if min_ > 1000:
				print(A)
				print(B)
				print(dist2ship)
				print(min_)
				print("\n")

			return min_
		#return 0


	""" 
	Vertical right sensor
	-------
	-------  
	---*...
	-------
	-------	
	"""
	def sensor1(self):
		sensor_lenght = 500		# Modulus of the sensor vector
		slope = 0 - self.angle  # The sensor follow the ship rotation

		# The sensor starts in center of ship
		sensor_coordinate = [math.cos(math.radians(slope))*sensor_lenght + self.position[0]+25,
					math.sin(math.radians(slope))*sensor_lenght + self.position[1]+25]

		#pygame.draw.line(self.SCREEN, colors.YELLOW, self.position+25, sensor_coordinate, 2)	

		# Verify if intersect some asteroid edge
		A = self.position+25
		B = sensor_coordinate

		dist = self.processSensor(A, B)		

		return dist/sensor_lenght

		
	""" 
	Vertical right sensor
	-------
	-------  
	---*.--
	-----.-
	------.	
	"""
	def sensor2(self):
		sensor_lenght = 500		# Modulus of the sensor vector
		slope = 45 - self.angle  # The sensor follow the ship rotation

		# The sensor starts in center of ship
		sensor_coordinate = [math.cos(math.radians(slope))*sensor_lenght + self.position[0]+25,
					math.sin(math.radians(slope))*sensor_lenght + self.position[1]+25]

		#pygame.draw.line(self.SCREEN, colors.YELLOW, self.position+25, sensor_coordinate, 2)	

		# Verify if intersect some asteroid edge
		A = self.position+25
		B = sensor_coordinate

		dist = self.processSensor(A, B)

		return dist/sensor_lenght
	""" 
	Vertical right sensor
	-------
	-------  
	---*.--
	-----.-
	------.	
	"""
	def sensor3(self):
		sensor_lenght = 500		# Modulus of the sensor vector
		slope = 90 - self.angle  # The sensor follow the ship rotation

		# The sensor starts in center of ship
		sensor_coordinate = [math.cos(math.radians(slope))*sensor_lenght + self.position[0]+25,
					math.sin(math.radians(slope))*sensor_lenght + self.position[1]+25]

		#pygame.draw.line(self.SCREEN, colors.YELLOW, self.position+25, sensor_coordinate, 2)	

		# Verify if intersect some asteroid edge
		A = self.position+25
		B = sensor_coordinate

		dist = self.processSensor(A, B)

		return dist/sensor_lenght


	""" 
	Vertical right sensor
	-------
	-------  
	---*.--
	-----.-
	------.	
	"""
	def sensor4(self):
		sensor_lenght = 500		# Modulus of the sensor vector
		slope = 135 - self.angle  # The sensor follow the ship rotation

		# The sensor starts in center of ship
		sensor_coordinate = [math.cos(math.radians(slope))*sensor_lenght + self.position[0]+25,
					math.sin(math.radians(slope))*sensor_lenght + self.position[1]+25]

		#pygame.draw.line(self.SCREEN, colors.YELLOW, self.position+25, sensor_coordinate, 2)	

		# Verify if intersect some asteroid edge
		A = self.position+25
		B = sensor_coordinate

		dist = self.processSensor(A, B)

		return dist/sensor_lenght


	""" 
	Vertical right sensor
	-------
	-------  
	---*.--
	-----.-
	------.	
	"""
	def sensor5(self):
		sensor_lenght = 500		# Modulus of the sensor vector
		slope = 180 - self.angle  # The sensor follow the ship rotation

		# The sensor starts in center of ship
		sensor_coordinate = [math.cos(math.radians(slope))*sensor_lenght + self.position[0]+25,
					math.sin(math.radians(slope))*sensor_lenght + self.position[1]+25]

		#pygame.draw.line(self.SCREEN, colors.YELLOW, self.position+25, sensor_coordinate, 2)	

		# Verify if intersect some asteroid edge
		A = self.position+25
		B = sensor_coordinate

		dist = self.processSensor(A, B)

		return dist/sensor_lenght
	""" 
	Vertical right sensor
	-------
	-------  
	---*.--
	-----.-
	------.	
	"""
	def sensor6(self):
		sensor_lenght = 500		# Modulus of the sensor vector
		slope = 225 - self.angle  # The sensor follow the ship rotation

		# The sensor starts in center of ship
		sensor_coordinate = [math.cos(math.radians(slope))*sensor_lenght + self.position[0]+25,
					math.sin(math.radians(slope))*sensor_lenght + self.position[1]+25]

		#pygame.draw.line(self.SCREEN, colors.YELLOW, self.position+25, sensor_coordinate, 2)	

		# Verify if intersect some asteroid edge
		A = self.position+25
		B = sensor_coordinate

		dist = self.processSensor(A, B)

		return dist/sensor_lenght

	""" 
	Vertical right sensor
	-------
	-------  
	---*.--
	-----.-
	------.	
	"""
	def sensor7(self):
		sensor_lenght = 500		# Modulus of the sensor vector
		slope = 270 - self.angle  # The sensor follow the ship rotation

		# The sensor starts in center of ship
		sensor_coordinate = [math.cos(math.radians(slope))*sensor_lenght + self.position[0]+25,
					math.sin(math.radians(slope))*sensor_lenght + self.position[1]+25]

		#pygame.draw.line(self.SCREEN, colors.YELLOW, self.position+25, sensor_coordinate, 2)	

		# Verify if intersect some asteroid edge
		A = self.position+25
		B = sensor_coordinate

		dist = self.processSensor(A, B)

		return dist/sensor_lenght

	""" 
	Vertical right sensor
	-------
	-------  
	---*.--
	-----.-
	------.	
	"""
	def sensor8(self):
		sensor_lenght = 500		# Modulus of the sensor vector
		slope = 315 - self.angle  # The sensor follow the ship rotation

		# The sensor starts in center of ship
		sensor_coordinate = [math.cos(math.radians(slope))*sensor_lenght + self.position[0]+25,
					math.sin(math.radians(slope))*sensor_lenght + self.position[1]+25]

		#pygame.draw.line(self.SCREEN, colors.YELLOW, self.position+25, sensor_coordinate, 2)	

		# Verify if intersect some asteroid edge
		A = self.position+25
		B = sensor_coordinate

		dist = self.processSensor(A, B)

		return dist/sensor_lenght
