import pygame
import signal
from classes.ship import Ship
from classes.asteroid import Asteroid
from classes.environment import Environment
from utils import colors
import numpy as np
import time
import multiprocessing
from itertools import product
import random
from print_nn import DrawNN
import copy

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from matplotlib.ticker import EngFormatter
import matplotlib.pylab as pylab

fig = plt.figure(figsize=[3.2, 2.5])
#plt.margins(0.01)
ax = fig.add_subplot(111)
ax.yaxis.label.set_color('white')
ax.tick_params(axis='y', colors='white')
plt.rcParams.update({'axes.titlesize': 'small'})
formatter0 = EngFormatter()
ax.yaxis.set_major_formatter(formatter0)

ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)
plt.locator_params(axis='y', nbins=4)

fig.patch.set_facecolor((0, 0, 0))
canvas = agg.FigureCanvasAgg(fig)

MUTATION_PROBABILITY = 1.2

# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_w,
    K_a,
    K_d,
    K_g,
    K_h,
    K_ESCAPE,
    K_SPACE,
    KEYDOWN,
    QUIT,
)


# Globals
pygame.init()
SCREEN = 0
BG = 0
GAME_AREA = 0
METADATA = 0
METADATA_SPACE = 300	# Right space to plot the evolutional graph and neural network visualization
WIDTH = 1280
HEIGHT = 720
GAME_AREA_WIDTH = WIDTH - METADATA_SPACE	# The real space the game will use


def readKeys():

	# Look at every event in the queue
	for event in pygame.event.get():
		# Did the user click the window close button? If so, stop the loop.
		if event.type == QUIT:
			return False	

	keys = pygame.key.get_pressed()
	key_pressed = []

	if keys[pygame.K_q]:
		print("Botão de saída pressianado. Falô brow.")
		return False

	if keys[pygame.K_a]:
		key_pressed.append("A")

	if keys[pygame.K_d]:
		key_pressed.append("D")

	if keys[pygame.K_w]:
		key_pressed.append("W")

	if keys[pygame.K_g]:
		key_pressed.append("G")		

	if keys[pygame.K_h]:
		key_pressed.append("H")				

	if keys[pygame.K_SPACE]:
		key_pressed.append("SPACE")		

	return key_pressed

def initDraw():
	global SCREEN, BG, GAME_AREA, METADATA
	SCREEN = pygame.display.set_mode([WIDTH, HEIGHT], pygame.NOFRAME)
	BG = pygame.image.load("images/background.jpg")
	GAME_AREA = pygame.Surface((GAME_AREA_WIDTH, HEIGHT))
	GAME_AREA.fill((0, 255, 0))
	GAME_AREA.set_colorkey((0,255,0)) # Like a Chroma Key, delete all GREEN in screen

	SCREEN.blit(GAME_AREA, (METADATA_SPACE,0))

	SCREEN.fill(colors.BLACK)
	SCREEN.blit(BG, (METADATA_SPACE,0))		
	pygame.display.flip()

	# Menu esquerdo dos gráficos
	METADATA = pygame.Surface((METADATA_SPACE, HEIGHT))
	SCREEN.blit(METADATA, (0,0))



def draw(ship, drawNeuralNetwork):	
	#for ship in ships:		
	ship.draw()
	
	# Draw the neural network in the left menu
	drawNeuralNetwork.setWeights(ship.brain)
	#drawNeuralNetwork.draw()

	for asteroid in ship.asteroids:
		asteroid.draw()

def updatePosition(keyPressed, ships):

	global MUTATION_PROBABILITY

	if keyPressed == False:
		print("Faloous")
		exit()

	for key in keyPressed:		
		if key == "A":
			ships[0].angle += 10

		if key == "D":
			ships[0].angle -= 10

		if key == "W":
			ships[0].accelerate()

		if key == "SPACE":
			ships[0].fire()		

		if key == "G":			
			MUTATION_PROBABILITY -= 0.05
			print("Valor da nova MUTATION_PROBABILITY: " + str(MUTATION_PROBABILITY))

		if key == "H":
			MUTATION_PROBABILITY += 0.05			
			print("Valor da nova MUTATION_PROBABILITY: " + str(MUTATION_PROBABILITY))



"""
Init the ships
@numShips The ship quantity to be created
@return A array containg instances of Ships
"""
def initShips(numShips, asteroids):

	ships = []
	for i in range(numShips):
		ship = Ship(np.asarray([int(GAME_AREA_WIDTH/2), int(HEIGHT/2)]), 0, GAME_AREA, asteroids, color=colors.YELLOW)
		#ship.initilizeRandomPosition(GAME_AREA_WIDTH, HEIGHT)
		ships.append(ship)

	return ships

def initAsteroids(numAsteroids):
	asteroids = []
	for i in range(numAsteroids):
		asteroid = Asteroid(GAME_AREA)
		#asteroid.newSeed()
		#asteroid.setRandomSeed()
		#ship.initilizeRandomPosition(GAME_AREA_WIDTH, HEIGHT)
		asteroids.append(asteroid)

	return asteroids

def saveGeneration(ships, currentGeneration):
	print("Salvando indivíduos a geração " + str(currentGeneration))
	np.save("checkpoints/"+str(currentGeneration)+".npy", ships)

def loadGeneration(generation, drawNeuralNetwork = None):
	global GAME_AREA
	print("checkpoints/"+str(generation)+".npy")
	ships = np.load("checkpoints/"+str(generation)+".npy", allow_pickle=True)


	for ship in ships:
		ship.SCREEN = GAME_AREA
		ship.initShotSurface()
		ship.drawShape()

		if drawNeuralNetwork is not None:
			ship.drawNN = drawNeuralNetwork
		
		for asteroid in ship.asteroids:
			asteroid.initShape()
			asteroid.SCREEN = GAME_AREA
			#asteroid.setRandomSeed()

	#ships[0].setRandomSeed()			
	return ships

def initializePopulation(numShips, numAsteroids, drawNN = None):

	ships = []
	for i in range(numShips):
		asteroids = initAsteroids(numAsteroids)
		ship = Ship(np.asarray([int(GAME_AREA_WIDTH/2), int(HEIGHT/2)]), 0, GAME_AREA, asteroids, drawNN = drawNN,color=colors.YELLOW)
		ships.append(ship)
	return ships

"""
New Ship ranodm position and restore ship statuss
"""
def restoreScene(ships, numAsteroids):
	for ship in ships:
		asteroids = initAsteroids(numAsteroids)
		ship.asteroids = asteroids
		ship.numAsteroids = len(asteroids)
		ship.position = np.asarray(np.asarray([int(GAME_AREA_WIDTH/2), int(HEIGHT/2)]))
		ship.status = 'alive'
		ship.timeout = 0
		ship.angle = 0
		ship.score = 1
		ship.shotsFired = 1
		ship.shotLock = 0
		ship.shotsHit = 1
		ship.timeout = 0
		ship.resultingForce	= np.asarray([0,0], dtype="float64")

		ship.newSeed()
		ship.setRandomSeed()

#def drawNN():
#	METADATA

def update(ship):
	#for ship in ships:		
	ship.update()

	#drawNN()

	for asteroid in ship.asteroids:
		asteroid.update()

	if ship.status == 'dead':
		print("Ship is dead")
		return 'dead'
	return 'alive'

def plotScore(score):
	myfont = pygame.font.SysFont('font.tff', 30)
	textsurface = myfont.render('Score: ' + str(score), False, (255, 255, 255))
	return textsurface


def plotData(data):

	ax.plot(data, color="red", linewidth=1)
	ax.axes.xaxis.set_visible(False)
	#ax.axes.yaxis.set_visible(False)
	canvas.draw()
	renderer = canvas.get_renderer()

	raw_data = renderer.tostring_rgb()
	size = canvas.get_width_height()

	return pygame.image.fromstring(raw_data, size, "RGB")

"""
	The main function. Contain the main looping
"""
def main(blind, mode, individuals, numAsteroids, generation2load = None):
	global SCREEN, BG, GAME_AREA, METADATA

	currentIndividual = 0
	currentGeneration = 0

	drawNeuralNetwork = None
	if blind == True:
		initDraw()
		drawNeuralNetwork = DrawNN([8,16,16,4], METADATA)


	# Init the ships	
	ships = None
	#if mode == 'genetic':


	generation = 0
	if generation2load is None:
		print("Inicializando uma população do zero")		
		ships = initializePopulation(individuals, numAsteroids, drawNeuralNetwork)	# # of ships and asteroids in each ship
		saveGeneration(ships, currentGeneration)
	else:
		generation = generation2load[0]
		individualNum = generation2load[1]

		ships = loadGeneration(generation, drawNeuralNetwork)
		#ships = loaded_data[0]
		currentGeneration = generation
		currentIndividual = individualNum

	# But a repeatead IF? Yes, motherfucker
	if blind == True:
		lines = np.asarray(np.loadtxt("checkpoints/scores.txt", comments="#", delimiter="-", unpack=False)).astype(int)
		lines = lines[:,0]
		lines = lines[0:generation+1]

		PLOT = plotData(lines)


	#elif mode == "normalGame":
	#	asteroids = initAsteroids(1)
	#	ships = initShips(1, asteroids)

	environment = Environment(individuals, ships)

	clock = pygame.time.Clock()
	running = True



	random.seed(ships[currentIndividual].seed)
	cont = 0
	while running:

		ship = ships[currentIndividual]

		# UPDATE SHIPS EM PARALELO
		if update(ship) == 'dead':
			print("O individuo " + str(currentIndividual) + " morreu")
			currentIndividual += 1			
			

			if generation2load is not None:
				exit()

			if currentIndividual == individuals:
				print("REPLICATING..")						
				environment.replicate(MUTATION_PROBABILITY)	

				print(ships)
				currentIndividual = 0
				restoreScene(ships, numAsteroids)			
				currentGeneration += 1	

				if generation2load is None:
					saveGeneration(ships, currentGeneration)

				print("Iniciando a geração " + str(currentGeneration))

			# Restore the seed of the new individual just before update him
			random.seed(ships[currentIndividual].seed)					

			continue

		if blind == True:

			draw(ship, drawNeuralNetwork)

			# Score text
			SCORE = plotScore(str(ship.shotsHit-1))
			METADATA.blit(SCORE, (125,20))

			# Score chart
			PLOT.set_colorkey((0,255,0))
			METADATA.blit(PLOT, (10,450))

			SCREEN.blit(GAME_AREA, (METADATA_SPACE,0))
			SCREEN.blit(METADATA, (0,0))

			#pygame.image.save(SCREEN, "/home/davi/asteroids/video/frames_dos_games/1/"+str(cont)+".png")
			cont += 1
			
			pygame.display.update()			
			SCREEN.blit(BG, (METADATA_SPACE,0))			

			GAME_AREA.fill((0, 255, 0))
			METADATA.fill((0, 0, 0))

			keyPressed = readKeys()		
			updatePosition(keyPressed, ships)			

			if keyPressed == False:
				running = False		



		clock.tick(10000)

	pygame.quit()

manager = multiprocessing.Manager()
ship_ = manager.list()  # Shared Proxy to a list	
def MainLooping(args):
	ship = args[0] 
	individualNum = args[1]
	#ship_ = args[2]

	print("Iniciando a thread")
	print(individualNum)

	clock = pygame.time.Clock()
	running = True
	random.seed(ship.seed)
	while running:

		# UPDATE SHIPS EM PARALELO

		if update(ship) == 'dead':
			print("Individuo num: " + str(individualNum))
			print("Score: " + str(ship.score))
			print("\n")
			ship_.append(copy.deepcopy([ship, individualNum]))
			return

		clock.tick(1000)

"""
	The main function. Contain the main looping
"""
def main_parallel(blind, mode, individuals, numAsteroids, generation2load = None):
	global SCREEN, BG, GAME_AREA, MUTATION_PROBABILITY

	currentIndividual = 0
	currentGeneration = 0

	if blind == True:
		initDraw()

	# Init the ships	
	ships = None
	#if mode == 'genetic':
	if generation2load is None:
		print("Inicializando uma população do zero")
		ships = initializePopulation(individuals, numAsteroids)	# # of ships and asteroids in each ship
		saveGeneration(ships, currentGeneration)
	else:

		generation = generation2load[0]
		individualNum = generation2load[1]		
		ships = loadGeneration(generation)
		#ships = loaded_data[0]		
		currentGeneration = generation

	environment = Environment(individuals, ships)



	numThreads = 8

	while True:

		# Here I have to generate a list so that each position is a arguments bag that each process will pick 
		argumentBag = []
		for currentIndividual in range(int(individuals/numThreads)):
			for i in range(numThreads):
				individualNum = i + currentIndividual*numThreads
				ship1 = ships[individualNum]

				argumentBag.append([ship1, individualNum])


		with multiprocessing.Pool(processes=numThreads) as pool:
			pool.map(MainLooping, argumentBag)
			pool.close()
			pool.join()			


		for i in range(individuals):
			ships[ship_[i][1]] = copy.deepcopy(ship_[i][0])

		ship_[:] = [] 

		#print(ships[1].brain)
		print("REPLICATING..")						
		environment.replicate(MUTATION_PROBABILITY)	
		currentIndividual = 0
		restoreScene(ships, numAsteroids)			
		currentGeneration += 1	
		saveGeneration(ships, currentGeneration)

		MUTATION_PROBABILITY -= MUTATION_PROBABILITY*0.01
		print("Nova MP: " + str(MUTATION_PROBABILITY))

	pygame.quit()


if __name__ == '__main__':

	# Draw will slow down the search of some individual
	blind = False
	#mode = 'normalGame'
	mode = "genetic"
	# Which generation and individual of this generation?
	#generation2Load = [269, 125]	# Load generation saved. Look at checkpoints/scores.txt
	generation2Load = -1			# Init a new generation

	individuals = 200
	numAsteroids = 9

	try:
		# Run with 1 thread
		if blind == True:
			if generation2Load != -1:
				main(blind, mode, individuals, numAsteroids, generation2Load)
			else:
				main(blind, mode, individuals, numAsteroids)
		else:
			# Run with multiples threads
			if generation2Load != -1:
				main_parallel(blind, mode, individuals, numAsteroids, generation2Load)
			else:
				main_parallel(blind, mode, individuals, numAsteroids)

	except KeyboardInterrupt:
		print("\n\n")
		print("Falo brother")
