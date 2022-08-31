import pygame
import math
from queue import PriorityQueue


WIDTH = 800 # width of screen
WIN = pygame.display.set_mode((WIDTH, WIDTH)) # setting Display 800x800
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)         # Closed
GREEN = (0, 255, 0)       # Open
BLUE = (0, 255, 0)        
YELLOW = (255, 255, 0) 
WHITE = (255, 255, 255)  
BLACK = (0, 0, 0)          # Wall
PURPLE = (128, 0, 128)     # Path
ORANGE = (255, 165 ,0)     # start
GREY = (128, 128, 128)     # Grid lines
TURQUOISE = (64, 224, 208) # End

#This class made to handle all the event for a spot(or node or box) on grid such to draw the box, to check whether the 
# node is closed,open or barrier and also to make node as closed,open and barrier, and to update the neighbors of a node.
class Spot:

# for more on _init__() function please go throught :shorturl.at/mnCUY
# if row col is (0,1) then x,y would be (0,16),it is the position of (0,1) cordinate box top-right corner
# every node will contains it's row and col number(0-index) and it's neighbours list and total rows(50)
	
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []  
		self.width = width
		self.total_rows = total_rows

# This fuction return the number of row and column of box such as (3,4) means this box is in 3rd row and 4th column.(0-Index)
	
	def get_pos(self):
		return self.row, self.col

# Below  functions will tell and will set the property of Node (or box) based on color we have decided above. 
	def is_closed(self):
		return self.color == RED

	def is_open(self):
		return self.color == GREEN

	def is_barrier(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == TURQUOISE

	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_open(self):
		self.color = GREEN

	def make_barrier(self):
		self.color = BLACK

	def make_end(self):
		self.color = TURQUOISE

# If current box is a part of shortest path then make it's color to purple
	def make_path(self):
		self.color = PURPLE

# Draw a rect angle box on window. Argument "win" tells us where to draw (x,y) are the topmost right corner point of 
# node, it will draw a rectangle of 16x16 and it point(x,y) topmost right corner of rectnagle 
	
	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

# Update the neighbors of a Node (or Box or spot)
	def update_neighbors(self, grid):
		
		self.neighbors = [] # List that will contain the neighbors of a Node (or Box or spot)

        # Check forDown neighbors Nodes Conditions:
        # 1. row < total_row -1 (ans total_row are 50 so index should be 48 because last row (49th )
        # doesn't have any down neighbor 2.The current spot should not a barrier)
		
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): 
			self.neighbors.append(grid[self.row + 1][self.col])

        # Check for up neighbors,conditions:
        # 1. row>0(as 0th row doesn't have any upper neighbors) 
        # 2.The current spot should not a barrier

		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append(grid[self.row - 1][self.col])

		# Check for right neighbors: conditions 
		# 1. col >0 (as 0th column doesn't have any right neighbours)
        # 2. The current spot should not a barrier
		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append(grid[self.row][self.col + 1])

        # Check for left neighbors: conditions
        # 1.col < total_rows-1(as 49th col doesn't have any left neighbour)
        # 2.The current spot should not a barrier

		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append(grid[self.row][self.col - 1])

	def __lt__(self, other):
		return False

# it's heuristic fuction, and here we are using manhattan distance
# p1 is like(3,4) where 3 is x and 4 is y cordinate
def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

# After reaching to end Node, this function will create path
# arguments : "came_from" => parents list (or closed list),"current" => end node,"draw" => lambda function

def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		draw()

# This is implementation of A* Algorithm

def algorithm(draw, grid, start, end):
# Suppose we have two node with same f(n) value then we can make difference that which one is inserted before to other one
	count = 0
# This is open set, it will contain the open node and we have used PriorityQueue because it's a Data structure which 
# will contain elements in sorted order
	open_set = PriorityQueue()

# Here we are putting start Node in open list, and start node has f(n) value 0. 
	open_set.put((0, count, start))

# The list "came_from" is closed list, it will contains the parents node or closed node
	came_from = {}

# Initially, Set g(n) for all n, to infinity 
	g_score = {spot: float("inf") for row in grid for spot in row}

# The start Node has g score zero
	g_score[start] = 0

# Initially, Set f(n) for all n, to infinity 
	f_score = {spot: float("inf") for row in grid for spot in row}

# The start Node will have f score equal to h(start) and g(start) =0. 
	f_score[start] = h(start.get_pos(), end.get_pos())

# This is open list, Through PriorityQueue we can get Node with minimum F(n) value but we cann't tall whether this node
# is opened or closed,So to check this we have used this list.
	open_set_hash = {start}

 # If open_set is empty then it means we have checked every possible single node, and if yet we haven't find path then
 # path doesn't exist
	while not open_set.empty():
		# If someone wants to Quit, they can quit by pressing "ESC" key on keywords
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

# Priority contain there things for every node in order of {F(n),count,Node}, So index of Node is 2.
		current = open_set.get()[2]

# Remove that node at which we are currently traversing 
		open_set_hash.remove(current)

# If current Node is end Node then it's means we have shortest path,and Draw it.
		if current == end:
			reconstruct_path(came_from, end, draw)
			end.make_end()
			return True

# Traverse the neighbors of current Node and update their F(n) value.
		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + 1

# This conditions tells that we have find another a less cost to reach this neighbor with less than previous cost path 
			if temp_g_score < g_score[neighbor]:
				# Update parent Node of this neighbor
				came_from[neighbor] = current
				# Update the g score of this neighbor
				g_score[neighbor] = temp_g_score
                # Update the f score of this neighbor
				f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())

# If we are coming this node first time, add it to open_set_hash and if we add it will become open node, so make it open as well.
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((f_score[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					neighbor.make_open()

		draw()
# If current node it not start node then make it close node
		if current != start:
			current.make_closed()

# If There is no path,return False
	return False


# This function will create every single Node,But it won't draw grid line so we have to draw seperatly 
# It will return 2-D list of Node,we are storing or making grid in 2D- list data structure so we can traverse whole list
# it will just create 2-D list with Node and nothing else
def make_grid(rows, width):
	grid = [] # A list row,and row are made up with box
	gap = width // rows
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			spot = Spot(i, j, gap, rows)
			grid[i].append(spot)

	return grid

# This function will draw grid on window screen.
def draw_grid(win, rows, width):
	gap = width // rows
	for i in range(rows):
		 # Passing argumnet are: window(where to draw),color(color of line),starting position and end postion
		pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
		for j in range(rows):
			pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


# This function will draw Sopt and make gird complete on window screen,it will complete everything.
# grid is data structure we have made because we can not traverse window grid.
def draw(win, grid, rows, width):
	win.fill(WHITE)

	for row in grid:
		for spot in row:
			spot.draw(win)

	draw_grid(win, rows, width)
	pygame.display.update()

# Find where and at which ro,col we have clicked
# Arguent pos --> postion on window where we clicked it can  be (0--800,0--800),number of rows and total width of screen
# This function return (row,col) eg (2,4) 2nd row and 4th column it means we have clicked on 2nd row and 4th column box on window.
def get_clicked_pos(pos, rows, width):
	gap = width // rows
	y, x = pos

	row = y // gap
	col = x // gap
 # If pos is (12,10) then row = 12//16 = 0,col = 10//16 = 0 (integer division)mena 0th row and 0th column
	return row, col


def main(win, width):
	ROWS = 50

# Make a 2-D list, so we can traverse in it and update the window the help of this 2-D grid.
	grid = make_grid(ROWS, width)

	start = None  # Start Node
	end = None    # End Node
 
	run = True    # While running loop
	while run:
		# Calling function to draw gird on window screen.
		# Now we will handle all the events happening on window screen.
		draw(win, grid, ROWS, width)

        # Get the events happening on window screen.
		for event in pygame.event.get():

			# If it is Quit event, means if somone pressed ESC button on keyword.
			if event.type == pygame.QUIT:
				run = False
# Now handle the all the event,click on mouse button events,like what should be done if left,right dowm up buttion is pressed
# For more detailes visit :https://www.pygame.org/docs/ref/mouse.html#pygame.mouse.get_pressed
			if pygame.mouse.get_pressed()[0]:   # LEFT
				pos = pygame.mouse.get_pos()    # pos can be (0---800,0--800)

# So now we want to find out at which row,col (or spot or Box) we just have pressed, so for that we have made one helper function use it
				row, col = get_clicked_pos(pos, ROWS, width)
				spot = grid[row][col]
# If yet, we haven't make start node or box or end, then make it start or end and always make sure that when you are
# making this spot or box as start node, it shouldn't not have made end before and vice-versa
				if not start and spot != end:
					start = spot
					start.make_start()

				elif not end and spot != start:
					end = spot
					end.make_end()

# If we have already made start and end node then now make it's barrier node. 
				elif spot != end and spot != start:
					spot.make_barrier()

# If we press right button on mouse
			elif pygame.mouse.get_pressed()[2]: # RIGHT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				spot = grid[row][col]
				spot.reset()
# If the reset spot was a barrier than we don't have to do anything but if it was a start or end node, then we have to make them None
# So in we can check in next loop then we haven't make start or end (depending whether it was start or end node) node yet
				if spot == start:
					start = None
				elif spot == end:
					end = None

			if event.type == pygame.KEYDOWN:
				# Start algorithm if somone press "space bar" on keyword and also check whether we have defined our end and start node.
				if event.key == pygame.K_SPACE and start and end:
					for row in grid:
						for spot in row:
							spot.update_neighbors(grid)
# For more on lambda : https://www.w3schools.com/python/python_lambda.asp
					algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
					
               # If somone wants to reset the grid on window screen, then make new 2-D list
				if event.key == pygame.K_c:
					start = None
					end = None
					grid = make_grid(ROWS, width)

	pygame.quit()

# calling main function
main(WIN, WIDTH)