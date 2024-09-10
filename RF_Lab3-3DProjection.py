# Robert Fernald
# CSCI 3800 Lab 3
# Import a library of functions called 'pygame'
import pygame
import numpy as np
from math import pi, sin, cos, tan, radians

class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Point3D:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        
class Line3D():

    def __init__(self, start, end):
        self.start = start
        self.end = end

def loadOBJ(filename):
    
    vertices = []
    indices = []
    lines = []

    f = open(filename, "r")
    for line in f:
        t = str.split(line)
        if not t:
            continue
        if t[0] == "v":
            vertices.append(Point3D(float(t[1]),float(t[2]),float(t[3])))
            
        if t[0] == "f":
            for i in range(1,len(t) - 1):
                index1 = int(str.split(t[i],"/")[0])
                index2 = int(str.split(t[i+1],"/")[0])
                indices.append((index1,index2))
            
    f.close()

    #Add faces as lines
    for index_pair in indices:
        index1 = index_pair[0]
        index2 = index_pair[1]
        lines.append(Line3D(vertices[index1 - 1],vertices[index2 - 1]))
        
    #Find duplicates
    duplicates = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            
            # Case 1 -> Starts match
            if line1.start.x == line2.start.x and line1.start.y == line2.start.y and line1.start.z == line2.start.z:
                if line1.end.x == line2.end.x and line1.end.y == line2.end.y and line1.end.z == line2.end.z:
                    duplicates.append(j)
            # Case 2 -> Start matches end
            if line1.start.x == line2.end.x and line1.start.y == line2.end.y and line1.start.z == line2.end.z:
                if line1.end.x == line2.start.x and line1.end.y == line2.start.y and line1.end.z == line2.start.z:
                    duplicates.append(j)
                    
    duplicates = list(set(duplicates))
    duplicates.sort()
    duplicates = duplicates[::-1]

    #Remove duplicates
    for j in range(len(duplicates)):
        del lines[duplicates[j]]

    return lines

def loadHouse():
    house = []
    #Floor
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(5, 0, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 0, 5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(-5, 0, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 0, -5)))
    #Ceiling
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 5, -5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(5, 5, 5), Point3D(-5, 5, 5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(-5, 5, -5)))
    #Walls
    house.append(Line3D(Point3D(-5, 0, -5), Point3D(-5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(5, 0, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(-5, 0, 5), Point3D(-5, 5, 5)))
    #Door
    house.append(Line3D(Point3D(-1, 0, 5), Point3D(-1, 3, 5)))
    house.append(Line3D(Point3D(-1, 3, 5), Point3D(1, 3, 5)))
    house.append(Line3D(Point3D(1, 3, 5), Point3D(1, 0, 5)))
    #Roof
    house.append(Line3D(Point3D(-5, 5, -5), Point3D(0, 8, -5)))
    house.append(Line3D(Point3D(0, 8, -5), Point3D(5, 5, -5)))
    house.append(Line3D(Point3D(-5, 5, 5), Point3D(0, 8, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(5, 5, 5)))
    house.append(Line3D(Point3D(0, 8, 5), Point3D(0, 8, -5)))
	
    return house

def loadCar():
    car = []
    #Front Side
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-2, 3, 2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(2, 3, 2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(3, 2, 2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 1, 2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(-3, 1, 2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 2, 2)))

    #Back Side
    car.append(Line3D(Point3D(-3, 2, -2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(-2, 3, -2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, -2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 2, -2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(3, 1, -2), Point3D(-3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, -2), Point3D(-3, 2, -2)))
    
    #Connectors
    car.append(Line3D(Point3D(-3, 2, 2), Point3D(-3, 2, -2)))
    car.append(Line3D(Point3D(-2, 3, 2), Point3D(-2, 3, -2)))
    car.append(Line3D(Point3D(2, 3, 2), Point3D(2, 3, -2)))
    car.append(Line3D(Point3D(3, 2, 2), Point3D(3, 2, -2)))
    car.append(Line3D(Point3D(3, 1, 2), Point3D(3, 1, -2)))
    car.append(Line3D(Point3D(-3, 1, 2), Point3D(-3, 1, -2)))

    return car

def loadTire():
    tire = []
    #Front Side
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-.5, 1, .5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(.5, 1, .5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(1, .5, .5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, -.5, .5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(.5, -1, .5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(-.5, -1, .5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-1, -.5, .5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, .5, .5)))

    #Back Side
    tire.append(Line3D(Point3D(-1, .5, -.5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, -.5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, -.5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, .5, -.5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, -.5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(.5, -1, -.5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, -.5), Point3D(-1, -.5, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, -.5), Point3D(-1, .5, -.5)))

    #Connectors
    tire.append(Line3D(Point3D(-1, .5, .5), Point3D(-1, .5, -.5)))
    tire.append(Line3D(Point3D(-.5, 1, .5), Point3D(-.5, 1, -.5)))
    tire.append(Line3D(Point3D(.5, 1, .5), Point3D(.5, 1, -.5)))
    tire.append(Line3D(Point3D(1, .5, .5), Point3D(1, .5, -.5)))
    tire.append(Line3D(Point3D(1, -.5, .5), Point3D(1, -.5, -.5)))
    tire.append(Line3D(Point3D(.5, -1, .5), Point3D(.5, -1, -.5)))
    tire.append(Line3D(Point3D(-.5, -1, .5), Point3D(-.5, -1, -.5)))
    tire.append(Line3D(Point3D(-1, -.5, .5), Point3D(-1, -.5, -.5)))
    
    return tire


DISPLAY_HEIGHT = 512
DISPLAY_WIDTH = 512

# Global variables for camera position and properties

fov_x = 100
fov_y = 100
aspect_ratio = DISPLAY_WIDTH / DISPLAY_HEIGHT
near = 1.0
far = 1000.0

# Initial camera properties
camera_position = np.array([0.0, 0.0, 10.0])
camera_target = np.array([0.0, 0.0, 0.0])
camera_up_vector = np.array([0.0, 1.0, 0.0])

# Movement and rotation speeds
movement_speed = 0.25
rotation_speed = radians(2)


def buildProjectionMatrix():
    # Construct the view matrix
    forward = camera_target - camera_position
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, camera_up_vector)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    view_matrix = np.array([
    [right[0], up[0], -forward[0], np.dot(right, camera_position)],
    [right[1], up[1], -forward[1], np.dot(up, camera_position)], 
    [right[2], up[2], -forward[2], np.dot(forward, camera_position)],
    [0, 0, 0, 1] 
])

    f = 1 / tan(np.radians(fov_x) / 2)
    a = f / aspect_ratio
    b = (far + near) / (near - far)
    c = (2 * far * near) / (near - far)

    projection_matrix = np.array([
    [a, 0, 0, 0],
    [0, -f, 0, 0],
    [0, 0, b, c],
    [0, 0, -1, 0]
])
    
    transform_matrix = projection_matrix @ view_matrix

    return transform_matrix


def clipTest(pt1, pt2):
    def inFrustum(pt):
        # Homogeneous divide
        x = pt[0][0].item() / pt[3][0].item() 
        y = pt[1][0].item() / pt[3][0].item() 
        z = pt[2][0].item() / pt[3][0].item() 

        # Frustum clipping tests
        return (abs(x) <= 1 and abs(y) <= 1 and 0 <= z <= 1)

    # Both points need to be in the frustum for the line to render
    return inFrustum(pt1) and inFrustum(pt2) 

def toScreen(pt):
   # Homogeneous divide
   pt /= pt[3]  

   # Viewport transform
   x = DISPLAY_WIDTH/2 * pt[0][0].item() + DISPLAY_WIDTH/2
   y = DISPLAY_HEIGHT/2 * pt[1][0].item() + DISPLAY_HEIGHT/2
   

   return Point(int(x), int(y)) 

# Function to move the camera
def move(direction):
    global camera_position, camera_target, camera_up_vector, movement_speed
    forward = camera_target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera_up_vector)
    right /= np.linalg.norm(right)
    
    # For debugging, I had a hard time getting this to work correctly.
    print(f"Before move: camera_position = {camera_position}, camera_target = {camera_target}")
    
    if direction == 'forward':
        camera_position += forward * movement_speed
    elif direction == 'backward':
        camera_position -= forward * movement_speed
    elif direction == 'left':
        camera_position += right * movement_speed
    elif direction == 'right':
        camera_position -= right * movement_speed
    elif direction == 'up':
        camera_position -= camera_up_vector * movement_speed
    elif direction == 'down':
        camera_position += camera_up_vector * movement_speed
    elif direction == 'home':
        camera_position = np.array([0.0, 0.0, 10.0])
        camera_target = np.array([0.0, 0.0, 0.0])
        camera_up_vector = np.array([0.0, 1.0, 0.0])

# Function to rotate the camera
def rotate(direction):
    global camera_position, camera_target, camera_up_vector
    forward = camera_target - camera_position
    right = np.cross(forward, camera_up_vector)
    right /= np.linalg.norm(right)

    rotation_matrix = np.array([[cos(rotation_speed), 0, sin(rotation_speed)],
                                [0, 1, 0],
                                [-sin(rotation_speed), 0, cos(rotation_speed)]])

    if direction == 'left':
        
        forward = np.dot(np.linalg.inv(rotation_matrix), forward)
        camera_target = camera_position + forward
    elif direction == 'right':
        forward = np.dot(rotation_matrix, forward)
        camera_target = camera_position + forward

# Initialize the game engine
pygame.init()
 
# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

# Set the height and width of the screen
size = [DISPLAY_WIDTH, DISPLAY_HEIGHT]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Shape Drawing")
 
#Set needed variables
done = False
clock = pygame.time.Clock()
start = Point(0.0,0.0)
end = Point(0.0,0.0)
linelist = loadHouse()

#Loop until the user clicks the close button.
while not done:
 
    # This limits the while loop to a max of 100 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(100)

    # Clear the screen and set the screen background
    screen.fill(BLACK)

    #Controller Code#
    #####################################################################

    for event in pygame.event.get():
        if event.type == pygame.QUIT: # If user clicked close
            done=True
            
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_w]:
        move('forward')
    if pressed[pygame.K_s]:
        move('backward')
    if pressed[pygame.K_a]:
        move('left')
    if pressed[pygame.K_d]:
        move('right')
    if pressed[pygame.K_r]:
        move('up')
    if pressed[pygame.K_f]:
        move('down')
    if pressed[pygame.K_q]:
        rotate('left')
    if pressed[pygame.K_e]:
        rotate('right')
    if pressed[pygame.K_h]:
        move('home')

    #Viewer Code#
    #####################################################################

    project = buildProjectionMatrix()

    for s in linelist:
        
        pt1_w = np.matrix([[s.start.x],[s.start.y],[s.start.z],[1]])
        pt2_w = np.matrix([[s.end.x],[s.end.y],[s.end.z],[1]])
        
        pt1_c = project*pt1_w
        pt2_c = project*pt2_w
        
        if clipTest(pt1_c,pt2_c):
            pt1_s = toScreen(pt1_c)
            pt2_s = toScreen(pt2_c)
            pygame.draw.line(screen, BLUE, (pt1_s.x, pt1_s.y), (pt2_s.x, pt2_s.y))

    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()

# Be IDLE friendly
pygame.quit()
