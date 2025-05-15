import pygame
import random
import math
import time
import threading
import sys
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
# simulation.py
from flask import Blueprint, render_template
app = Flask(__name__)

simulation_bp = Blueprint('simulation', __name__)

@simulation_bp.route('/simulation')
def show_simulation():
    return render_template('simulation.html')


# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

# Simulation configuration
simTime = 300       # Total simulation time in seconds (5 minutes by default)
timeElapsed = 0
signals = []
noOfSignals = 4
currentGreen = 0   # Indicates which signal is green
nextGreen = (currentGreen+1) % noOfSignals
currentYellow = 0   # Indicates whether yellow signal is on or off 

# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

# Count of vehicles at traffic signals
noOfCars = 0
noOfBikes = 0
noOfBuses = 0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

# Red signal time at which vehicles will be detected at a signal
detectionTime = 5

# Average speeds of different vehicle types
speeds = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'rickshaw': 2, 'bike': 2.5}

# Coordinates for vehicle spawn positions
x = {'right': [0, 0, 0], 'down': [755, 727, 697], 'left': [1400, 1400, 1400], 'up': [602, 627, 657]}    
y = {'right': [348, 370, 398], 'down': [0, 0, 0], 'left': [498, 466, 436], 'up': [800, 800, 800]}

# Vehicle data structure
vehicles = {
    'right': {0: [], 1: [], 2: [], 'crossed': 0}, 
    'down': {0: [], 1: [], 2: [], 'crossed': 0}, 
    'left': {0: [], 1: [], 2: [], 'crossed': 0}, 
    'up': {0: [], 1: [], 2: [], 'crossed': 0}
}

vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'rickshaw', 4: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Coordinates for UI elements
signalCoords = [(530, 230), (810, 230), (810, 570), (530, 570)]
signalTimerCoords = [(530, 210), (810, 210), (810, 550), (530, 550)]
vehicleCountCoords = [(480, 210), (880, 210), (880, 550), (480, 550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates for vehicle stop lines
stopLines = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
defaultStop = {'right': 580, 'down': 320, 'left': 810, 'up': 545}
stops = {'right': [580, 580, 580], 'down': [320, 320, 320], 'left': [810, 810, 810], 'up': [545, 545, 545]}

# Coordinates for vehicle turning
mid = {
    'right': {'x': 705, 'y': 445}, 
    'down': {'x': 695, 'y': 450}, 
    'left': {'x': 695, 'y': 425}, 
    'up': {'x': 695, 'y': 400}
}
rotationAngle = 3

# Gap between vehicles
gap = 15    # stopping gap
gap2 = 15   # moving gap

# Flag to control simulation
simulationActive = True

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        
        # Load vehicle image
        path = os.path.join("images", direction, f"{vehicleClass}.png")
        try:
            self.originalImage = pygame.image.load(path)
            self.currentImage = pygame.image.load(path)
        except pygame.error:
            # Create a colored rectangle if image is not found
            if vehicleClass == 'bike':
                size = (20, 10)
                color = (0, 0, 255)  # Blue
            elif vehicleClass == 'car':
                size = (30, 15)
                color = (255, 0, 0)  # Red
            elif vehicleClass == 'pickup' or vehicleClass == 'rickshaw':
                size = (35, 20)
                color = (0, 255, 0)  # Green
            elif vehicleClass == 'bus':
                size = (45, 20)
                color = (255, 255, 0)  # Yellow
            elif vehicleClass == 'truck':
                size = (50, 20)
                color = (255, 0, 255)  # Purple
            else:
                size = (30, 15)
                color = (100, 100, 100)  # Gray
                
            self.originalImage = pygame.Surface(size)
            self.originalImage.fill(color)
            self.currentImage = self.originalImage.copy()
    
        # Set vehicle position and stop coordinate
        if direction == 'right':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'left':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif direction == 'down':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif direction == 'up':
            if len(vehicles[direction][lane]) > 1 and vehicles[direction][lane][self.index-1].crossed == 0:
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
            
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))

    def move(self):
        if self.direction == 'right':
            if self.crossed == 0 and self.x + self.currentImage.get_rect().width > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x + self.currentImage.get_rect().width < mid[self.direction]['x']:
                    if (self.x + self.currentImage.get_rect().width <= self.stop or (currentGreen == 0 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2) or vehicles[self.direction][self.lane][self.index-1].turned == 1):                
                        self.x += self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2):
                            self.y += self.speed
            else: 
                if (self.x + self.currentImage.get_rect().width <= self.stop or self.crossed == 1 or (currentGreen == 0 and currentYellow == 0)) and (self.index == 0 or self.x + self.currentImage.get_rect().width < (vehicles[self.direction][self.lane][self.index-1].x - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                    self.x += self.speed

        elif self.direction == 'down':
            if self.crossed == 0 and self.y + self.currentImage.get_rect().height > stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y + self.currentImage.get_rect().height < mid[self.direction]['y']:
                    if (self.y + self.currentImage.get_rect().height <= self.stop or (currentGreen == 1 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2) or vehicles[self.direction][self.lane][self.index-1].turned == 1):                
                        self.y += self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.y < (vehicles[self.direction][self.lane][self.index-1].y - gap2):
                            self.x -= self.speed
            else: 
                if (self.y + self.currentImage.get_rect().height <= self.stop or self.crossed == 1 or (currentGreen == 1 and currentYellow == 0)) and (self.index == 0 or self.y + self.currentImage.get_rect().height < (vehicles[self.direction][self.lane][self.index-1].y - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                    self.y += self.speed
            
        elif self.direction == 'left':
            if self.crossed == 0 and self.x < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.x > mid[self.direction]['x']:
                    if (self.x >= self.stop or (currentGreen == 2 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index-1].turned == 1):                
                        self.x -= self.speed
                else: 
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or self.x > (vehicles[self.direction][self.lane][self.index-1].x + gap2):
                            self.y -= self.speed
            else: 
                if (self.x >= self.stop or self.crossed == 1 or (currentGreen == 2 and currentYellow == 0)) and (self.index == 0 or self.x > (vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                    self.x -= self.speed
                    
        elif self.direction == 'up':
            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if self.willTurn == 1:
                if self.crossed == 0 or self.y > mid[self.direction]['y']:
                    if (self.y >= self.stop or (currentGreen == 3 and currentYellow == 0) or self.crossed == 1) and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or vehicles[self.direction][self.lane][self.index-1].turned == 1):
                        self.y -= self.speed
                else:   
                    if self.turned == 0:
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if self.rotateAngle == 90:
                            self.turned = 1
                    else:
                        if self.index == 0 or self.x < (vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2) or self.y > (vehicles[self.direction][self.lane][self.index-1].y + gap2):
                            self.x += self.speed
            else: 
                if (self.y >= self.stop or self.crossed == 1 or (currentGreen == 3 and currentYellow == 0)) and (self.index == 0 or self.y > (vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned == 1)):                
                    self.y -= self.speed

# Initialize signals with default values
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()

# Set time according to traffic density
def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime
    
    # Count vehicles at the next intersection
    noOfCars, noOfBuses, noOfTrucks, noOfRickshaws, noOfBikes = 0, 0, 0, 0, 0
    for j in range(len(vehicles[directionNumbers[nextGreen]][0])):
        vehicle = vehicles[directionNumbers[nextGreen]][0][j]
        if vehicle.crossed == 0:
            vclass = vehicle.vehicleClass
            noOfBikes += 1
            
    for i in range(1, 3):
        for j in range(len(vehicles[directionNumbers[nextGreen]][i])):
            vehicle = vehicles[directionNumbers[nextGreen]][i][j]
            if vehicle.crossed == 0:
                vclass = vehicle.vehicleClass
                if vclass == 'car':
                    noOfCars += 1
                elif vclass == 'bus':
                    noOfBuses += 1
                elif vclass == 'truck':
                    noOfTrucks += 1
                elif vclass == 'rickshaw':
                    noOfRickshaws += 1
    
    # Calculate green time based on traffic density
    greenTime = math.ceil(((noOfCars * carTime) + 
                          (noOfRickshaws * rickshawTime) + 
                          (noOfBuses * busTime) + 
                          (noOfTrucks * truckTime) + 
                          (noOfBikes * bikeTime)) / (noOfLanes + 1))
    
    print('Green Time:', greenTime)
    if greenTime < defaultMinimum:
        greenTime = defaultMinimum
    elif greenTime > defaultMaximum:
        greenTime = defaultMaximum
        
    signals[(currentGreen + 1) % (noOfSignals)].green = greenTime
   
def repeat():
    global currentGreen, currentYellow, nextGreen, simulationActive
    
    if not simulationActive:
        return
        
    while signals[currentGreen].green > 0:   # while the timer of current green signal is not zero
        if not simulationActive:
            return
        printStatus()
        updateValues()
        if signals[(currentGreen + 1) % (noOfSignals)].red == detectionTime:    # set time of next green signal 
            thread = threading.Thread(name="detection", target=setTime, args=())
            thread.daemon = True
            thread.start()
        time.sleep(1)
        
    currentYellow = 1   # set yellow signal on
    vehicleCountTexts[currentGreen] = "0"
    
    # reset stop coordinates of lanes and vehicles 
    for i in range(0, 3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
            
    while signals[currentGreen].yellow > 0:  # while the timer of current yellow signal is not zero
        if not simulationActive:
            return
        printStatus()
        updateValues()
        time.sleep(1)
        
    currentYellow = 0   # set yellow signal off
    
    # reset all signal times of current signal to default times
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen # set next signal as green signal
    nextGreen = (currentGreen + 1) % noOfSignals    # set next green signal
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green    # set the red time of next to next signal
    
    repeat()

# Print the signal timers on cmd
def printStatus():                                                                                           
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                print(" GREEN TS", i+1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
            else:
                print("YELLOW TS", i+1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
        else:
            print("   RED TS", i+1, "-> r:", signals[i].red, " y:", signals[i].yellow, " g:", signals[i].green)
    print()

# Update values of the signal timers after every second
def updateValues():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
                signals[i].totalGreenTime += 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

# Generating vehicles in the simulation
def generateVehicles():
    global simulationActive
    
    while simulationActive:
        vehicle_type = random.randint(0, 4)
        if vehicle_type == 4:
            lane_number = 0
        else:
            lane_number = random.randint(0, 1) + 1
            
        will_turn = 0
        if lane_number == 2:
            temp = random.randint(0, 4)
            will_turn = 1 if temp <= 2 else 0
            
        temp = random.randint(0, 999)
        direction_number = 0
        a = [400, 800, 900, 1000]
        if temp < a[0]:
            direction_number = 0
        elif temp < a[1]:
            direction_number = 1
        elif temp < a[2]:
            direction_number = 2
        elif temp < a[3]:
            direction_number = 3
            
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, 
                directionNumbers[direction_number], will_turn)
        time.sleep(0.75)

def simulationTime():
    global timeElapsed, simTime, simulationActive
    
    while timeElapsed < simTime and simulationActive:
        timeElapsed += 1
        time.sleep(1)
        
    # When time is up, print statistics and end simulation
    simulationActive = False
    totalVehicles = 0
    
    print('\n' + '='*50)
    print('SIMULATION FINISHED')
    print('='*50)
    print('Lane-wise Vehicle Counts:')
    for i in range(noOfSignals):
        print('Lane', i+1, ':', vehicles[directionNumbers[i]]['crossed'])
        totalVehicles += vehicles[directionNumbers[i]]['crossed']
    
    print('\nTotal vehicles passed:', totalVehicles)
    print('Total time passed:', timeElapsed)
    print('Vehicles per second:', round(float(totalVehicles)/float(timeElapsed), 2))
    print('='*50)
    
    # Give time for the user to read the stats before exiting
    time.sleep(5)
    pygame.quit()
    sys.exit()

def main():
    global simTime
    
    # Parse command line arguments for simulation time
    if len(sys.argv) > 1:
        try:
            simTime = int(sys.argv[1])
            print(f"Simulation will run for {simTime} seconds")
        except ValueError:
            print("Invalid simulation time provided. Using default:", simTime)
    
    # Start timing thread
    timer_thread = threading.Thread(name="simulationTime", target=simulationTime, args=())
    timer_thread.daemon = True
    timer_thread.start()

    # Start signal initialization thread
    init_thread = threading.Thread(name="initialization", target=initialize, args=())
    init_thread.daemon = True
    init_thread.start()

    # Pygame setup
    screenWidth = 1400
    screenHeight = 800
    screenSize = (screenWidth, screenHeight)
    
    # Try to load background image
    try:
        background = pygame.image.load('images/mod_int.png')
    except pygame.error:
        background = pygame.Surface(screenSize)
        background.fill((200, 200, 200))  # Gray background as fallback
        # Draw intersection
        pygame.draw.rect(background, (100, 100, 100), (500, 200, 400, 400))  # Road area
        pygame.draw.rect(background, (50, 50, 50), (500, 350, 400, 100))  # Horizontal road
        pygame.draw.rect(background, (50, 50, 50), (650, 200, 100, 400))  # Vertical road
    
    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("TRAFFIC SIMULATION")

    # Load signal images and font
    try:
        redSignal = pygame.image.load('images/signals/red.png')
        yellowSignal = pygame.image.load('images/signals/yellow.png')
        greenSignal = pygame.image.load('images/signals/green.png')
    except pygame.error:
        # Create simple colored rectangles as fallbacks
        redSignal = pygame.Surface((30, 90))
        redSignal.fill((255, 0, 0))
        yellowSignal = pygame.Surface((30, 90))
        yellowSignal.fill((255, 255, 0))
        greenSignal = pygame.Surface((30, 90))
        greenSignal.fill((0, 255, 0))
    
    font = pygame.font.Font(None, 30)

    # Start vehicle generation thread
    vehicle_thread = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    vehicle_thread.daemon = True
    vehicle_thread.start()

    # Main simulation loop
    clock = pygame.time.Clock()
    
    while simulationActive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                simulationActive = False
                pygame.quit()
                sys.exit()

        screen.blit(background, (0, 0))  # Display background
        
        # Display signals and timers
        for i in range(0, noOfSignals):
            if i == currentGreen:
                if currentYellow == 1:
                    if signals[i].yellow == 0:
                        signals[i].signalText = "STOP"
                    else:
                        signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoords[i])
                else:
                    if signals[i].green == 0:
                        signals[i].signalText = "SLOW"
                    else:
                        signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoords[i])
            else:
                if signals[i].red <= 10:
                    if signals[i].red == 0:
                        signals[i].signalText = "GO"
                    else:
                        signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoords[i])
        
        # Display signal timers and vehicle counts
        for i in range(0, noOfSignals):
            signalText = font.render(str(signals[i].signalText), True, (255, 255, 255), (0, 0, 0))
            screen.blit(signalText, signalTimerCoords[i])
            
            displayText = vehicles[directionNumbers[i]]['crossed']
            vehicleCountText = font.render(str(displayText), True, (0, 0, 0), (255, 255, 255))
            screen.blit(vehicleCountText, vehicleCountCoords[i])

        # Display time elapsed and countdown
        timeElapsedText = font.render(f"Time Elapsed: {timeElapsed}/{simTime}", True, (0, 0, 0), (255, 255, 255))
        screen.blit(timeElapsedText, (1100, 50))
        
        timeRemainingText = font.render(f"Time Remaining: {max(0, simTime - timeElapsed)}", True, (0, 0, 0), (255, 255, 255))
        screen.blit(timeRemainingText, (1100, 80))

        # Move and render vehicles
        for vehicle in simulation:
            vehicle.move()
            screen.blit(vehicle.currentImage, (vehicle.x, vehicle.y))
            
        pygame.display.update()
        clock.tick(60)
        # simulation.py
# simulation.py

def run_simulation():
    # Example logic — replace with your real simulation
    return {"status": "success", "value": 42}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
       # Updated from ensure_columns_exist
    app.run(debug=True, threaded=True)