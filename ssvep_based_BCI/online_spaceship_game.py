import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
import numpy as np
import math
import random
import time as T
import pygame, sys
from pylsl import StreamInlet, resolve_byprop

# hide the simplegui control panel area
simplegui.Frame._hide_controlpanel = True

# WIDTH = 2560
# HEIGHT = 1440
WIDTH = 1960 
HEIGHT = 1040

# open a window on the screen
pygame.init()
screen = pygame.display.set_mode([WIDTH, HEIGHT])

# globals for user interface
ROCK_NUMBER = 3
score = 0
lives = 3
time = 0.5
started = False

sprite_lifespan = 100
sprite_speed_up = 0
asteroid_duration = 100

frame_rate = 60
frameN = 1
current_draw_time = T.time()
last_draw_time = T.time()

time_interval_rock = 2  # milliseconds
time_interval_predict = 20

# list_freqs = [240/28, 240/23, 240/21, 240/18]
list_freqs = [240/28, 240/23, 240/21]

sine_waves = {}
binary_vector = {} 
for freq in list_freqs:
    sine_waves[freq] = np.sin(2 * np.pi * freq * np.arange(0, asteroid_duration, 1/frame_rate))
    binary_vector[freq] = np.where(sine_waves[freq] > 0, 1, -1)

print("looking for Predict streams...")
streams_predict = resolve_byprop('type', 'Predicts', timeout=15)
if streams_predict:
    inlet_predict = StreamInlet(streams_predict[0])
    started = True
    print("Find Predicts stream.")   
else:
    inlet_predict = False
    print("Can\'t find Predicts stream.")

# get image and sound infomation
class ImageInfo:
    def __init__(self, center, size, radius = 0, lifespan = None, animated = False):
        self.center = center
        self.size = size
        self.radius = radius
        if lifespan:
            self.lifespan = lifespan
        else:
            self.lifespan = float('inf')
        self.animated = animated

    def get_center(self):
        return self.center

    def get_size(self):
        return self.size

    def get_radius(self):
        return self.radius

    def get_lifespan(self):
        return self.lifespan

    def get_animated(self):
        return self.animated

# debris images
debris_info = ImageInfo([320, 240], [640, 480])
debris_image = simplegui._load_local_image("assets/debris2_blue.png")

# nebula images
nebula_info = ImageInfo([960, 540], [1920, 1080])
# nebula_info = ImageInfo([400, 300], [800, 600])
nebula_image = simplegui._load_local_image("assets/nebula_1.jpg")

# splash image
splash_info = ImageInfo([200, 150], [400, 300])
splash_image = simplegui._load_local_image("assets/splash.png")

# ship image
ship_info = ImageInfo([45, 45], [90, 90], 35)
ship_image = simplegui._load_local_image("assets/double_ship.png")

# missile image
missile_info = ImageInfo([5, 5], [10, 10], 3, 300)
missile_image = simplegui._load_local_image("assets/shot2.png")

# asteroid images
asteroid_info = ImageInfo([45, 45], [90, 90], 40)
asteroid_image = [simplegui._load_local_image("assets/asteroid_green.png"), 
                  simplegui._load_local_image("assets/asteroid_gray.png")]
# asteroid_info = ImageInfo([150, 150], [300, 300], 40)
# asteroid_image = [simplegui._load_local_image("assets/random_textures_300_700.png"), 
#                   simplegui._load_local_image("assets/gray_background_300.png")]

# animated explosion
explosion_info = ImageInfo([64, 64], [128, 128], 17, 24, True)
explosion_image = simplegui._load_local_image("assets/explosion_alpha.png")

# sound assets
soundtrack = simplegui._load_local_sound("assets/soundtrack.mp3")
missile_sound = simplegui._load_local_sound("assets/missile.mp3")
ship_thrust_sound = simplegui._load_local_sound("assets/thrust.mp3")
explosion_sound = simplegui._load_local_sound("assets/explosion.mp3")
missile_sound.set_volume(.5)

# helper functions to handle transformations
def angle_to_vector(ang):
    return [math.cos(ang), math.sin(ang)]

# helper functions to calculate distance of two points
def dist(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

# handle Sprite group
def process_sprite_group(group, time, canvas):
    remove_objects = set()
    group_copy = group.copy()
    for object in group_copy:
        object.draw(time, canvas)
        # delete sprite if it reaches its lifespan
        if object.update():
            remove_objects.add(object)
    group.difference_update(remove_objects)

def group_collide(group, other_object):
    '''
    Checks for collisions between all elements of a group
    with another object (rocks and ship)
    '''
    global explosion_group
    remove_objects = set()
    for object in group:
        if object.collide(other_object):
            remove_objects.add(object)
            explosion_group.add(Sprite([(object.get_position()[0] + other_object.get_position()[0]) / 2, 
                                        (object.get_position()[1] + other_object.get_position()[1]) / 2], 
                                        [0, 0], 0, 0, explosion_image, explosion_info, explosion_sound))
    group.difference_update(remove_objects)
    return len(remove_objects)

# track number of missile-rock collision to add to score
def group_group_collide(rock_group, missile_group):
    '''
    Checks for collisions between two groups of objects
    (rocks and missiles)
    '''
    global explosion_group
    remove_objects = set()
    num_collision = 0
    for object in rock_group:
        n = group_collide(missile_group, object)
        if n > 0:
            remove_objects.add(object)
            num_collision += n
    rock_group.difference_update(remove_objects)
    return num_collision
        
# Ship class
class Ship:
    def __init__(self, pos, vel, angle, image, info):
        self.pos = [pos[0], pos[1]]
        self.vel = [vel[0], vel[1]]
        self.thrust = False
        self.angle = angle   # In radians, not degrees
        self.angle_vel = 0
        self.image = image
        self.image_center = info.get_center()
        self.image_size = info.get_size()
        self.radius = info.get_radius()
        
        self.brake = False
        self.forward_vel = 0
        
    def get_position(self):
        return self.pos
    
    def get_radius(self):
        return self.radius
    
    def draw(self,canvas):
        # if thrust is on use thrust image
        if self.thrust:
            # need to specify the center of the thrust image on the ship_image tilesheet
            ship_center = (self.image_center[0] + self.image_size[0], self.image_center[1])
            # draw_image(image, center_source, width_height_source, center_dest, width_height_dest, rotation=0)
            canvas.draw_image(self.image, ship_center, self.image_size, 
                              self.pos, [self.image_size[0]*1.5, self.image_size[1]*1.5], self.angle)
        else:
            # Center of non-thrust image is same as self.image_center
            canvas.draw_image(self.image, self.image_center, self.image_size,
                              self.pos, [self.image_size[0]*1.5, self.image_size[1]*1.5], self.angle)
            # canvas.draw_circle(self.pos, self.radius, 1, "White", "White")

    def update(self):
        # update ship position, ensuring ship wraps screen
        # self.pos[0] = (self.pos[0] + self.vel[0]) % WIDTH
        # self.pos[1] = (self.pos[1] + self.vel[1]) % HEIGHT
        self.pos[0] = self.pos[0] + self.vel[0]
        self.pos[1] = self.pos[1] + self.vel[1]
        
        # update angular position, angle_vel controls how fast the ship rotates
        self.angle += self.angle_vel             
                
        # accelerate forward if self.thrust
        # multipler is arbitrary, larger multipler means more acceleration
        step = [e * self.forward_vel for e in angle_to_vector(self.angle)]
        if self.thrust:  
            self.pos[0] += step[0]
            self.pos[1] += step[1]
        
        # Sticky edges
        if (self.pos[0] < 0):
            self.pos[0] = 0
        elif (self.pos[0] > WIDTH):
            self.pos[0] = WIDTH
        elif (self.pos[1] < 0):
            self.pos[1] = 0               
        elif (self.pos[1] > HEIGHT):
            self.pos[1] = HEIGHT
                    
        # if self.brake:
        #     self.vel[0] -= acc[0]
        #     self.vel[1] -= acc[1]
        
        # have the ship automatically slow down over time (numbers should be less than, but close to, 1)
        # self.vel[0] *= .98
        # self.vel[1] *= .98
    
    def thrusting(self):
        self.thrust = True
        self.brake = False
        self.forward_vel += 4
        ship_thrust_sound.rewind()
        ship_thrust_sound.play()
            
    def go_back(self):
        self.thrust = True
        self.brake = False
        self.forward_vel -= 4
        ship_thrust_sound.rewind()
        ship_thrust_sound.play()
    
    def braking(self):
        self.brake = True
        self.thrust = False
        self.forward_vel = 0
        ship_thrust_sound.pause()
    
    # set angular velocity to 0.05 if right is down
    def increment_angle_vel(self):
        self.angle_vel += .05
    
    # set angular velocity to -0.05 if left is down
    def decrement_angle_vel(self):
        self.angle_vel -= .05
    
    def left(self):
        # set velocity to -2 if left is down
        self.vel[0] -= 2
    
    def right(self):
        # Set velocity to 2 if right is down
        self.vel[0] += 2
                    
    def shoot(self):
        global missile_group
        # set the starting position of the missile to the tip of the ship
        forward = angle_to_vector(self.angle)
        
        # missile_pos = [self.pos[0] + self.radius * forward[0], 
        #                self.pos[1] + self.radius * forward[1]]
        missile_pos = [self.pos[0] + (self.image_size[0] / 2) * forward[0],
                       self.pos[1] + (self.image_size[1] / 2) * forward[1]]
        
        # set the velocity of the missile as the velocity of the ship plus some forward missile velocity
        # missile_vel = [self.vel[0] + 6 * forward[0], self.vel[1] + 6 * forward[1]]        
        missile_vel = [0, self.vel[1] + 10 * forward[1]]
        missile_ang_vel = 0
        
        missile_group.add(Sprite(missile_pos, missile_vel, self.angle, missile_ang_vel, 
                                 missile_image, missile_info, missile_sound))
       
# Sprite class
class Sprite:
    def __init__(self, pos, vel, ang, ang_vel, image, info, sound = None, flip = False, freq = list_freqs[0]):
        self.pos = [pos[0],pos[1]]
        self.vel = [vel[0],vel[1]]
        self.angle = ang
        self.angle_vel = ang_vel
        self.image = image
        self.image_center = info.get_center()
        self.image_size = info.get_size()
        self.radius = info.get_radius()
        # measure of how long sprite 'lives' before automatic removal
        self.lifespan = info.get_lifespan()
        self.animated = info.get_animated()
        self.age = 0
        
        self.flip = flip
        self.freq = freq
        self.image_i = 0
        self.current_time = T.time()
        self.last_flip_time = T.time()
        self.flip_interval = 1/2/frame_rate
        self.frame_list = binary_vector[self.freq]

        if sound:
            sound.rewind()
            sound.play()
        
    def get_freq(self):
        return self.freq

    def get_position(self):
        return self.pos
    
    def get_radius(self):
        return self.radius
        
    def draw(self, time, canvas):
        if self.flip:
            # canvas.draw_image(self.image[self.image_i], self.image_center, self.image_size,
            #                 self.pos, self.image_size, self.angle) 
            canvas.draw_image(self.image[self.image_i], self.image_center, self.image_size,
                            self.pos, [self.image_size[0]*1.5, self.image_size[1]*1.5], self.angle) 
        elif self.animated:
            index = (time % self.lifespan) // 1 
            current_center = [self.image_center[0] + index * self.image_size[0], self.image_center[1]]
            canvas.draw_image(self.image, current_center, self.image_size, 
                              self.pos, [self.image_size[0]*1.5, self.image_size[1]*1.5], self.angle)
        else:
            canvas.draw_image(self.image, self.image_center, self.image_size,
                              self.pos, self.image_size, self.angle)

    def update(self):
        global frameN
        
        # update position
        # self.pos[0] = (self.pos[0] + self.vel[0]) % WIDTH
        # self.pos[1] = (self.pos[1] + self.vel[1]) % HEIGHT
        self.pos[0] = self.pos[0] + self.vel[0]
        self.pos[1] = self.pos[1] + self.vel[1]
        
        # update angle
        self.angle += self.angle_vel   
        
        # update age of Sprite
        self.age += 1
        
        # update picture
        if self.flip:
            self.current_time = T.time()
            if self.current_time - self.last_flip_time > self.flip_interval:
                if self.frame_list[frameN % (frame_rate * asteroid_duration)] == 1:
                    self.image_i = 0
                elif self.frame_list[frameN % (frame_rate * asteroid_duration)] == -1:
                    self.image_i = 1
                    
                # print('freq %d Hz flips %d th at %.5f' %(self.freq, frameN, self.current_time))
                # print('time interval is %.5f\n' %(self.current_time- self.last_flip_time))
                
                self.last_flip_time = self.current_time

        if self.pos[1] > HEIGHT:
            return True
        
        if self.age < self.lifespan:
            return False
        else:
            return True
        
    def collide(self, other_object):
        if dist(self.pos, other_object.get_position()) <= self.radius + other_object.get_radius():
            return True
        else:
            return False

# key handlers to control ship   
def keydown(key):
    if key == simplegui.KEY_MAP['up']:
        # my_ship.set_thrust(True)
        my_ship.thrusting()        
    elif key == simplegui.KEY_MAP['left']:
        # my_ship.decrement_angle_vel()
        my_ship.left()        
    elif key == simplegui.KEY_MAP['right']:
        # my_ship.increment_angle_vel()
        my_ship.right()   
    elif key == simplegui.KEY_MAP['down']:
        # my_ship.set_brake(True)
        my_ship.go_back()
    elif key == simplegui.KEY_MAP['space']:
        my_ship.shoot()
        
def keyup(key):
    if key == simplegui.KEY_MAP['up']:
        # my_ship.set_thrust(False)
        my_ship.braking()
    elif key == simplegui.KEY_MAP['left']:
        # my_ship.increment_angle_vel()
        my_ship.right()
    elif key == simplegui.KEY_MAP['right']:
        # my_ship.decrement_angle_vel()
        my_ship.left()   
    elif key == simplegui.KEY_MAP['down']:
        # my_ship.set_brake(False)
        my_ship.braking()
    elif key == 27:  # Escape
        timer1.stop()
        timer2.stop()
        frame.stop()
        # pygame.display.quit()
        pygame.quit()
        sys.exit()
        # exit()
                   
# start the game if user clicks in certain play area
def click(pos):
    global started, lives, score
    center = [WIDTH / 2, HEIGHT / 2]
    size = splash_info.get_size()
    inwidth = (center[0] - size[0] / 2) < pos[0] < (center[0] + size[0] / 2)
    inheight = (center[1] - size[1] / 2) < pos[1] < (center[1] + size[1] / 2)
    if (not started) and inwidth and inheight:
        started = True
        lives = 3
        score = 0
        soundtrack.rewind()
        soundtrack.play()

def draw(canvas):
    global time, started, lives, score, sprite_speed_up, frameN, last_draw_time, current_draw_time
    
    current_draw_time = T.time()
    
    # print('flips %d th at %.5f' %(frameN, current_draw_time))   
    # print('time interval is %.5f\n' %(current_draw_time - last_draw_time))
    
    last_draw_time = current_draw_time
    
    # animate background - nebula
    canvas.draw_image(nebula_image, nebula_info.get_center(), nebula_info.get_size(), 
                      [WIDTH / 2, HEIGHT / 2], [WIDTH, HEIGHT])
    
    time += 1
    
    # animate background - debris   
    # center = debris_info.get_center()
    # size = debris_info.get_size()
    # # wtime = (time / 4) % WIDTH
    # wtime = (time / 8) % center[0]
    
    # canvas.draw_image(debris_image, center, size, 
    #                   (wtime - WIDTH / 2, HEIGHT / 2), (WIDTH, HEIGHT))
    # canvas.draw_image(debris_image, center, size, 
    #                   (wtime + WIDTH / 2, HEIGHT / 2), (WIDTH, HEIGHT))
    
    # canvas.draw_image(debris_image, [center[0] - wtime, center[1]], [size[0] - 2 * wtime, size[1]], 
    #                   [WIDTH / 2 + 1.25 * wtime, HEIGHT / 2], [WIDTH - 2.5 * wtime, HEIGHT])
    # canvas.draw_image(debris_image, [size[0] - wtime, center[1]], [2 * wtime, size[1]], 
    #                   [1.25 * wtime, HEIGHT / 2], [2.5 * wtime, HEIGHT])

    # draw lives and score
    canvas.draw_text("Lives", [50, 50], 22, "White")
    canvas.draw_text("Score", [WIDTH-100, 50], 22, "White")
    canvas.draw_text(str(lives), [50, 80], 22, "White")
    canvas.draw_text(str(score), [WIDTH-100, 80], 22, "White")

    # draw ship
    my_ship.draw(canvas)
    
    # update rocks
    process_sprite_group(rock_group, time, canvas)
    
    # update missiles
    process_sprite_group(missile_group, time, canvas)   
    
    # update explosions
    process_sprite_group(explosion_group, time, canvas)
    
    # update ship
    my_ship.update()

    # draw splash screen if not started
    if not started:
        canvas.draw_image(splash_image, splash_info.get_center(), splash_info.get_size(), 
                          [WIDTH / 2, HEIGHT / 2], splash_info.get_size())
    
    # check for collisions
    lives -= group_collide(rock_group, my_ship)
    score += group_group_collide(rock_group, missile_group)
    
    if lives <= 0:
        started = False
        rock_group.difference_update(rock_group)
        sprite_speed_up = 0
        soundtrack.rewind()
        
    frameN += 1
    
# timer handler that spawns a rock    
def rock_spawner():
    global rock_group, sprite_speed_up
    if started:
        # rock_pos = [random.randrange(0, WIDTH), random.randrange(0, HEIGHT)]
        # rock_vel = [random.random() * .6 - .3, random.random() * .6 - .3]
        # rock_vel = [random.random() - .5 , random.random() - .5]
        # rock_avel = random.random() * .2 - .1
        rock_pos_all = {}
        step = WIDTH // len(list_freqs)
        for i, f in enumerate(list_freqs):
            rock_pos_all[f] = [random.randrange(i*step+step//2-100, i*step+step//2+100), random.randrange(400, 600)]
        #  = {list_freqs[0]:[random.randrange(400, 600), random.randrange(400, 600)],
        #                 list_freqs[1]:[random.randrange(WIDTH/2-100, WIDTH/2+100), random.randrange(400, 600)],
        #                 list_freqs[2]:[random.randrange(WIDTH-600, WIDTH-400), random.randrange(400, 600)]}  

        # rock_pos_all = {list_freqs[0]:[WIDTH/2 - 690, HEIGHT/2],
        #                 list_freqs[1]:[WIDTH/2, HEIGHT/2],
        #                 list_freqs[2]:[WIDTH/2 + 690, HEIGHT/2]} 
        # if dist(rock_pos01, my_ship.get_position()) > 200 and dist(rock_pos02, my_ship.get_position()) > 200 and dist(rock_pos03, my_ship.get_position()) > 200:            
        # rock_vel = [0, random.random()]
        rock_vel = [0, 0]
        rock_avel = 0
        
        # # the speed of rock increases gradually
        # if rock_vel[0] >= 0:
        #     rock_vel[0] += sprite_speed_up
        # else:
        #     rock_vel[0] -= sprite_speed_up
            
        # if rock_vel[1] >= 0:
        #     rock_vel[1] += sprite_speed_up
        # else:
        #     rock_vel[1] -= sprite_speed_up
                        
        while len(rock_group) < ROCK_NUMBER:
            rock_freqs = []
            for rock in rock_group:
                rock_freqs.append(rock.get_freq())
            temp_list_freqs = list_freqs.copy()
            temp_list_freqs = list(set(temp_list_freqs).difference(set(rock_freqs)))
            random.shuffle(temp_list_freqs)

            for f in temp_list_freqs:
                rock_group.add(Sprite(rock_pos_all[f], rock_vel, 3.14/2, rock_avel, 
                               asteroid_image, asteroid_info, flip=True, freq=f))
                if len(rock_group) == ROCK_NUMBER:
                    break
            
            # sprite_speed_up += 0.05

def do_action():    
    global rock_group, started
    try:            
        if inlet_predict:
            predict, predict_timestamp = inlet_predict.pull_sample()
            if predict_timestamp:
                print("got Predict %s at time %.3f" % (predict[0], predict_timestamp))
                action = float(predict[0])
                          
    except EOFError:
        print("got an error.")
           
    for rock in rock_group:
        print(action, np.round(rock.get_freq(),3))
        if np.round(rock.get_freq(),3) == action:
            pos = rock.get_position()[0]
            my_ship.pos[0] = pos
            my_ship.shoot()
            print('Shoot successfully.\n')     
            
    action = -1       
                        
# initialize frame
frame = simplegui.create_frame("Control Spaceship", WIDTH, HEIGHT)
frame.set_canvas_background("grey")

# initialize ship and three sprites
# my_ship = Ship([WIDTH / 2, HEIGHT / 2], [0, 0], 0, ship_image, ship_info)
my_ship = Ship([WIDTH / 2, HEIGHT - 40], [0, 0], -3.14 / 2, ship_image, ship_info)
rock_group = set([])
missile_group = set([])
explosion_group = set([])

# register handlers
frame.set_draw_handler(draw)
frame.set_keyup_handler(keyup)
frame.set_keydown_handler(keydown)
frame.set_mouseclick_handler(click)

timer1 = simplegui.create_timer(time_interval_rock, rock_spawner)
timer2 = simplegui.create_timer(time_interval_predict, do_action)

# get things rolling
timer1.start()
timer2.start()
frame.start()

    