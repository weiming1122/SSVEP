# load modules
from psychopy import visual, event, data, core
from pylsl import StreamInfo, StreamOutlet
from pylsl import StreamInlet, resolve_byprop, local_clock
import numpy as np
import string
import time

freq_line = [[x+8.0 for x in range(8)], 
             [x+8.2 for x in range(8)], 
             [x+8.4 for x in range(8)], 
             [x+8.6 for x in range(8)],
             [x+8.8 for x in range(8)]]

freqs = [freq for line in freq_line for freq in line]

# create Marker stream
info_marker = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'MarkerID')
outlet_marker = StreamOutlet(info_marker) 

# find Predict stream   
print('looking for Predict stream...')
stream_predict = resolve_byprop('type', 'Predicts', timeout=10)
if stream_predict:
    inlet_predict = StreamInlet(stream_predict[0])
    print('find Predicts stream')
else:
    inlet_predict = False
    print("can't find Predicts stream")

# config window object
# width, hight = 800, 600
width, hight = 2560, 1440

white = [1, 1, 1]
black = [-1, -1, -1]
red = [1, -1, -1]

win = visual.Window(size=[width,hight], color=black, screen=0, units='pix', fullscr=False,
                    monitor='testmonitor', waitBlanking=False, allowGUI=True)
win.mouseVisible = False
        
win_size = np.array(win.size)
columns, rows = 8, 5

# config basic parameters of stimuli
n_elements = columns * rows                # number of the objects
stim_sizes = np.zeros((n_elements, 2))     # size array | unit: pix
stim_pos = np.zeros((n_elements, 2))       # position array
stim_opacities = np.ones((n_elements,))    # opacity array (default 1)
stim_contrs = np.ones((n_elements,))       # contrast array (default 1)

square_len = 100                           # side length of a single square | unit: pix
square_size = np.array([square_len, square_len])
stim_sizes[:] = square_size 

# divide the whole screen into rows*columns blocks, and pick the center of each block as the position
origin_pos = np.array([win_size[0]/columns, win_size[1]/(rows+1)]) / 2
if (origin_pos[0]<(square_len/2)) or (origin_pos[1]<(square_len/2)):
    raise Exception('too much blocks or too big the single square!')
    
for i in range(rows): 
    for j in range(columns):  
        stim_pos[i*columns+j] = origin_pos + [j,i+1]*origin_pos*2

stim_pos -= win_size/2     # center of the screen >>> the upper left corner
stim_pos[:,1] *= -1        # invert the y-axis

# config ssvep
refresh_rate = 60

display_time = 5.0    # keyboard display time before 1st stimulus
index_time = 1.0      # indicator display time
flash_time= 3.0
rest_time = 2.0       # rest-state time

# config sinusoidal
def calc_sinusoidal(freqs, refresh_rate, flash_time):
    time_point = np.arange(0, flash_time, 1/refresh_rate)
    sinws = []  
    for freq in freqs:
        sinw = np.sin(2*np.pi*freq*time_point)
        sinws.append(np.tile(sinw,(3,1)))
            
    sinws = np.asarray(sinws)
    
    sinws[np.round(sinws,5) > 0] = 1
    sinws[np.round(sinws,5) == 0] = 0
    sinws[np.round(sinws,5) < 0] = -1
            
    return sinws

# def calc_sinusoidal(freqs, refresh_rate, flash_time):
#     time_point = np.arange(0, flash_time, 1/refresh_rate)
#     all_binws = []
#     for freq in freqs:
#         binws = np.zeros((3, len(time_point)))
#         sinw = np.sin(2*np.pi*freq*time_point )
#         binw = np.where(sinw > 0, 1, -1) 
#         for i in range(len(time_point)):
#             if binw[i] == 1:
#                 binws[:,i] = [-1,1,-1]
#             else:
#                 binws[:,i] = [0.25,0.25,0.25]
                
#         all_binws.append(binws)
            
#     all_binws = np.asarray(all_binws)
    
            
#     return all_binws

stim_sinusoidal = calc_sinusoidal(freqs, refresh_rate, flash_time)

# config flashing elements
ssvep_stimuli = []
for i in range(int(flash_time*refresh_rate)):
    ssvep_stimuli.append(visual.ElementArrayStim(win=win, units='pix', nElements=n_elements,
                         sizes=stim_sizes, xys=stim_pos, colors=stim_sinusoidal[:,:,i], 
                         opacities=stim_opacities, contrs=stim_contrs, elementTex=None,
                         elementMask=None))
# config rect simuli
rect_stimuli = visual.ElementArrayStim(win=win, units='pix', nElements=n_elements,
                         sizes=stim_sizes, xys=stim_pos, colors= white, 
                         opacities=1, contrs=1, elementTex=None,
                         elementMask=None)
    
# config text simuli
symbols = ''.join([string.ascii_lowercase, '0123456789', ' ,.', '\u2190'])  # if you want more stimulus, just add more symbols
text_stimuli = []
for symbol, pos in zip(symbols, stim_pos):    
    text_stimuli.append(visual.TextStim(win=win, text=symbol, font='Arial', pos=pos, color=black,
                                        units='pix', height=square_len/2, bold=True))
     
# config output
output_pos = origin_pos - win_size/2
output_pos[1] *= -1

output = visual.TextStim(win=win, text='>> ', font='Arial', pos=output_pos, color=white,
                                units='pix', height=square_len/2, bold=True, alignHoriz = 'left',
                                wrapWidth = width-origin_pos[0]*2)

# config experiment parameters
ssvep_conditions = [{'id': i} for i in range(n_elements)]
ssvep_nrep = 1
trials = data.TrialHandler(ssvep_conditions, ssvep_nrep, method='random')

# config index rect
index_rect = visual.Rect(win=win, units="pix", pos=(0,0), size=square_size, fillColor=red, lineColor = red)

# start routine
# display speller interface
rect_stimuli.draw()
for text_stimulus in text_stimuli:
    text_stimulus.draw()
output.draw()
win.flip()
# Push marker to mark the start of the experiment
outlet_marker.push_sample(['111'], local_clock())
core.wait(display_time)

# begin to flash
for trial in trials:
    # initialise index position
    id = int(trial['id'])
    
    index_rect.pos = stim_pos[id,:]

    # Phase 1: index (eye shifting)
    rect_stimuli.draw()        
    index_rect.draw()
    for text_stimulus in text_stimuli:
        text_stimulus.draw()
    output.draw()
    win.flip()
    core.wait(index_time)
    
    # Phase 2: SSVEP flashing
    start_time = time.time()
    for i in range(int(flash_time*refresh_rate)):        
        ssvep_stimuli[i].draw()
        output.draw()
        win.flip()
        
        for keys in event.getKeys():
            if keys in ['escape', 'q']:
                # send command to end the experiment
                outlet_marker.push_sample(['999'], local_clock())
                win.close()
                core.quit()
            
    print('\ntime interval for flashing:', round((time.time()-start_time), 3)) 
    
    # send command to calc CCA
    outlet_marker.push_sample(['1'], local_clock())
    
    # receive predict    
    if inlet_predict:
        predict, timestamp = inlet_predict.pull_sample(timeout=2.0)
        if predict:    
            print('get predict:', predict[0])
            decision = int(predict[0])
            
            if symbols[decision] == '\u2190':
                output.text = output.text[:-1]
            else:
                output.text += symbols[decision]
                 
    # Phase 3: rest state
    rect_stimuli.draw()
    for text_stimulus in text_stimuli:
        text_stimulus.draw()
    output.draw()
    win.flip()
    core.wait(rest_time)

# send command to end the experiment        
outlet_marker.push_sample(['999'], local_clock()) 
win.close()
core.quit()

