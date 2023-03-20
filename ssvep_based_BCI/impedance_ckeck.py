# Part 1 (Record)
# EEG is recorded using OpenBCI Cyton Board and streamed through LSL.

# need to install pyOpenBCI using: pip install pyOpenBCI
# import necessary libralies
from pyOpenBCI import OpenBCICyton
from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import threading
#import math

channels = 8
sample_rate = 250
display_interval = 0.2

series_resistor_ohms = 4990     # Ohms. There is a series resistor on the 32 bit board.
leadOffDrive_amps = 6.0e-9      # 6 nA, set by its Arduino code

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1)      #uV/count
channel_names = ['O1', 'PO3', 'Pz', 'Oz', 'unkown','POz', 'PO4', 'O2']
#channel_names = ['O1', 'O2', 'P3', 'P4', 'unkown1','unkown2', 'unkown3', 'unkown4']

print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: OpenBCIEEGtest\n")      
# info_eeg = StreamInfo(name='OpenBCIEEG', type='EEG', channel_count=8, 
#                   nominal_srate=250, channel_format='float32', source_id='OpenBCIEEGtest')
info_eeg = StreamInfo('OpenBCIEEG', 'EEG', channels, sample_rate, 'float32', 'OpenBCIEEGtest')

info_eeg.desc().append_child_value("manufacturer", "LSLTestAmp")

eeg_channels = info_eeg.desc().append_child("channels")

for c in channel_names:
    eeg_channels.append_child("channel") \
                .append_child_value("label", c) \
                .append_child_value("unit", "microvolts") \
                .append_child_value("type", "EEG")
                
# make an outlet
outlet_eeg = StreamOutlet(info_eeg)
#impedance = []
dataRaw = []
dataStd = []
impedence = [0 for i in range(channels)]

EEG_raw = []
maxlength = 5 * sample_rate
for i in range(channels):
    EEG_raw.append(deque([], maxlen=maxlength))

start_time = time.time()

def lsl_streamers(sample):
    global start_time, impedence
    # send EEG data and wait for a bit
    outlet_eeg.push_sample(np.array(sample.channels_data)*SCALE_FACTOR_EEG, local_clock())

    for i in range(channels):
        EEG_raw[i].append(sample.channels_data[i]*SCALE_FACTOR_EEG)

    current_time = time.time()
    if current_time - start_time > display_interval and len(EEG_raw[0]) >= sample_rate:
        dataStd = np.std(np.array([list(d)[-250:] for d in EEG_raw]), axis=1)
        ims = (np.sqrt(2.0) * dataStd * 1.0e-6 / leadOffDrive_amps -  series_resistor_ohms) / 1000
        print(ims)
        for i in range(channels):
            impedence[i] = np.round(ims[i], 2)
        start_time = time.time()

patch_color = ['green', 'orange', 'red']
green_threshold = 100
red_threshold = 200
color_radius = 15
color_pos = [(200, 450), (275, 460), (350, 450), (190, 390), (275, 390), (360, 390), (275, 300)]
text_pos = [(190, 455), (265, 465), (340, 455), (175, 395), (260, 395), (345, 395), (265, 305)]
text_label = ['O1', 'Oz', 'O2', 'PO3', 'POz', 'PO4', 'Pz']
text_imp_pos = [(180, 480), (260, 490), (335, 480), (170, 370), (255, 370), (340, 370), (260, 280)]
# plot
plt.rcParams["figure.figsize"] = (9,6)
fig, ax = plt.subplots()

def plot_impedence():
    while True:
        try:
            text_imp_label = [k for i,k in enumerate(impedence) if i != 4]
            text_imp_label = [k if k > 0 else 0 for k in text_imp_label]

            with Image.open('assets/head.png') as image_file:
                ax.imshow(image_file)
            for i, ims in enumerate(text_imp_label):
                if ims > 0 and ims <= green_threshold:
                    c = 'green'
                elif ims >= red_threshold or ims == 0:
                    c = 'red'
                else:
                    c = 'orange'
                ax.add_patch(patches.Circle(color_pos[i], radius=color_radius, color=c))

            # text
            for i, p in enumerate(text_pos):
                ax.text(p[0], p[1], text_label[i])

            for i, p in enumerate(text_imp_pos):
                ax.text(p[0], p[1], str(text_imp_label[i]))

            ax.axis('off')
            plt.draw()
            plt.pause(0.1)
            ax.clear()

        except KeyboardInterrupt:
            plt.close()
            break


# set (daisy = True) when stream 16 channels    
# board = OpenBCICyton(port='COM6', daisy=False)
board = OpenBCICyton(port='COM4', daisy=False)
board.write_command("z101Z")
board.write_command("z201Z")
board.write_command("z301Z")
board.write_command("z401Z")
board.write_command("z501Z")
board.write_command("z601Z")
board.write_command("z701Z")
board.write_command("z801Z")
# threading.Thread(target=plot_impedence).start()
threading.Thread(target=board.start_stream, args=(lsl_streamers,)).start()
plot_impedence()