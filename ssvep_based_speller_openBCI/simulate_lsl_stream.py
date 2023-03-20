"""
Simulate multi-channel EEG streams, and send these signals through LSL.
"""

from pylsl import StreamInfo, StreamOutlet, local_clock
import random as rnd
import time

n_channel = 8
sample_rate = 250
# channel_names = ['O1', 'PO3', 'Pz', 'Oz', 'unkown','POz', 'PO4', 'O2']   # dry electrode system
# channel_names = ['O1', 'O2', 'Pz', 'P3', 'P4', 'unkown1','unkown2', 'unkown3']  # wet electrode system
channel_names = ['O1', 'O2', 'unkown1','unkown2', 'Oz', 'PO3', 'PO4', 'unkown3'] # new device 

def main():
    
    # Simulation of OpenBCI streaming using 8 channels and 250 Hz as sample rate
    info = StreamInfo('OpenBCI', 'EEG', n_channel, sample_rate, 'float32', 'myuid34234')
    
    info.desc().append_child_value("manufacturer", "LSLTestAmp")
    eeg_channels = info.desc().append_child("channels")
    
    for c in channel_names:
        eeg_channels.append_child("channel") \
                    .append_child_value("label", c) \
                    .append_child_value("unit", "microvolts") \
                    .append_child_value("type", "EEG")
                
    outlet = StreamOutlet(info)

    input('Start recording via Lab Recorder and press enter...')
    print('Streaming EEG data...')

    while True:
        
        # Randomize some EEG sample
        eeg_sample = [rnd.random() for i in range(n_channel)]

        # Now send it and wait for a bit
        outlet.push_sample(eeg_sample, local_clock())
        time.sleep(1 / sample_rate)

if __name__ == '__main__':
    
    main()
