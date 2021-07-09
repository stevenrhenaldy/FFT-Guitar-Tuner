import pyaudio
import struct
import numpy as np
import time
import matplotlib.pyplot as plt


def FFT(arr):
    # Cooley-Tukey Algorithm with recursive function
    # input using numpy array because it has native support of complex number
    length = len(arr)  # get the length of the array

    if length == 1:
        # return the array if the length is equal to one
        return arr
    else:
        # Split the array into even and odd array
        ARR_even = FFT(arr[::2])
        ARR_odd = FFT(arr[1::2])

        factor = np.exp(-2j * np.pi * np.arange(length) / length)

        # Put them back into one array
        ARR = np.concatenate(
            [ARR_even + factor[:int(length / 2)] * ARR_odd, ARR_even + factor[int(length / 2):] * ARR_odd])
        return ARR


def getNote(freq):
    # Get musical note from the nearest frequency
    if 80 < freq < 95:
        return 'low E'
    elif 95 < freq < 125:
        return 'A'
    elif 130 < freq < 157:
        return 'D'
    elif 170 < freq < 210:
        return 'G'
    elif 220 < freq < 260:
        return 'B'
    elif 310 < freq < 350:
        return 'high E'


# THE PROGRAM STARTS HERE
if __name__ == '__main__':
    # INITIALIZATION
    CHUNK = 4 * 1024            # Set Chunk size to 4068 to get the detail number
    FORMAT = pyaudio.paInt16
    CHANNEL = 1                 # Mic input channel
    RATE = 44100                # Audio sampling rate (Default: 44100Hz)

    p = pyaudio.PyAudio()

    # Audio stream using pyAudio
    stream = p.open(
        format=FORMAT,
        channels=CHANNEL,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK
    )

    # Create Plot using matplotlib
    fig, (ax, ax2) = plt.subplots(2)
    x = np.arange(0, 2 * CHUNK, 2)
    x_fft = np.linspace(0, RATE, CHUNK)

    line, = ax.plot(x, np.random.rand(CHUNK), '-', lw=2)
    line_fft, = ax2.semilogx(x_fft, np.random.rand(CHUNK), '-', lw=2)

    ax.set_title("AUDIO WAVEFORM")
    ax.set_xlabel('samples')
    ax.set_ylabel('volume')
    ax.set_ylim(-60000, 60000)
    ax.set_xlim(0, CHUNK)
    ax2.set_ylim(0, 20)
    fig.show()

    ax2.set_xlim(20, RATE / 10)
    ax2.axvline(x=82.407, color='gray', linestyle='--')
    ax2.text(82.407, 1, 'E')
    ax2.axvline(x=110, color='gray', linestyle='--')
    ax2.text(110, 1, 'A')
    ax2.axvline(x=146.832, color='gray', linestyle='--')
    ax2.text(146.832, 1, 'D')
    ax2.axvline(x=195.998, color='gray', linestyle='--')
    ax2.text(195.998, 1, 'G')
    ax2.axvline(x=256.942, color='gray', linestyle='--')
    ax2.text(256.942, 1, 'B')
    ax2.axvline(x=329.628, color='gray', linestyle='--')
    ax2.text(329.628, 1, 'E')

    label_freq = plt.xlabel('')

    label_note = plt.figtext(0.8, 0, 'TEST', fontsize=14)

    plt.show(block=False)

    # Performance enquiry
    frame_count = 0
    start_time = time.time()

    # Start the infinite loop for the program
    highest = 0
    while 1:
        data = stream.read(CHUNK)

        dataInt = struct.unpack(str(CHUNK) + 'h', data)

        line.set_ydata(dataInt)

        y_fft = FFT(dataInt)

        # FFT returns complex number, so we need to parse the complex number to real number.
        getHeight = np.abs(y_fft[0:CHUNK]) * 2 / (256 * CHUNK)

        # Plot the Frequency based data
        line_fft.set_ydata(getHeight)

        # Cut the array to get rid of high frequency noise
        # Low pass filter may be a better choice but this method also works well
        getHeight_cut = getHeight[:4000]

        # Take the index of highest amplitude from inside of the frequency array
        # We use this trick to get the most dominant frequency
        getAmplitude = getHeight_cut.argmax()

        # Times the index of the highest amplitude with sample rate / full array size
        Freq = getAmplitude * (RATE / CHUNK)

        # Set some threshold to get rid of low amplitude audio noise
        threshold = 5  # Change the frequency only if the amplitude is more than threshold
        threshold2 = 2  # Hide the note if the amplitude is less than threshold2

        if (getHeight_cut[getAmplitude] >= threshold):
            highest = Freq

        label_freq.set_text(f"Frequency: {highest}Hz")

        if (getHeight_cut[getAmplitude] >= threshold2):
            label_note.set_text(getNote(highest))
        else:
            label_note.set_text('')

        # Terminal logger for debugging purpose and to get FPS
        print(f"{Freq}Hz Note: {getNote(highest)}")

        try:
            # Draw every updates
            fig.canvas.draw()
            fig.canvas.flush_events()
            frame_count += 1

        except:
            # Exit if user click the X sign in matplotlib window
            frame_rate = frame_count / (time.time() - start_time)  # Get the frame rate
            print("Stream stopped")
            print(f"Average frame rate = {frame_rate}")
            break
