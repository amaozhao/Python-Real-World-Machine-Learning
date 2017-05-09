import matplotlib.pyplot as plt
from scipy.io import wavfile
from features import mfcc, logfbank

# Read input sound file
sampling_freq, audio = wavfile.read("input_freq.wav")

# Extract MFCC and Filter bank features
mfcc_features = mfcc(audio, sampling_freq)
filterbank_features = logfbank(audio, sampling_freq)

# Print parameters
print ('MFCC: Number of windows =', mfcc_features.shape[0])
print ('Length of each feature =', mfcc_features.shape[1])
print ('Filter bank: Number of windows =', filterbank_features.shape[0])
print ('Length of each feature =', filterbank_features.shape[1])

# Plot the features
mfcc_features = mfcc_features.T
plt.matshow(mfcc_features)
plt.title('MFCC')

filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')

plt.show()
