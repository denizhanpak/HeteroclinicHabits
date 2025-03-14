import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

# Parameters
dt = 0.01  # Time step
T = 100    # Total time
N = int(T / dt)  # Number of time steps
gamma = 0.5  # Friction coefficient
alpha = 1    # Pink noise exponent (1/f noise)

# Generate Pink Noise in Frequency Domain
freqs = fftfreq(N, d=dt)  # Frequency axis
white_noise = np.random.normal(0, 1, N)  # White noise
spectrum = 1 / (np.abs(freqs) + 1e-6)**(alpha/2)  # Pink noise scaling
spectrum[0] = 0  # Remove DC component
pink_noise = np.real(ifft(fft(white_noise) * spectrum))  # Inverse FFT to time domain

# Define potential and force
def potential(x):
    return (x**4) / 4 - (x**2) / 2  # Double-well potential

def force(x):
    return -x**3 + x  # -dV/dx

# Simulate Stochastic Differential Equation
x = np.zeros(N)
x[0] = -1  # Start in one well

for i in range(1, N):
    x[i] = x[i-1] + force(x[i-1]) * dt + gamma * pink_noise[i] * np.sqrt(dt)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, T, N), x, label="Particle Trajectory")
plt.xlabel("Time")
plt.ylabel("Position x")
plt.title("Stochastic Dynamics in a Double-Well Potential with Pink Noise")
plt.legend()
plt.show()
