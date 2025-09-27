import numpy as np
import matplotlib.pyplot as plt

# Parameters
parent = 5.0  # Parent gene value
sigma = 1.0   # Mutation strength (standard deviation)
low, high = 0, 10  # Gene bounds

# Generate values (explicit dtype for safety)
x = np.linspace(low, high, 500, dtype=np.float64)
pdf = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-((x-parent)**2)/(2*sigma**2))

# Truncate PDF outside bounds
pdf = np.where((x < low) | (x > high), 0, pdf)

# Plot
plt.plot(x, pdf, label=f'Parent = {parent}, σ = {sigma}')
plt.title('Gaussian Mutation Offspring Distribution')
plt.xlabel('Gene Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
