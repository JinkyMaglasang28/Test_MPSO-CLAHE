import cv2
from pso_clahe import *
from trial import *

# Load the input image.
image = cv2.imread('/gray.jpg', cv2.IMREAD_GRAYSCALE)

# Create a PSO optimizer with 10 particles and 50 iterations.
pso = PSO(image, num_particles=10, max_iter=50)

# Run the optimizer.
enhanced = pso.optimize()


# Display the enhanced image.
cv2.imshow('Enhanced Image', enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
