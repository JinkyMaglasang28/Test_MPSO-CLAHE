import cv2
import numpy as np

class PSO:
    def __init__(self, image, num_particles=10, max_iter=50, c1=2.0, c2=2.0, w=0.7):
        """
        Initializes the PSO algorithm.

        :param image: Input image (in grayscale).
        :param num_particles: Number of particles in the swarm (default=10).
        :param max_iter: Maximum number of iterations (default=50).
        :param c1: Cognitive coefficient (default=2.0).
        :param c2: Social coefficient (default=2.0).
        :param w: Inertia weight (default=0.7).
        """
        self.image = image
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w

        # Initialize the swarm.
        self.particles = np.random.rand(num_particles, 2)
        self.velocities = np.zeros((num_particles, 2))
        self.best_positions = self.particles.copy()
        self.best_fitness = np.zeros((num_particles, 1))

        for i in range(num_particles):
            # Evaluate the fitness of each particle.
            self.best_fitness[i] = self.fitness(self.particles[i])

    def fitness(self, x):
        """
        Computes the fitness value for a given particle position.

        :param x: Particle position.
        :return: Fitness value.
        """
        # Extract the CLAHE parameters from the particle position.
        clip_limit = int(x[0])
        tile_size = (int(x[1]), int(x[1]))
        
        try:
            # Apply CLAHE to the input image with the specified parameters.
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            enhanced = clahe.apply(self.image)

            # Compute the entropy of the enhanced image.
            entropy = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
            entropy = entropy[entropy != 0]
            entropy /= np.sum(entropy)
            entropy = -np.sum(entropy * np.log2(entropy))

            # The fitness value is the negative entropy.
            return -entropy
        except cv2.error as e:
            print("OpenCV error:", e)

    def optimize(self):
        """
        Runs the PSO algorithm for the specified number of iterations.
        """
        for i in range(self.max_iter):
            # Update the velocity and position of each particle.
            r1 = np.random.rand(self.num_particles, 2)
            r2 = np.random.rand(self.num_particles, 2)
            self.velocities = self.w * self.velocities + \
                              self.c1 * r1 * (self.best_positions - self.particles) + \
                              self.c2 * r2 * (np.tile(np.max(self.best_positions, axis=0), (self.num_particles, 1)) - self.particles)
            self.particles = np.clip(self.particles + self.velocities, 0, 255)

            # Evaluate the fitness of each particle and update the best positions.
            for j in range(self.num_particles):
                fitness = self.fitness(self.particles[j])
                if fitness is not None and fitness > self.best_fitness[j]:
                    self.best_positions[j] = self.particles[j]
                    self.best_fitness[j] = fitness

        # Return the best particle as the enhanced image.
        best_index = np.argmax(self.best_fitness)
        best_particle = self.best_positions[best_index]
                # Extract the CLAHE parameters from the best particle.
        clip_limit = int(best_particle[0])
        tile_size = (int(best_particle[1]), int(best_particle[1]))

        # Apply CLAHE to the input image with the best parameters.
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced = clahe.apply(self.image)

        return enhanced

