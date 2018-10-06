from numpy import array, random, argmin
import copy

class Pso():
	def __init__(self, 
		num_particles=100, 
		interval=(0, 1), 
		dimension=2):

		self.num_particles = num_particles
		self.range_min, self.range_max = interval
		self.dimension = dimension
		self.goal = None

		self.renew(
			num_particles=num_particles, 
			interval=interval, 
			dimension=dimension)

	def renew(self, 
		num_particles=100, 
		interval=(0, 1), 
		dimension=2):
		# Initing each particle position in a n-dimensional space
		# using a uniform distribution in range [range_min, range_max)
		self.position = array([random.random(size=num_particles) * \
			(self.range_max - self.range_min) + self.range_min
			for i in range(dimension)])

		# Particle velocity, should be initialized in the range
		# [- |range_max - range_min|, |range_max - range_min|),
		# randomly following a uniform distribution. Note that this
		# is a VECTORIAL velocity.
		self.velocity = array([random.random(size=num_particles) * \
			(2 * (self.range_max - self.range_min)) - \
			(self.range_max - self.range_min)
			for i in range(dimension)])

		# The best position of each particle is initialized as
		# its initial position
		self.particle_best_position = copy.deepcopy(self.position)

	def run(self, 
		goal=(0.5, 0.5), 
		iteration_max=1000, 
		omega=0.5, 
		phi_p=0.7, 
		phi_g=0.3,
		minimization=True,
		print_progress=True,
		min_variation_accepted=1.0e-10,
		it_until_break=20):

		"""
			Attributes documentation:
			goal = a n-dimensional tuple representing the optimization goal
			iteration_max = maximum number of iterations of Particle Swarm Algorithm
			omega = multiplicative factor to keep current particle speed (in other words,
				is the percentage of current speed retainined for each particle)
			phi_p = multiplicative factor of each particle's personal best answer to
				consider in each iteration
			phi_g = same as above, but considering the global best answer instead
			minimization = if enabled, then it's a minimizing problem (follows the
				opposite side of the "cost function gradient" in theory, but that
				function does not need to be differentiable at all.
			print_progress = print progress in each 10% of total iterations count
			min_variation_accepted = minimal difference (calculated by infinite norm)
				between each iteration best answer and the global best solution
				accepted. If the variation is less than this number, a counter
				will be decremented. When this counter reaches 0, the algorithm
				assumes convergence and returns. This counter is initialized by
				the parameter...
			in_until_break = initial value of the "probably reached convergence counter".
			
		"""

		# Phi factor normalization
		if phi_p + phi_g > 1.0:
			phi_p, phi_g = phi_p / (phi_p + phi_g), phi_g / (phi_p + phi_g)

		# Check if goal setup must be updated for another run,
		# supposing that the previous one exists
		if not self.goal or len(goal) < len(self.goal) or any(goal != self.goal):
			self.goal = array(goal)

			# Number of dimensions on goal array must match the number of
			# dimensions of the particle space
			if len(self.goal) < self.dimension:
				self.goal += tuple(0.0 \
					for _ in range(len(self.goal) - dimensions))

			elif len(self.goal) > self.dimension:
				print("Warning: picking up a", 
					len(self.dimension) + "-dimensional goal projection")
				self.goal = tuple(self.goal[:self.dimension])

			# Calculate the euclidean distance of each particle from
			# the goal
			self.particle_cost = self.__costfunction__(minimization=minimization)

			# Pick up the global answer
			self.global_best_particle = argmin(self.particle_cost)
			self.global_best_position = self.position[:, \
				self.global_best_particle]

		best_paticle_id = -1
		it_num = 0
		best_to_global_var = 1.0 + min_variation_accepted
		previous_global_best = None
		break_counter = it_until_break

		if print_progress:
			it_per_print = iteration_max // 20

		while it_num < iteration_max and break_counter > 0:
			it_num += 1
			previous_global_best = self.global_best_position

			for particle_id in range(self.num_particles):
				# Local and global random rates used to 
				# update particle velocity
				rate_p, rate_g = random.random(size=2)

				# Update particle velocity
				self.velocity[:, particle_id] = omega * self.velocity[:, particle_id] + \
					phi_p * rate_p * (self.particle_best_position[:, particle_id] - \
						self.position[:, particle_id]) + \
					phi_g * rate_g * (self.global_best_position - \
						self.position[:, particle_id])

				# Update particles position (based on current
				# velocity)
				self.position[:, particle_id] += \
					self.velocity[:, particle_id]
				
				# Calculate current particle new cost
				particle_new_cost = self.__costfunction__(\
						index=particle_id, 
						minimization=minimization)

				# Update particle best position
				if particle_new_cost < self.particle_cost[particle_id]:
					self.particle_best_position[:, particle_id] = \
						self.position[:, particle_id]

				# Update particle new cost
				self.particle_cost[particle_id] = particle_new_cost

				if self.particle_cost[particle_id] < \
					self.particle_cost[self.global_best_particle]:

					# Update global best particle id
					self.global_best_particle = particle_id

					# Update global best position
					self.global_best_position = self.position[:, \
						self.global_best_particle]

			# Calculate current local best to best global variation.
			# (using infinite norm)
			best_to_global_var = max(abs(self.global_best_position - \
				previous_global_best))

			if best_to_global_var < min_variation_accepted:
				break_counter -= 1
			else:
				break_counter = it_until_break

			if print_progress and (it_num % it_per_print == 0):
				print(it_num, ":", 
					self.global_best_position, 
					"(variance:", best_to_global_var, ")")
		
		# Build up method output
		ans = {
			"minimization" : minimization,
			"desired goal" : copy.deepcopy(goal),
			"particles num" : self.num_particles,
			"total iterations" : it_num,
			"best particle id" : self.global_best_particle,
			"best global solution" : self.global_best_position,
			"particle velocity" : self.velocity[:, self.global_best_particle],
			"Factors" : {"omega" : omega, "phi_p" : phi_p, "phi_g" : phi_g}
		}
		
		return ans

	def __costfunction__(self, index=-1, minimization=True):
		# Calculate the euclidean distance of each particle from
		# the goal position
		if index < 0:
			cost_array = array(\
				[(sum((self.position[:, i] - self.goal)**2.0))**0.5 \
				for i in range(self.num_particles)])
			return cost_array if minimization else (-1.0 * cost_array)
		else:
			res = (sum((self.position[:, index] - self.goal)**2.0))**0.5
			return res if minimization else (-1.0 * res)

if __name__ == "__main__":
	import sys

	if len(sys.argv) < 1:
		print("usage:", sys.argv[0], 
			"[-it max_number_of_iterations, default to 1000]",
			"[-num number_of_particles, default to 10]",
			"[-minvar, default to 1.0e-10]",
			"[-min, default to min(goal) - 1.0]",
			"[-max, default to max(goal) + 1.0]",
			"[-phip, default to 0.7]",
			"[-phig, default to (1 - phip)]",
			"[-omega, default to 0.5]",
			"[-goal x y ... z]", sep="\n\t")
		exit(1)

	try:
		particle_num = int(sys.argv[1 + sys.argv.index("-num")])
	except:
		particle_num = 10

	try:
		it_num = int(sys.argv[1 + sys.argv.index("-it")])
	except:
		it_num = 1000

	try:
		goal = tuple(map(float, sys.argv[1 + sys.argv.index("-goal"):]))
		dim = len(goal)
	except:
		goal = (0.5, 0.5)
		dim = 2

	try:
		min_range_val = float(sys.argv[1 + sys.argv.index("-min")])
	except:
		min_range_val = min(goal) - 1.0

	try:
		max_range_val = float(sys.argv[1 + sys.argv.index("-max")])
	except:
		max_range_val = max(goal) + 1.0

	try:
		min_var_accepted = float(sys.argv[1 + sys.argv.index("-minvar")])
	except:
		min_var_accepted = 1.0e-10

	try:
		omega = float(sys.argv[1 + sys.argv.index("-omega")])
		if not 0 <= omega <= 1.0:
			print("Warning: \"omega\" must be in [0.0, 1.0]")
			raise Exception

	except:
		omega = 0.5

	try:
		phi_p = float(sys.argv[1 + sys.argv.index("-phip")])
		if not 0 <= phi_p <= 1.0:
			print("Warning: \"phip\" must be in [0.0, 1.0]")
			raise Exception
	except:
		phi_p = 0.7

	try:
		phi_g = float(sys.argv[1 + sys.argv.index("-phig")])
		if not 0 <= phi_g <= 1.0:
			print("Warning: \"phig\" must be in [0.0, 1.0]")
			raise Exception
	except:
		phi_g = 1.0 - phi_p

	pso = Pso(
		interval=(min_range_val, max_range_val),
		dimension=dim,
		num_particles=particle_num)

	ans = pso.run(
		omega=omega,
		phi_p=phi_p,
		phi_g=phi_g,
		iteration_max=it_num,
		goal=goal,
		min_variation_accepted=min_var_accepted)

	print("\nResult:")
	for key in ans:
		print(key, "\t:", ans[key])
	
