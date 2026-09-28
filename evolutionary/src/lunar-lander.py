import random
import copy
import numpy as np
import warnings
import gymnasium as gym 
import os
import struct
import math
from multiprocessing import Process, Queue
import argparse

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module="pygame.pkgdata",
)

NUM_PROCESSES: int = os.cpu_count() or 1

evaluationQueue = Queue()
evaluatedQueue = Queue()

# ======== Configuration parameters ==========

ENABLE_WIND = True
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 1000
STEPS = 500

# =========== Neural Network parameters ===========

INPUT_NUMBER = 8
OUTPUT_NUMBER = 2
NEURON_NUMBER = 36 #12 ou 36
SHAPE = (INPUT_NUMBER, NEURON_NUMBER, OUTPUT_NUMBER)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

# =========== Experiment parameters ===========

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.05
ELITE_SIZE = 1

# =========== Genetic Algorithm settings ===========

SELECTION_TYPE = "tournament" # 'roulette' or 'tournament'
SELECTION_SUBSET_SIZE = 5 # Only used for tournament selection, defines the size of the subset of the population that will compete in each tournament.

CROSSOVER_TYPE = "uniform" # 'two_point' or 'uniform' or 'arithmetic'
CROSSOVER_UNIFORM_PROBABILITY = 0.5 # Only used for uniform crossover, defines the probability of each gene coming from the father (as opposed to the mother)

MUTATION_TYPE="uniform" # 'none', 'bit_flip', 'bitwise_inversion', 'gaussian', 'uniform'
MUTATION_STRENGTH = 0.2 # Only used for gaussian mutation, defines the standard deviation of the gaussian distribution used to mutate the weights, should be between 0.01 and 0.5
MUTATION_DELTA = 0.1 # Only used for uniform mutation, defines the range of the uniform distribution used to mutate the weights, should be between 0.01 and 0.5

# =========== Simulation parameters ===========

EVOLVE = False # If False, the best individual from a log file will be tested instead of evolving new individuals
RENDER_MODE = "human" # 'human' or None
N_RUNS = 5 # Number of runs to evolve, only used if EVOLVE is True

## =============== Pipeline arguments ===============

parser = argparse.ArgumentParser()
parser.add_argument("--evolve", type=str, help="Whether to evolve or test individuals")
parser.add_argument("--render", type=str, help="Whether to render the environment during testing")
parser.add_argument("--wind", type=str, help="Whether to enable wind in the environment")
parser.add_argument("--neuron-number", type=int, help="Number of neurons in the hidden layer")
parser.add_argument("--mutation", type=float, help="Mutation rate")
parser.add_argument("--crossover", type=float, help="Crossover rate")
parser.add_argument("--elitism", type=int, help="Elite size")
parser.add_argument("--render-mode", type=str, help="Render mode to use when testing")
args, unknown = parser.parse_known_args()

if args.evolve is not None:
    EVOLVE = args.evolve.lower() == "true"
if args.render is not None:
    RENDER_MODE = args.render if args.render.lower() != "none" else None
if args.wind is not None:
    ENABLE_WIND = args.wind.lower() == "true"
if args.neuron_number is not None:
    NEURON_NUMBER = args.neuron_number
    SHAPE = (INPUT_NUMBER, NEURON_NUMBER, OUTPUT_NUMBER)
    GENOTYPE_SIZE = 0
    for i in range(1, len(SHAPE)):
        GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]
if args.mutation is not None:
    MUTATION_PROBABILITY = args.mutation
if args.crossover is not None:
    CROSSOVER_PROBABILITY = args.crossover
if args.elitism is not None:
    ELITE_SIZE = args.elitism
if args.render_mode is not None:
    RENDER_MODE = None if args.render_mode.lower() == "none" else args.render_mode

# =============== OBJECTIVE FUNCTION  ===============

def fitness(observation_history) -> tuple[float, bool]:
    """
    Fitness inspired by the TP1 reactive controller rules.
    """

    trajectory_score = 0.0

    for observation in observation_history:

        x, y, vx, vy, theta, vtheta, left_leg, right_leg = observation

        x_pred = x + vx
        if x_pred > 0:
            target_angle = 0.3
        elif x_pred < 0:
            target_angle = -0.3
        else:
            target_angle = 0.0

        sigma = (theta - target_angle) + vtheta
        altitude_low = y < abs(x)
        falling_fast = vy < -0.20
        descent_error = max(0.0, -vy - 0.2)

        trajectory_score -= 0.35 * abs(x)
        trajectory_score -= 0.10 * y
        trajectory_score -= 0.10 * abs(vx)
        trajectory_score -= 0.30 * abs(sigma)

        if altitude_low or falling_fast:
            trajectory_score -= 1.00 * descent_error
        else:
            trajectory_score -= 0.35 * descent_error

        if left_leg == 1 and right_leg == 1:
            trajectory_score += 0.5

    trajectory_score /= len(observation_history)

    final_observation = observation_history[-2]
    x, y, vx, vy, theta, vtheta, left_leg, right_leg = final_observation

    legs_touching = left_leg == 1 and right_leg == 1
    on_landing_pad = abs(x) <= 0.2
    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    successful_landing = legs_touching and on_landing_pad and stable_velocity and stable_orientation

    terminal_score = 0.0
    terminal_score -= 12.0 * abs(x)
    terminal_score -= 6.0 * abs(y)
    terminal_score -= 5.0 * abs(vx)
    terminal_score -= 14.0 * max(0.0, -vy - 0.2)
    terminal_score -= 8.0 * abs(theta)
    terminal_score -= 3.0 * abs(vtheta)

    if successful_landing:
        terminal_score += 100.0
    elif y <= 0.05:
        terminal_score -= 50.0

    fitness = trajectory_score + terminal_score

    return fitness, check_successful_landing(final_observation)


def objective(observation_history) -> tuple[float, bool]:
    return fitness(observation_history)


# =============== SELECTION ===============


def roulette_selection(population) -> dict:
    """
    Selects an individual from the population using roulette selection.
    """

    fitnesses = [individual['fitness'] for individual in population]
    
    if min(fitnesses) <= 0:
        fitnesses = [fitness - min(fitnesses) + 1 for fitness in fitnesses]
    
    total = sum(fitnesses)
    
    probabilities = [fitness / total for fitness in fitnesses]
    
    return copy.deepcopy(random.choices(population, weights=probabilities, k=1)[0])


def tournament_selection(population) -> dict:
    """
    Selects an individual from the population using tournament selection.
    """

    subset = random.sample(population, SELECTION_SUBSET_SIZE)
    subset.sort(key=lambda individual: individual['fitness'], reverse=True)
    return copy.deepcopy(subset[0])


def selection(population) -> dict:
    
    if SELECTION_TYPE == "roulette":
        return roulette_selection(population)
    elif SELECTION_TYPE == "tournament":
        return tournament_selection(population)
    return {}


# =============== CROSSOVER ===============


def crossover_two_point(father, mother):
    """
    Two-Point Crossover: two crossover points are randomly chosen, and the segment between them is swapped
    """

    genotype_1 = father['genotype']
    genotype_2 = mother['genotype']
    
    point1 = random.randint(1, GENOTYPE_SIZE - 2)
    point2 = random.randint(point1 + 1, GENOTYPE_SIZE - 1)
    
    child_genotype = genotype_1[:]
    child_genotype[point1:point2] = genotype_2[point1:point2]
    
    return {'genotype': child_genotype, 'fitness': None}


def crossover_uniform(father, mother):
    """
    Uniform Crossover: each gene has a chance of coming from each parent
    """

    father_genotype = father['genotype']
    mother_genotype = mother['genotype']
    
    child_genotype = []

    for father_gene, mother_gene in zip(father_genotype, mother_genotype):
        if random.random() < CROSSOVER_UNIFORM_PROBABILITY:
            child_genotype.append(father_gene)
        else:
            child_genotype.append(mother_gene)
    
    return {'genotype': child_genotype, 'fitness': None}


def crossover_arithmetic(father, mother):
    """
    Arithmetic Crossover: weighted average of the parents' genes
    """

    father_genotype = father['genotype']
    mother_genotype = mother['genotype']

    alpha = random.random() # alpha defines the weight of each parent (0.5 = simple average)
    
    child_genotype = []
    for father_gene, mother_gene in zip(father_genotype, mother_genotype):
        child_genotype.append(alpha * father_gene + (1 - alpha) * mother_gene)
    
    return {'genotype': child_genotype, 'fitness': None}


def crossover(father, mother):
    """
    Selects which crossover to use based on a global variable
    """
    
    if CROSSOVER_TYPE == "two_point":
        return crossover_two_point(father, mother)
    elif CROSSOVER_TYPE == "uniform":
        return crossover_uniform(father, mother)
    elif CROSSOVER_TYPE == "arithmetic":
        return crossover_arithmetic(father, mother)
    return None

# =============== MUTATION ===============


def _get_bits(weight_float: float) -> str:
    """
    Convert floats to strings of 32 bits
    """
    as_int = struct.unpack('I', struct.pack('f', weight_float))[0]
    # Fill with 0s to get 32 bits
    bits = bin(as_int)[2:].zfill(32)

    return bits


def bit_flip(genotype, mutation_rate=0.5):
    """
    Has a mutation_rate chance of flipping each bit
    """

    new_genotype = list()

    for gene in genotype:

        if random.random() > MUTATION_PROBABILITY:
            new_genotype.append(gene)
            continue

        bits = _get_bits(gene)

        new_bits = ""

        for bit in bits:
            if random.random() < mutation_rate:
                new_bits += str(int(bit) ^ 1)
                continue
            new_bits += bit

        # Repack float
        new_gene = struct.unpack('f', struct.pack('I', int(new_bits, 2)))[0]
        # Flipping float bits can create infinite numbers
        # if it happens, just ignore, or clamp to [-1,1]
        if not math.isfinite(new_gene):
            new_gene = gene
        else:
            new_gene = max(min(new_gene, 1.0), -1.0)
        new_genotype.append(new_gene)

    return new_genotype


def bitwise_inversion(genotype):
    """
    Flips all bits in a random interval defined by start and end
    """

    start = random.randint(0, 31)
    # NOTE: no -1 because the end index is exclusive
    end = random.randint(start, 32)

    new_genotype = list()

    for gene in genotype:

        if random.random() > MUTATION_PROBABILITY:
            new_genotype.append(gene)
            continue

        bits = _get_bits(gene)

        new_bits = bits[:start] + "".join(str(int(bi) ^ 1) for bi in bits[start:end]) + bits[end:]

        new_gene = struct.unpack('f', struct.pack('I', int(new_bits, 2)))[0]
        # Flipping float bits can create infinite numbers and huge numbers
        # if it happens, just ignore, or clamp to [-1,1]
        if not math.isfinite(new_gene):
            new_gene = gene
        else:
            new_gene = max(min(new_gene, 1.0), -1.0)
        new_genotype.append(new_gene)

    return new_genotype


def gaussian_mutation(genotype, sigma=MUTATION_STRENGTH):
    """
    Apply Gaussian Mutation to the genotype w/ boundary handling
    """

    new_genotype = list()

    for gene in genotype:

        if random.random() > MUTATION_PROBABILITY:
            new_genotype.append(gene)
            continue

        new_gene = max(min(gene + random.gauss(0, sigma), 1.0), -1.0)
        new_genotype.append(new_gene)

    return new_genotype


def uniform_mutation(genotype, delta=MUTATION_DELTA):
    """
    Apply a standard uniform mutation: add a random value from a uniform distribution in the range [-delta, delta] to the weight, with boundary handling
    """

    new_genotype = list()

    for gene in genotype:

        if random.random() > MUTATION_PROBABILITY:
            new_genotype.append(gene)
            continue

        new_gene = gene + random.uniform(-delta, delta)
        new_gene = max(min(new_gene, 1.0), -1.0)
        new_genotype.append(new_gene)

    return new_genotype


def mutation(genotype):

    if MUTATION_TYPE == "none":
        return genotype

    elif MUTATION_TYPE == "bit_flip":
        return bit_flip(genotype)
    
    elif MUTATION_TYPE == "bitwise_inversion":
        return bitwise_inversion(genotype)

    elif MUTATION_TYPE == "gaussian":
        return gaussian_mutation(genotype)

    elif MUTATION_TYPE == "uniform":
        return uniform_mutation(genotype)

    return genotype

# =============== SIMULATION AND EVALUATION ===============

def network(shape, observation, ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x


def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False


def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)

    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective(observation_history)


def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        fit, success = simulate(ind['genotype'], seed = None, env = env)
        ind['fitness'] = fit
        ind['success'] = success
        evaluatedQueue.put(ind)
    env.close()


def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop


def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population


def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    


def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    results = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    successes = sum(1 for ind in population if ind.get('success', False))
    success_rate = successes / POPULATION_SIZE * 100
    average_fitness = sum(ind['fitness'] for ind in population) / POPULATION_SIZE
    result = (population[0]['genotype'], population[0]['fitness'], average_fitness, success_rate)
    results.append(result)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS-1):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:

            if random.random() < CROSSOVER_PROBABILITY:
                p1 = selection(population)
                p2 = selection(population)
                ni = crossover(p1, p2)
                #print(f"Crossover between {p1['fitness']} and {p2['fitness']}")
            else:
                ni:dict = selection(population)

            ni['genotype'] = mutation(ni['genotype'])
            #print(f"Mutation happened to an individual")

            offspring.append(ni)

        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        successes = sum(1 for ind in population if ind.get('success', False))
        success_rate = successes / POPULATION_SIZE * 100

        average_fitness = sum(ind['fitness'] for ind in population) / POPULATION_SIZE

        #Print and save the best of the current generation
        result = (population[0]['genotype'], population[0]['fitness'], average_fitness, success_rate)
        results.append(result)
        print(f'GENERATION {gen+1}: Best fitness: {result[1]:.2f} | Average fitness: {average_fitness:.2f} | Success rate: {success_rate:.1f}%')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return results


def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            parts = line.split('\t')
            fitness = float(parts[0])
            shape = eval(parts[3])
            genotype = eval(parts[4])
            bests.append((fitness, shape, genotype))
    return bests


def evolve():

    seeds = [963, 952, 364, 912, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
    for i in range(N_RUNS):    
        random.seed(seeds[i])
        results = evolution()
        with open(f'logs/log{i}.txt', 'w') as f:
            for r in results:
                f.write(f'{r[1]}\t{r[2]}\t{r[3]}\t{SHAPE}\t{r[0]}\n')


def test():
    #test evolved individuals
    #pick the file to test
    filename = 'logs/log0.txt'
    bests = load_bests(filename)
    best = bests[-1]
    SHAPE = best[1]
    ind = best[2]

    ind: dict = {'genotype': ind, 'fitness': None}

    ntests = TEST_EPISODES

    fit, success = 0, 0
    for i in range(1,ntests+1):
        f, s = simulate(ind['genotype'], render_mode=RENDER_MODE, seed = None)
        print(f"Test {i}/{ntests} | Fitness: {f:.2f} | Success: {'Yes' if s else 'No'}")
        fit += f
        success += s
    print(fit/ntests, success/ntests)


if __name__ == '__main__':

    print(f"Configuration: EVOLVE={EVOLVE}, RENDER_MODE={RENDER_MODE}, ENABLE_WIND={ENABLE_WIND}, NEURON_NUMBER={NEURON_NUMBER}, MUTATION_PROBABILITY={MUTATION_PROBABILITY}, CROSSOVER_PROBABILITY={CROSSOVER_PROBABILITY}, ELITE_SIZE={ELITE_SIZE}")
    
    if EVOLVE:
        evolve()          
    else:
        test()
