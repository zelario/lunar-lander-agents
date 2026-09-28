import matplotlib.pyplot as plt
import os
import subprocess
import sys
import numpy as np
import seaborn as sns
import scipy.stats as stats
import zipfile

# ========== Experiment parameters ===========

N_RUNS = 5 # Manter os mesmos da main
N_EXPERIMENTS = 8
EXPERIMENTS = [
    (0.008, 0.5, 0),
    (0.05, 0.5, 0),
    (0.008, 0.9, 0),
    (0.05, 0.9, 0),
    (0.008, 0.5, 1),
    (0.05, 0.5, 1),
    (0.008, 0.9, 1),\
    (0.05, 0.9, 1),
]

EXPERIMENTS_TO_RUN = [1,2,3,4,5,6,7,8] 
WIND = True

def run_experiments(wind):

    if wind:
        neurons_number = 36
    else:
        neurons_number = 12
 
    for index, (mutation, crossover, elite) in enumerate(EXPERIMENTS):

        if (index + 1) not in EXPERIMENTS_TO_RUN:
            print(f"Skipping experiment {index+1} with mutation={mutation}, crossover={crossover}, elitism={elite}")
            continue

        print(f"Running experiment {index+1} with mutation={mutation}, crossover={crossover}, elitism={elite}")

        experiment_directory = f"logs/experiment{index+1}"
        os.makedirs(experiment_directory, exist_ok=True)
        subprocess.run([
            sys.executable, "src/lunar-lander.py",
            f"--evolve=True",
            f"--wind={'True' if wind else 'False'}",
            f"--neuron-number={neurons_number}",
            f"--mutation={mutation}",
            f"--crossover={crossover}",
            f"--elitism={elite}"
        ])

        for run in range(N_RUNS):
            logname = f"log{run}.txt"
            src_log = os.path.join("logs", logname)
            dst_log = os.path.join(experiment_directory, logname)
            if os.path.exists(src_log):
                os.replace(src_log, dst_log)


def test_elite(wind):
    """
    Testa o desempenho do indivíduo elite de cada experiência.
    Para cada experiência, o indivíduo elite é testado em 1000 execuções para avaliar
    a sua taxa de sucesso de pouso e o seu fitness."""

    if wind:
        neurons_number = 36
    else:
        neurons_number = 12

    base_dir = os.path.join("logs")
    results_path = os.path.join(base_dir, "results.txt")

    with open(results_path, "a") as results:

        for exp in range(1, N_EXPERIMENTS + 1):

            if exp not in EXPERIMENTS_TO_RUN:
                print(f"Skipping testing for experiment {exp}")
                continue

            experiment_directory = os.path.join(base_dir, f"experiment{exp}")
            print(f"Processing {experiment_directory}")
            for run in range(N_RUNS):
                log_path = os.path.join(experiment_directory, f"log{run}.txt")
                print(f"  Evaluating {log_path}")
                temp_log = os.path.join(base_dir, "log0.txt")
                if os.path.exists(temp_log):
                    os.remove(temp_log)
                with open(log_path, "r") as src, open(temp_log, "w") as dst:
                    dst.write(src.read())
                arguments = [
                    sys.executable, "src/lunar-lander.py",
                    "--evolve=False",
                    f"--wind={'True' if wind else 'False'}",
                    f"--neuron-number={neurons_number}",
                    "--mutation=0",
                    "--crossover=0",
                    "--elitism=0",
                ]
                print(f"    Running lunar-lander.py for {log_path}")
                result = subprocess.run(arguments, capture_output=True, text=True)
                output = result.stdout.strip().split("\n")
                for line in reversed(output):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        print(f"    Result: {line.strip()}")
                        results.write(f"experiment{exp}/log{run}.txt: {line.strip()}\n")
                        results.flush()
                        break


def fitness_success_correlation():
    """
    Analisa a correlação entre fitness e taxa de sucesso de pouso.
    Usa os resultados do teste das elites de cada experiencia ao longo de 1000 execucoes
    para calcular a relação entre o fitness medio e a taxa de sucesso de pouso desse individuo.
    """

    experiments = {i: {'fitness': [], 'success': []} for i in range(1, 9)}

    with open('logs/results.txt', 'r') as f:

        for line in f:

            if not line.strip():
                continue

            # Exemplo de linha: experiment1/log0.txt: -22.146732 0.242
            path, values = line.strip().split(':')
            fitness, success = map(float, values.strip().split())
            experiment_number = int(path.split('/')[0].replace('experiment', '').replace('experiencia', ''))
            experiments[experiment_number]['fitness'].append(fitness)
            experiments[experiment_number]['success'].append(success)

    all_fitness = []
    all_success = []
    for data in experiments.values():
        all_fitness.extend(data['fitness'])
        all_success.extend(data['success'])

    if len(all_fitness) > 1:

        # Pearson correlation
        mean_fitness = sum(all_fitness) / len(all_fitness)
        mean_success = sum(all_success) / len(all_success)
        numerator = sum((f - mean_fitness) * (s - mean_success) for f, s in zip(all_fitness, all_success))
        denominator_fitness = sum((f - mean_fitness) ** 2 for f in all_fitness)
        denominator_success = sum((s - mean_success) ** 2 for s in all_success)
        corr = numerator / (denominator_fitness ** 0.5 * denominator_success ** 0.5)

    plt.figure(figsize=(10, 6))
    for experiment_number, data in experiments.items():
        plt.scatter(data['fitness'], data['success'], label=f'Experiment {experiment_number}')

    plt.xlabel('Average Fitness')
    plt.ylabel('Success Rate')
    plt.title('Fitness vs Landing Success Rate per Experiment')

    x = all_fitness
    y = all_success
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = sum((xi - mean_x) ** 2 for xi in x)

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    x_min, x_max = min(x), max(x)
    x_vals = [x_min, x_max]
    y_vals = [slope * xi + intercept for xi in x_vals]
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Linear fit')
    
    if len(all_fitness) > 1:
        plt.gca().text(0.02, 0.98, f"Pearson r = {corr:.3f}", transform=plt.gca().transAxes,
                      fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.legend()
    plt.grid(True, which='both', axis='both')
    plt.tight_layout()
    plt.show()


def fitness_success_evolution():
    """
    Analisa a evolução do fitness e da taxa de sucesso de pouso ao longo das gerações.
    Mostra uma tendencia de melhoria do fitness médio ao longo das gerações,
    indicando que o algoritmo está a optimizar a solução ao longo do tempo."""

    log_path = "logs/log0.txt"
    generations = []
    mean_fitness = []
    success_rate = []
    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            generations.append(i)
            mean_fitness.append(float(parts[1]))
            success_rate.append(float(parts[2]))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Fitness", color=color1)
    ax1.plot(generations, mean_fitness, marker="o", color=color1, label="Mean Fitness")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', axis='both')

    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel("Success Rate", color=color2)
    ax2.plot(generations, success_rate, marker="s", color=color2, label="Success Rate")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, which='both', axis='both')

    plt.title("Mean Fitness and Success Rate Evolution Over Generations")
    fig.tight_layout()
    plt.show()


def fitness_success_boxplots():
    """
    Mostra boxplots do fitness e da taxa de sucesso para cada experiencia e a sua dispersão.
    """

    experiments = {i: {'fitness': [], 'success': []} for i in range(1, 9)}
    with open('logs/results.txt', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            path, values = line.strip().split(':')
            fitness, success = map(float, values.strip().split())
            experiment_number = int(path.split('/')[0].replace('experiment', '').replace('experiencia', ''))
            experiments[experiment_number]['fitness'].append(fitness)
            experiments[experiment_number]['success'].append(success)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].boxplot([experiments[i]['fitness'] for i in range(1, 9)])
    axs[0].set_title('Fitness per Experiment')
    axs[0].set_xlabel('Experiment')
    axs[0].set_ylabel('Fitness')
    axs[0].set_xticklabels([str(i) for i in range(1, 9)])

    axs[1].boxplot([experiments[i]['success'] for i in range(1, 9)])
    axs[1].set_title('Success Rate per Experiment')
    axs[1].set_xlabel('Experiment')
    axs[1].set_ylabel('Success Rate')
    axs[1].set_xticklabels([str(i) for i in range(1, 9)])

    plt.tight_layout()
    plt.show()


def success_rate_heatmap():
    """
    Mostra um heatmap da média de success rate para cada combinação de mutation, crossover e elite.
    """
    experiments = [
        (0.008, 0.5, 0),
        (0.05, 0.5, 0),
        (0.008, 0.9, 0),
        (0.05, 0.9, 0),
        (0.008, 0.5, 1),
        (0.05, 0.5, 1),
        (0.008, 0.9, 1),
        (0.05, 0.9, 1),
    ]
    results = []
    with open('logs/results.txt', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            path, values = line.strip().split(':')
            fitness, success = map(float, values.strip().split())
            experiment_number = int(path.split('/')[0].replace('experiment', '').replace('experiencia', ''))
            mutation, crossover, elite = experiments[experiment_number-1]
            results.append((mutation, crossover, elite, success))
    # Elite 0
    data0 = {(m, c): [] for m in [0.008, 0.05] for c in [0.5, 0.9]}
    # Elite 1
    data1 = {(m, c): [] for m in [0.008, 0.05] for c in [0.5, 0.9]}
    for m, c, e, s in results:
        if e == 0:
            data0[(m, c)].append(s)
        else:
            data1[(m, c)].append(s)
    heat0 = np.array([[np.mean(data0[(m, c)]) for c in [0.5, 0.9]] for m in [0.008, 0.05]])
    heat1 = np.array([[np.mean(data1[(m, c)]) for c in [0.5, 0.9]] for m in [0.008, 0.05]])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(heat0, annot=True, fmt='.2f', xticklabels=[0.5, 0.9], yticklabels=[0.008, 0.05], ax=axs[0], cmap='YlGnBu')
    axs[0].set_title('Elite=0')
    axs[0].set_xlabel('Crossover')
    axs[0].set_ylabel('Mutation')
    sns.heatmap(heat1, annot=True, fmt='.2f', xticklabels=[0.5, 0.9], yticklabels=[0.008, 0.05], ax=axs[1], cmap='YlGnBu')
    axs[1].set_title('Elite=1')
    axs[1].set_xlabel('Crossover')
    axs[1].set_ylabel('Mutation')
    plt.suptitle('Heatmap of Mean Success Rate')
    plt.tight_layout()
    plt.show()


def statistic_tests():

    results = []
    with open('logs/results.txt', 'r') as f:
        for line in f:
            if not line.strip():
                continue
            path, values = line.strip().split(':')
            fitness, success = map(float, values.strip().split())
            experiment_number = int(path.split('/')[0].replace('experiment', '').replace('experiencia', ''))
            mutation, crossover, elite = EXPERIMENTS[experiment_number-1]
            results.append({'mutation': mutation, 'crossover': crossover, 'elite': elite, 'success': success, 'fitness': fitness, 'exp': experiment_number})

    # Normality test (Shapiro-Wilk) for each experiment group
    print('Normality test (Shapiro-Wilk) for success rate of each experiment:')
    normal = True
    for experiment in range(1, 9):
        group = [d['success'] for d in results if d['exp'] == experiment]
        if len(group) > 2:
            stat, p = stats.shapiro(group)
            print(f'  Exp {experiment}: W={stat:.3f}, p={p:.3f} (', end='')
            if p > 0.05:
                print('normal)')
            else:
                print('not normal)')
                normal = False
    print()

    # Group comparison (t-test if normal, else Mann-Whitney)
    def compare_groups(label, group1, group2):
        if normal:
            stat, p = stats.ttest_ind(group1, group2, equal_var=False)
            print(f't-test for {label}: t={stat:.3f}, p={p:.3f}')
        else:
            stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
            print(f'Mann-Whitney for {label}: U={stat:.3f}, p={p:.3f}')

    # Elite=0 vs Elite=1
    elite0 = [d['success'] for d in results if d['elite']==0]
    elite1 = [d['success'] for d in results if d['elite']==1]
    compare_groups('Elite=0 vs Elite=1', elite0, elite1)

    # Mutation low vs high
    mutation_low = [d['success'] for d in results if d['mutation']==0.008]
    mutation_high = [d['success'] for d in results if d['mutation']==0.05]
    compare_groups('Mutation 0.008 vs 0.05', mutation_low, mutation_high)

    # Crossover low vs high
    crossover_low = [d['success'] for d in results if d['crossover']==0.5]
    crossover_high = [d['success'] for d in results if d['crossover']==0.9]
    compare_groups('Crossover 0.5 vs 0.9', crossover_low, crossover_high)

def analyse_data():

    # ===== Analise do LOG0.TXT =====

    # Fitness and Success Rate evolution over generations

    fitness_success_evolution()

    # ===== Analise dos RESULTS.TXT =====

    # Fitness vs Success Rate correlation

    fitness_success_correlation()

    # Experiments metrics comparison

    fitness_success_boxplots()

    # Parameters influence on success rate
    success_rate_heatmap()

    # Statistical tests
    statistic_tests()

def backup_logs(wind):

    if wind:
        filename = 'evo-wind-test-wind.zip'
    else:
        filename = 'evo-basic-test-basic.zip'

    logs_dir = 'logs'
    backup_dir = 'backup'
    os.makedirs(backup_dir, exist_ok=True)
    zip_path = os.path.join(backup_dir, f'{filename}.zip')

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, subfolders, filenames in os.walk(logs_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, os.path.dirname(logs_dir))
                zipf.write(file_path, arcname)
    print(f'Logs folder zipped as a single folder to {zip_path}')


if __name__ == '__main__':

    print("What do you want to do?\n1. Evolve Individuals\n2. Test Elite\n3. Analyse data\n4. Run all (1, 2, 3)")
    choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

    if choice == '1':

        print("\nWhat experiments do you want to run? (1-8 separated by commas (e.g. 1,3,5), '0' for all experiments) or blank for what is already defined in the code")
        experiment_choice = input("Enter your choice: ").strip()
        if experiment_choice == '0':
            EXPERIMENTS_TO_RUN = [1, 2, 3, 4, 5, 6, 7, 8]
            print(f"Running all experiments: {EXPERIMENTS_TO_RUN}")
        elif experiment_choice:
            EXPERIMENTS_TO_RUN = [int(x) for x in experiment_choice.split(',') if x.strip().isdigit() and 1 <= int(x.strip()) <= 8]
            print(f"Running selected experiments: {EXPERIMENTS_TO_RUN}")
        else:
            print(f"Using default experiments to run: {EXPERIMENTS_TO_RUN}")

        print("\nDo you want to evolve with wind? (y/n)")
        wind_choice = input("Wind: ").strip().lower() == 'y'
        
        run_experiments(wind=wind_choice)

    elif choice == '2':

        print("\nDo you want to test a specific individual or various individuals?\n1. Specific individual (e.g. experiment3/log2.txt)\n2. All elites from certain experiments")
        test_choice = input("Enter your choice (1 or 2): ").strip()

        if test_choice == '1':
            print("\nWhich experiment is that individual from?")
            experiment = input("Experiment: ").strip()
            print("\nWhich log file is that individual from?")
            log_file = input("Log File: ").strip()
            print(f"\nWas it evolved with wind? (y/n)")
            wind = input("Wind: ").strip().lower() == 'y'

            print("\nWhat render mode do you want to use?\n1. Human\n2. None")
            render_mode = input("Render Mode: ").strip()

            if render_mode == '1':
                render_mode = 'human'
            else:                
                render_mode = 'none'

            if not experiment.startswith("experiment"):
                experiment = f"experiment{experiment}"

            if log_file.isdigit():
                log_file = f"log{log_file}.txt"
            elif not log_file.endswith(".txt"):
                log_file = f"{log_file}.txt"

            individual_path = os.path.join(experiment, log_file)
            if os.path.exists(os.path.join("logs", individual_path)):
                print(f"Testing individual from {individual_path}")
                with open(os.path.join("logs", individual_path), "r") as src, open(os.path.join("logs", "log0.txt"), "w") as dst:
                    dst.write(src.read())
                arguments = [
                    sys.executable, "src/lunar-lander.py",
                    "--evolve=False",
                    f"--render-mode={render_mode}",
                    f"--wind={'True' if wind else 'False'}",
                    f"--neuron-number={12}",
                    "--mutation=0",
                    "--crossover=0",
                    "--elitism=0",
                ]
                subprocess.run(arguments)
            else:
                print(f"File {individual_path} does not exist. Please check the path and try again.")

        elif test_choice == '2':

            print("What experiments do you want to test? (1-8 separated by commas (e.g. 1,3,5), '0' for all experiments) or blank for what is already defined in the code")
            experiment_choice = input("Enter your choice: ").strip()
            if experiment_choice == '0':
                EXPERIMENTS_TO_RUN = [1, 2, 3, 4, 5, 6, 7, 8]
                print(f"Testing elites from all experiments: {EXPERIMENTS_TO_RUN}")
            elif experiment_choice:
                EXPERIMENTS_TO_RUN = [int(x) for x in experiment_choice.split(',') if x.strip().isdigit() and 1 <= int(x.strip()) <= 8]
                print(f"Testing elites from selected experiments: {EXPERIMENTS_TO_RUN}")
            else:
                print(f"Using default experiments to test: {EXPERIMENTS_TO_RUN}")

            wind_choice = input("Wind: ").strip().lower() == 'y'
            test_elite(wind=wind_choice)

    elif choice == '3':
        print("Analysing data...\n")
        analyse_data()

    elif choice == '4':
        print("Running all steps: evolve, test and analyse...")
        run_experiments(wind=WIND)
        test_elite(wind=WIND)
        analyse_data()
