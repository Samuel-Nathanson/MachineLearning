import argparse
import multiprocessing
import time
import os
import traceback
import numpy as np
import sys
import random


def racetrack_experiment(track_name: str,
                         track_matrix: np.chararray,
                         algorithm_name: str,
                         crash_scenario_name: str,
                         discount_factor: float,
                         learning_rate: float,
                         convergence_threshold: float,
                         results_dict: dict,
                         lock):
    '''
    :param track_name:
    :param track_matrix:
    :param algorithm_name:
    :param crash_scenario_name:
    :param discount_factor:
    :param learning_rate:
    :param convergence_threshold:
    :param results_dict:
    :param lock:
    :return:
    '''


    '''
    ================================================================
    Random Action Functions
    ================================================================

    '''
    def get_random_acceleration():
        return random.choice(get_possible_accelerations())


    def get_random_direction():
        return random.choice(get_possible_directions())


    '''
    ================================================================
    State + Action Possibilities
    ================================================================
    '''

    def update_state(state, action):
        position = (state[0], state[1])
        velocity = (state[2], state[3])

        direction = (action[0], action[1])
        acceleration_magnitude = (action[2])

        if(random.random() > get_probability_of_action_success()):
            safe_print("Action failed")
        else:
            new_v_y = velocity[0] + direction[0] * acceleration_magnitude
            new_v_x = velocity[1] + direction[1] * acceleration_magnitude
            velocity = (new_v_y, new_v_x)


    def get_probability_of_action_success():
        return 0.8

    def get_possible_accelerations():
        return [-1, 0, 1]


    def get_possible_directions():
        north = (1,0)
        east = (0, 1)
        south = (-1, 0)
        west = (0, -1)
        return [north, east, south, west]


    def get_possible_velocities():
        velocities = []
        for v_x in range(-5,6):
            for v_y in range(-5,6):
                velocities.append((v_y, v_x))
        return velocities


    def get_possible_positions():
        positions = []
        for y, x in np.ndindex(track_matrix.shape):
            positions.append((y,x))
        return positions

    def get_possible_valid_positions():
        valid_positions = []
        for (y,x) in get_possible_positions():
            if(track_matrix[y,x] == '.'):
                valid_positions.append((y,x))
        return valid_positions


    def get_possible_actions():
        actions = []
        for direction in get_possible_directions():
            for acceleration in get_possible_accelerations():
                actions.append((direction + acceleration))


    def get_possible_valid_states():
        valid_states = []
        for position in get_possible_valid_positions():
            for velocity in get_possible_velocities():
                state = position + velocity
                valid_states.append(state)
        return valid_states


    def get_possible_states():
        states = []
        for position in get_possible_positions():
            for velocity in get_possible_velocities():
                state = position + velocity
                states.append(state)
        return states


    def safe_print(*args):
        lock.acquire()
        print(" ".join(map(str, args)))
        sys.stdout.flush()
        lock.release()



    def reward(state, action):
        new_state = update_state(state, action)

    def value_iteration_experiment():
        # initialize states
        v = {}
        q = {}
        pi = {}
        valid_states = get_possible_valid_states
        actions = get_possible_actions()
        for state in valid_states():
            v[state] = random.random() * 10 - 5 # Arbitrary random values between -5, 5

        t = 0
        while True:
            t = t+1
            q[t] = {}
            pi[t] = {}
            v[t] = {}

            for s in valid_states:
                q[t][s] = {}
                for a in actions:
                    q[t][s][a] = reward(s,a) + discount_factor * sum([get_probability_of_action_success()])

                pi[t][s] = max(q[t][s], key=q[t][s].get)
                v[t][s] = q[t][s][pi[t][s]]

            if(all(i < convergence_threshold for i in v[t][s])):
                break


    def q_learning_experiment():
        pass

    def sarsa_experiment():
        pass

    '''
    Run experiment
    '''
    experiments = {
        "value-iteration" : value_iteration_experiment,
        "q-learning" : q_learning_experiment,
        "SARSA": sarsa_experiment
    }

    safe_print(f"Reporting from process: Experiment {algorithm_name} on matrix of shape {track_matrix.shape} with crash scenario {crash_scenario_name}")

    result = experiments[algorithm_name]()




def read_racetrack_file(file_name: str, tracks_dict: dict):
    '''
    Example
    :param file_name:
    :param tracks_dict:
    :return:
    Example:
    11,37
    #####################################
    ################################FFFF#
    ################################....#
    ################################....#
    ################################....#
    ################################....#
    #S..................................#
    #S..................................#
    #S..................................#
    #S..................................#
    #####################################
    '''
    try:
        with open(file_name, "r") as racetrack_file:
            n_rows, n_cols = [int(x) for x in (racetrack_file.readline()).split(",")]
            track_matrix = np.chararray([n_rows, n_cols], unicode=True)

            for row in range(0, n_rows):
                line = racetrack_file.readline()
                for col in range(0, n_cols):
                    track_matrix[row][col] = line[col]

            tracks_dict[file_name] = track_matrix
    except:
        traceback.print_exc()
        exit(56)

def validate_convergence_threshold(threshold):
    assert (threshold > 0)
    return threshold


def validate_input(input):
    if type(input) != list:
        input = [input]
    for i in input:
        assert (os.path.exists(i))
    return input


def validate_discount_factor(discount):
    assert (discount <= 1)
    assert (discount >= 0)
    return discount


def validate_learning_rate(rate):
    assert (rate > 0)
    assert (rate < 1)
    return rate


def main():

    '''
    Parse arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-inputs",
                        nargs="+",
                        help="input files",
                        type=str,
                        default=['L-track.txt', 'R-track.txt', 'O-track.txt'])
    parser.add_argument("-algorithms",
                        nargs="+",
                        help="Algorithms to run",
                        choices=['value-iteration', ' q-learning', 'SARSA'],
                        type=str,
                        default=['value-iteration', ' q-learning', 'SARSA'])
    parser.add_argument("-crash-scenarios",
                        nargs="+",
                        help="Crash scenario",
                        choices=['previous-state', 'restart'],
                        type=str,
                        default=['previous-state', 'restart'])
    parser.add_argument("-discount-factor",
                        help="Discount factor. Larger discount factors mean that rewards further in the future count more",
                        type=float,
                        default=0.01)
    parser.add_argument("-learning_rate",
                        help="Learning rate. Higher learning rates cause learning to occur faster, but may cause oscillation around convergence",
                        type=float,
                        default=0.1)
    parser.add_argument("-convergence_threshold",
                        help="Convergence Threshold",
                        type=float,
                        default=0.1)
    args = parser.parse_args()
    inputs = validate_input(args.inputs)
    algorithms = args.algorithms
    crash_scenarios = args.crash_scenarios
    discount_factor = validate_discount_factor(args.discount_factor)
    convergence_threshold = validate_convergence_threshold(args.convergence_threshold)
    learning_rate = validate_learning_rate(args.learning_rate)

    '''
    Read Input File(s)
    '''
    t0 = time.time()
    num_files = len(inputs)
    file_reading_jobs = []
    manager = multiprocessing.Manager()
    input_tracks_dict = manager.dict()

    for n in range(0, num_files):
        proc = multiprocessing.Process(target=read_racetrack_file,
                                       args=(inputs[n],
                                             input_tracks_dict))
        file_reading_jobs.append(proc)

    for proc_num, j in enumerate(file_reading_jobs):
        print(f"Reading file {inputs[proc_num]} using process {proc_num}")
        j.start()

    for j in file_reading_jobs:
        j.join()
    t1 = time.time()
    print(f"Time to read {num_files} files: {t1 - t0}")

    '''
    Run algorithm(s) for each track.
    '''
    experiment_jobs = {}
    manager = multiprocessing.Manager()
    results_dict = manager.dict()
    lock = multiprocessing.Lock()
    experiment_timing = {}  # Time for each individual experiment

    for input_num, track_name in enumerate(input_tracks_dict.keys()):
        for algo_num, algorithm_name in enumerate(algorithms):
            for scenario_num, crash_scenario_name in enumerate(crash_scenarios):
                proc = multiprocessing.Process(target=racetrack_experiment,
                                               args=(track_name,
                                                     input_tracks_dict[track_name],
                                                     algorithm_name,
                                                     crash_scenario_name,
                                                     discount_factor,
                                                     learning_rate,
                                                     convergence_threshold,
                                                     results_dict,
                                                     lock))
                experiment_jobs[f"{algorithm_name} on {track_name} with {crash_scenario_name} policy"] = proc

    for proc_num, experiment_key in enumerate(experiment_jobs.keys()):
        print(f"Performing experiment {experiment_key} with process {proc_num}")
        experiment_jobs[experiment_key].start()
        experiment_timing[experiment_key] = time.time()

    for experiment_key, job in experiment_jobs.items():
        job.join()
        experiment_timing[experiment_key] = time.time() - experiment_timing[experiment_key]
        print(f"Finished experiment: {experiment_key} in {experiment_timing[experiment_key]} seconds")

    t1 = time.time()
    print(f"Average time per experiment: {np.average(list(experiment_timing.values()))}")
    print(f"Time to perform {len(experiment_timing.keys())} experiments concurrently: {t1 - t0}")


if __name__ == "__main__":
    main()