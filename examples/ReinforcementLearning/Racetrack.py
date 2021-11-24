import argparse
import multiprocessing
import time
import os
import traceback
import numpy as np
import sys
import matplotlib.pyplot as plt

def racetrack_experiment(track_name: str,
                         track_matrix: np.chararray,
                         algorithm_name: str,
                         crash_scenario_name: str,
                         discount_factor: float,
                         learning_rate: float,
                         convergence_threshold: float,
                         results_dict: dict,
                         lock):

    def safe_print(*args):
        lock.acquire()
        print(" ".join(map(str, args)))
        sys.stdout.flush()
        lock.release()

    safe_print(f"Reporting from process: Experiment {algorithm_name} on matrix of shape {track_matrix.shape} with crash scenario {crash_scenario_name}")

'''

    
'''
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
            track_matrix = np.chararray([n_rows, n_cols])

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

    plt.pause(100)


if __name__ == "__main__":
    main()