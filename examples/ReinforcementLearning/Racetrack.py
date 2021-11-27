import argparse
import copy
import multiprocessing
import time
import os
import traceback
import numpy as np
import sys
import random
np.set_printoptions(threshold=sys.maxsize, linewidth=2000)

MAX_VELOCITY_MAGNITUDE = 1
MIN_VELOCITY_MAGNITUDE = -1

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
    Data / Visualizations
    ================================================================
    '''
    def visualize_board(optimal_path, optimal_policy):
        import matplotlib.pyplot as plt
        for y, x in np.ndindex(track_matrix.shape):
            if (track_matrix[y, x] == '#'):
                color = 'black'
                shape = 's'
            if (track_matrix[y, x] == 'S'):
                color = 'yellow'
                shape = 's'
            if (track_matrix[y, x] == 'F'):
                color = 'green'
                shape = 's'
            if (track_matrix[y, x] == '.'):
                color = 'white'
                shape = 'x'
            plt.plot([x], [y], marker=shape, color=color, markersize=10)

        for (y, x) in optimal_path:
            plt.plot([x], [y], marker='.', color='blue', markersize=5)

        plt.show()
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
    Getting / Updating State and Action Parameters
    ================================================================
    '''

    def get_velocity(state):
        return (state[2], state[3])

    def get_position(state):
        return (state[0], state[1])

    def get_acceleration(action):
        return action[2]

    def update_velocity(state, velocity):
        state[2] = velocity[1]
        state[3] = velocity[2]

    def update_position(state, position):
        state[0] = position[0]
        state[1] = position[1]

    def update_acceleration(action, acceleration):
        action[2] = acceleration


    '''
    ================================================================
    State Update and Reward Functions
    ================================================================
    '''

    def update_state(state, action, failed_action=None):
        position = (state[0], state[1])
        velocity = (state[2], state[3])
        acceleration = action

        y = position[0]
        x = position[1]

        v_y = velocity[0]
        v_x = velocity[1]

        new_v_y = v_y
        new_v_x = v_x

        # For Value Iteration Algorithm, we need to find rewards for both scenarios (where action fails/succeeds)
        if(failed_action):
            pass
        elif random.random() > get_probability_of_action_success() and failed_action==None:
            pass
        else:
            # Update velocity
            new_v_y = v_y + acceleration[0]
            new_v_x = v_x + acceleration[1]

            # Set bounds to min, max
            new_v_y = new_v_y if new_v_y <= MAX_VELOCITY_MAGNITUDE else MAX_VELOCITY_MAGNITUDE
            new_v_y = new_v_y if new_v_y >= MIN_VELOCITY_MAGNITUDE else MIN_VELOCITY_MAGNITUDE
            new_v_x = new_v_x if new_v_x <= MAX_VELOCITY_MAGNITUDE else MAX_VELOCITY_MAGNITUDE
            new_v_x = new_v_x if new_v_x >= MIN_VELOCITY_MAGNITUDE else MIN_VELOCITY_MAGNITUDE

        new_y = y + new_v_y
        new_x = x + new_v_x

        new_position = (new_y, new_x)
        new_velocity = (new_v_y, new_v_x)

        crashed = False
        out_of_bounds = False

        if(is_outside_bounds(new_position)):
            out_of_bounds = True
            new_velocity = (0,0)
            if(crash_scenario_name == "near-wall"):
                new_position = get_crash_location((y,x), (new_y, new_x))
            else:
                new_position = get_starting_coordinates()

        if(crashed_through_wall(position, new_position)):
            crashed = True
            new_velocity = (0,0)
            if(crash_scenario_name == "near-wall"):
                new_position = (y, x)
            else:
                new_position = get_starting_coordinates()

        new_state = new_position + velocity

        return new_state, new_position, new_velocity, crashed, out_of_bounds


    def get_probability_of_action_success():
        return 0.8

    '''
    ================================================================
    State + Action Possibilities
    ================================================================
    '''

    def get_possible_accelerations():
        directions = get_possible_directions()
        zero_acceleration = (0,0)
        return directions + [zero_acceleration]


    def get_possible_directions():
        north = (1,0)
        east = (0, 1)
        south = (-1, 0)
        west = (0, -1)
        return [north, east, south, west]


    def get_possible_velocities():
        velocities = []
        # for v_x in range(-5,6):
        #     for v_y in range(-5,6):
        #         velocities.append((v_y, v_x))

        for v_x in range(MIN_VELOCITY_MAGNITUDE,MAX_VELOCITY_MAGNITUDE +1):
            for v_y in range(MIN_VELOCITY_MAGNITUDE,MAX_VELOCITY_MAGNITUDE +1):
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
            if(track_matrix[y,x] == '.' or track_matrix[y,x] == 'F' or track_matrix[y,x] == 'S'):
                valid_positions.append((y,x))
        return valid_positions


    def get_possible_actions():
        return get_possible_accelerations()

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


    def get_starting_coordinates():
        starting_positions = []
        for y, x in np.ndindex(track_matrix.shape):
            if(track_matrix[y,x] == 'S'):
                starting_positions.append((y, x))
        return starting_positions[0]

    def get_starting_state():
        start_coord = get_starting_coordinates()
        start_velocity = (0,0)
        start_state = start_coord + start_velocity
        return start_state


    def get_finish_coordinates():
        finish_positions = []
        for y, x in np.ndindex(track_matrix.shape):
            if(track_matrix[y,x] == 'F'):
                finish_positions.append((y, x))
        return finish_positions


    def safe_print(*args):
        if(lock):
            lock.acquire()
            print(" ".join(map(str, args)))
            sys.stdout.flush()
            lock.release()
        else:
            print(" ".join(map(str, args)))

    def manhattan_distance(p1, p2):
        y1 = p1[0]
        x1 = p1[1]
        y2 = p2[0]
        x2 = p2[1]
        return abs(x2 - x1) + abs(y2-y1)


    def swap(a, b):
        # challenged myself, swapping without a tmp variable.
        # a = a + b
        # b = b + a
        # a = a - b
        # b = b + 2*a
        # a = -1 * a
        # return a, b
        return b, a

    def get_path(p_start, p_end):
        path = set([])

        y_0 = p_start[0]
        y_1 = p_end[0]
        x_0 = p_start[1]
        x_1 = p_end[1]

        # Cases where x or y are constant
        if x_0 == x_1 or y_0 == y_1:
            if(x_0 == x_1):
                if (y_0 > y_1):
                    y_0, y_1 = swap(y_0, y_1)
                for y in range(y_0, y_1 + 1):
                    path.add((y, x_0))
            if(y_0 == y_1):
                if (x_0 > x_1):
                    x_0, x_1 = swap(x_0, x_1)
                for x in range(x_0, x_1 + 1):
                    path.add((y_0, x))
        else:
            # Case for non-constant x, y
            m = (y_1 - y_0) / (x_1 - x_0)
            b = y_0 - m * x_0

            # Assert x0 < x1 for iterating over values
            if(x_0 > x_1):
                x_0, x_1 = swap(x_0, x_1)
            for x in range(x_0, x_1+1):
                y = m * x + b
                path.add((int(np.round(y)), x))

            # Assert y0 < y1 for iterating over values
            if(y_0 > y_1):
                y_0, y_1 = swap(y_0, y_1)
            for y in range(y_0, y_1+1):
                x = (y - b) / m
                path.add((y, int(np.round(x))))
        return path


    def crashed_through_wall(p_start, p_end):
        points_to_check = get_path(p_start, p_end)
        for point in points_to_check:
            if(track_matrix[point] == "#"):
                return True
        return False

    def get_crash_location(p_start, p_end):
        vehicle_path = get_path(p_start, p_end)

        # need to find points closest to the starting point, just before the wall
        distances_from_start = {}
        for point in vehicle_path:
            distances_from_start[point] = manhattan_distance(p_start, point)
        sorted_points = sorted(distances_from_start, key=lambda k: distances_from_start[k])

        prev_point = p_start
        for point in sorted_points:
            if(track_matrix[point] == '#'):
                return prev_point



    def is_in_finish(position):
        y = position[0]
        x = position[1]
        finish = track_matrix[y,x] == 'F'
        return finish


    def is_outside_bounds(position):
        y = position[0]
        x = position[1]

        if(y >= track_matrix.shape[0]):
            return True
        if(x >= track_matrix.shape[1]):
            return True
        if ( y < 0 or x < 0):
            return True
        return False


    def reward(state, action, failed_action=None):
        new_state, new_position, new_velocity, crashed, out_of_bounds = update_state(state, action, failed_action)

        x_0 = state[1]
        y_0 = state[0]
        p0 = (y_0, x_0)

        previous_finish_distances = 0
        new_finish_distances = 0
        finish_coordinates = get_finish_coordinates()

        for finish_point in finish_coordinates:
            previous_finish_distances += manhattan_distance(p0, finish_point)
            new_finish_distances += manhattan_distance(new_position, finish_point)

        # We want reward to be positive
        total_reward = (previous_finish_distances - new_finish_distances) / len(finish_coordinates)

        if( crashed or out_of_bounds ):
            total_reward = -20

        if is_in_finish(new_position):
            total_reward += 10

        return total_reward


    def value_iteration_experiment():
        # initialize states
        v = {}
        q = {}
        pi = {}
        valid_states = get_possible_valid_states()
        actions = get_possible_actions()

        v[0] = {}
        for state in valid_states:
            v[0][state] = random.random() * 10 - 5 # Arbitrary random values between -5, 5

        t = 0
        while True:
            t = t+1
            q[t] = {}
            pi[t] = {}
            v[t] = {}

            # Monitor progress
            # total_iterations = len(valid_states) * len(actions)
            # iteration = 0

            for s in valid_states:
                q[t][s] = {}
                for a in actions:

                    new_state_s = update_state(s, a, failed_action=True)[0]
                    new_state_f = update_state(s, a, failed_action=False)[0]
                    value_f = discount_factor * v[t-1][new_state_s]
                    value_s = discount_factor * v[t-1][new_state_f]

                    expected_value_f = value_s * get_probability_of_action_success()
                    expected_value_s = value_f * (1-get_probability_of_action_success())
                    expected_value = expected_value_s + expected_value_f

                    expected_reward_f = reward(s, a, failed_action=True) * (1-get_probability_of_action_success())
                    expected_reward_s = reward(s, a, failed_action=False) * get_probability_of_action_success()
                    expected_reward = expected_reward_s + expected_reward_f

                    q[t][s][a] = expected_reward + expected_value * discount_factor

                    # iteration+=1
                    # print(f"{iteration}/{total_iterations}")

                pi[t][s] = max(q[t][s], key=q[t][s].get)
                v[t][s] = q[t][s][pi[t][s]]


            differences = abs(np.subtract(list(v[t].values()), list(v[t-1].values())))

            safe_print(f"t={t}, Vdiff_max={max(differences)}, threshold={convergence_threshold} : {track_name}/{algorithm_name}/{crash_scenario_name}")

            if(all(i < convergence_threshold for i in differences)):
                break

        current_state = get_starting_state()
        current_position = get_position(current_state)
        travel_positions = [current_position]
        while(track_matrix[current_position] != 'F'):
            current_state, current_position, nv, crashed, out_of_bounds = update_state(current_state, pi[t][current_state])
            travel_positions.append(current_position)
            if(crashed or out_of_bounds):
                print("Crashed during test!")
                exit(95)

        visualize_board(optimal_path=travel_positions, optimal_policy=pi)




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
                        choices=['near-wall', 'restart'],
                        type=str,
                        default=['near-wall', 'restart'])
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