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

MAX_VELOCITY_MAGNITUDE = 5
MIN_VELOCITY_MAGNITUDE = -5

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
    def visualize_board(optimal_path, title=""):
        import matplotlib.pyplot as plt
        ax = plt.figure()

        num_crashses = 0

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

        for i, state_t in enumerate(optimal_path):
            new_position = state_t["position"]
            new_velocity = state_t["velocity"]
            will_crash = state_t["will_crash"]
            acceleration = state_t["acceleration"]
            will_go_out_of_bounds = state_t["will_go_out_of_bounds"]
            x = new_position[1]
            y = new_position[0]

            marker = 'H'
            position_color = "blue"
            velocity_color = "green"
            acceleration_color = "purple"
            crash_color = "red"
            if(will_crash or will_go_out_of_bounds):
                num_crashses +=1
                velocity_color = crash_color
                acceleration_color = crash_color

            # Plot position
            plt.plot([x], [y], marker=marker, color=position_color, markersize=5)
            # Plot velocity in color
            plt.arrow(x,
                      y,
                      new_velocity[1],
                      new_velocity[0],
                      color=velocity_color,
                      length_includes_head=True,
                      overhang=1,
                      width=0.01,
                      shape="left",
                      head_width=0.2, head_length=0.2 )
            # Plot acceleration
            plt.arrow(x,
                      y,
                      acceleration[1],
                      acceleration[0],
                      color=acceleration_color,
                      length_includes_head=True,
                      overhang=1,
                      width=0.01,
                      shape="right",
                      head_width=0.1, head_length=0.2,
                      head_starts_at_zero=True)

        ## Add legend, title, and axes label, save

        plt.ylabel("y-coordinate")
        plt.xlabel("x-coordinate")
        plt.title(title)

        t = int(title.split(' ')[2])

        # plt.figure(figsize=(1.92, 1.08), dpi=100)
        figure = plt.gcf()
        figure.set_size_inches(19.2, 10.8)

        legend_elements = [plt.Line2D([0], [0], color='blue', lw=4, label='Position'),
                           plt.Line2D([0], [0], color='green', lw=4, label='Velocity'),
                           plt.Line2D([0], [0], color='purple', lw=4, label='Acceleration'),
                           plt.Line2D([0], [0], color='red', lw=4, label='Crash occurred')]

        stats_elements = [plt.Line2D([0], [0], marker='o', color='yellow', label='Training Episodes',
                                     markerfacecolor='yellow', markersize=15),
                          plt.Line2D([0], [0], marker='o', color='blue', label=f'# Actions Required: {len(optimal_path)}',
                                    markerfacecolor='blue', markersize=15),
                          plt.Line2D([0], [0], marker='o', color='red', label=f'# Crashses: {num_crashses}',
                                     markerfacecolor='red', markersize=15)]

        # Create the figure
        ax.legend(handles=legend_elements, loc='upper right')
        ax.legend(handles=stats_elements, loc='upper left')


        plt.savefig(f"results/{algorithm_name}_{track_name}_{t}.png", dpi=100)
        plt.clf()
        plt.close('all')


    def race_with_policy(policy, visualize=False, title=""):
        start_state = get_starting_state()
        state_path = []

        state_0 = {
            "state": start_state,
            "position": get_position(start_state),
            "velocity": (0,0),
            "crashed": False,
            "out_of_bounds": False
        }
        state_path.append(state_0)

        while True:
            previous_state = state_path[-1]["state"]
            new_state, new_position, new_velocity, crashed, out_of_bounds = update_state(previous_state, policy[previous_state])
            state_path[-1]["will_crash"] = crashed
            state_path[-1]["will_go_out_of_bounds"] = out_of_bounds

            state_t = {
                "state": new_state,
                "position": new_position,
                "velocity": new_velocity,
                "crashed": crashed,
                "out_of_bounds": out_of_bounds,
                "acceleration": policy[previous_state],
                "will_crash": False, # Adding this to prevent exceptions on final state plotting
                "will_go_out_of_bounds": False #
            }

            state_path.append(state_t)

            if (track_matrix[new_position] == 'F'):
                break

        if(visualize):
            visualize_board(optimal_path=state_path,  title=title)


    def race_with_q_table(q, visualize=False, title=""):

        # Epsilon Greedy gradually decreases epsilon and uses a temperature variables to switch from exploration
        # To exploitation
        # We want to promote exploitation once we have a well-trained q-table, thus we should artificially simulate
        # exploitation
        EXPLOITATION_PROMOTING_EPISODE_CONSTANT = 100

        start_state = get_starting_state()
        state_path = []

        state_0 = {
            "state": start_state,
            "position": get_position(start_state),
            "velocity": (0,0),
            "crashed": False,
            "out_of_bounds": False
        }
        state_path.append(state_0)

        while True:
            previous_state = state_path[-1]["state"]
            action = choose_action_epsilon_greedy(q, previous_state, iteration=EXPLOITATION_PROMOTING_EPISODE_CONSTANT)
            new_state, new_position, new_velocity, crashed, out_of_bounds = update_state(previous_state, action)
            state_path[-1]["will_crash"] = crashed
            state_path[-1]["will_go_out_of_bounds"] = out_of_bounds
            state_path[-1]["acceleration"] = action

            state_t = {
                "state": new_state,
                "position": new_position,
                "velocity": new_velocity,
                "crashed": crashed,
                "out_of_bounds": out_of_bounds,
                "will_crash": False, # Adding this to prevent exceptions on final state plotting
                "will_go_out_of_bounds": False, #
                "acceleration": (0,0)
            }

            state_path.append(state_t)

            if (track_matrix[new_position] == 'F'):
                break

        if(visualize):
            visualize_board(optimal_path=state_path, title=title)


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



        new_state = new_position + new_velocity

        return new_state, new_position, new_velocity, crashed, out_of_bounds


    def get_reward(previous_position, new_position, crashed, out_of_bounds):
        previous_finish_distances = 0
        new_finish_distances = 0
        finish_coordinates = get_finish_coordinates()

        total_reward = -5

        # Reward for moving closer to the finish line
        for finish_point in finish_coordinates:
            previous_finish_distances += manhattan_distance(previous_position, finish_point)
            new_finish_distances += manhattan_distance(new_position, finish_point)
        total_reward += (previous_finish_distances - new_finish_distances) / len(finish_coordinates)

        # Penalty for crashing
        if( crashed or out_of_bounds ):
            total_reward = -1000

        if is_in_finish(new_position):
            total_reward += 500

        return total_reward


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
        '''
        Thread-safe printing to console
        '''
        if(lock):
            lock.acquire()
            print(" ".join(map(str, args)))
            sys.stdout.flush()
            lock.release()
        else:
            print(" ".join(map(str, args)))


    def manhattan_distance(p1, p2):
        '''
        Manhattan distance between two points
        '''
        y1 = p1[0]
        x1 = p1[1]
        y2 = p2[0]
        x2 = p2[1]
        return abs(x2 - x1) + abs(y2-y1)


    def swap(a, b):
        # challenged myself to swap without a tmp variable.
        # a = a + b
        # b = b + a
        # a = a - b
        # b = b + 2*a
        # a = -1 * a
        # return a, b
        return b, a


    def get_path(p_start, p_end):
        '''
        Returns the path [(y0, x0), ... (yN, xN)] between two points
        .#...
        ..#..
        ...#.
        ....#
        '''
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
        '''
        Given a starting and ending point, detects whether or not a wall ('#') intersects the line segment between
        the starting and ending point.
        '''
        points_to_check = get_path(p_start, p_end)
        for point in points_to_check:
            if(track_matrix[point] == "#"):
                return True
        return False


    def get_crash_location(p_start, p_end):
        '''
        Given a starting point and ending point, returns the point before the crash occurred.
        '''
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
        '''
        Detects whether the given position is at the finish line
        # TODO: *Crosses* finish line
        '''
        y = position[0]
        x = position[1]
        finish = track_matrix[y,x] == 'F'
        return finish


    def is_outside_bounds(position):
        '''
        Detects whether the position is outside of the bounds of the track
        '''
        y = position[0]
        x = position[1]

        if(y >= track_matrix.shape[0]):
            return True
        if(x >= track_matrix.shape[1]):
            return True
        if ( y < 0 or x < 0):
            return True
        return False


    def reward(state, action, failed_action: bool=None):
        '''
        Reward function, given a state and action.
        :param failed_action: Manually detect whether or not a failed action occurred. Default is none, which means this
        will be randomly selected
        '''
        new_state, new_position, new_velocity, crashed, out_of_bounds = update_state(state, action, failed_action)
        return get_reward(previous_position=get_position(state),
                          new_position=new_position,
                          crashed=crashed,
                          out_of_bounds=out_of_bounds)


    def value_iteration_experiment():
        '''
        Value Iteration Experiment
        '''
        # initialize list of valid states and actions
        valid_states = get_possible_valid_states()
        actions = get_possible_actions()

        # initialize tables
        v = {} # value table
        q = {}
        pi = {} # optimal policy table

        # initialize value table to arbitrary random values
        v[0] = {}
        for state in valid_states:
            v[0][state] = random.random() * 10 - 5 # Arbitrary random values between -5, 5

        t = 0
        while True:
            t = t+1
            q[t] = {}
            pi[t] = {}
            v[t] = {}

            # Update the value table, q-table, and policy table for state/action pairs
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

            # Visualize agent behavior after each iteration
            visualization_title = f"Policy after {t} iterations for experiment: {algorithm_name} on {track_name} with crash scenario {crash_scenario_name}"
            race_with_policy(pi[t], visualize=True, title=visualization_title)

            # Detect convergence
            differences = abs(np.subtract(list(v[t].values()), list(v[t-1].values())))
            safe_print(f"t={t}, Vdiff_max={max(differences)}, threshold={convergence_threshold} : {track_name}/{algorithm_name}/{crash_scenario_name}")
            if(all(i < convergence_threshold for i in differences)):
                break


    def choose_action_epsilon_greedy(q: dict, state: list, iteration: int):
        '''
        Given a state and Q-Table, choose an action using the Epsilon-Greedy Policy.
        '''
        actions = get_possible_actions()

        # Iteration (episode) number will be used to control both temperature and epsilon.
        temperature = iteration + 1
        initial_epsilon = 0.1

        # Gradually decrease epsilon with more episodes
        epsilon = np.power(initial_epsilon, 1 + iteration / 10)

        # With probability epsilon, choose a random action with uniform probability. Otherwise exploit Q-Values.
        if(random.random() < epsilon):
            #explore
            return actions[random.randint(0,len(actions) - 1)]
        else:
            #exploit
            # Choose action at random, with weights given by probabilities
            q_values = [q[state][action] for action in actions]
            softmax_denominator = np.sum([np.exp(q[state][action])/temperature for action in actions])
            softmax_numerators = [np.exp(q_value)/temperature for q_value in q_values]
            action_probabilities = np.divide(softmax_numerators, softmax_denominator)
            chosen_action = random.choices(actions, action_probabilities)
            return chosen_action[0]


    def q_learning_experiment():

        # Initialize q-table, possible states & actions.
        q = {}
        valid_states = get_possible_valid_states()
        actions = get_possible_actions()

        # Initialize Q-Values to random values
        for state in valid_states:
            q[state] = {}
            for action in actions:
                q[state][action] = random.random() * 0.001 # Arbitrary random values


        # Set number of full episodes (Start->Finish) to run.
        episodes = 150
        for episode_num in range(0, episodes):
            state = get_starting_state()

            while True:
                # Loop until the agent finishes the race

                # Choose an action using an epsilon greedy policy
                action = choose_action_epsilon_greedy(q, state, episode_num)

                # Get new state and reward using action derived from Epsilon greedy
                new_state, new_position, new_velocity, crashed, out_of_bounds = update_state(state, action)
                reward = get_reward(previous_position=get_position(state),
                                  new_position=new_position,
                                  crashed=crashed,
                                  out_of_bounds=out_of_bounds)

                # Update Q-Values
                best_future_action = max(q[new_state], key=q[new_state].get)
                q[state][action] += learning_rate * ( reward + discount_factor * q[state][best_future_action] - q[state][action])

                # Update state
                state = new_state

                # Detect whether or not the agent finished
                if is_in_finish(new_position):
                    break

            safe_print(f"Policy after {episode_num} episodes for experiment: {algorithm_name} on {track_name} with crash scenario {crash_scenario_name}")
            # Visualize resulting agent behavior at the end of each episode
            visualization_title = f"Policy after {episode_num} episodes for experiment: {algorithm_name} on {track_name} with crash scenario {crash_scenario_name}"
            race_with_q_table(q, visualize=True, title=visualization_title)


    def sarsa_experiment():

        # Initialize q-table, possible states & actions.
        q = {}
        valid_states = get_possible_valid_states()
        actions = get_possible_actions()

        # Initialize Q-Values to random values
        for state in valid_states:
            q[state] = {}
            for action in actions:
                q[state][action] = random.random() * 0.001 # Arbitrary random values


        # Set number of full episodes (Start->Finish) to run.
        episodes = 150
        for episode_num in range(0, episodes):
            state = get_starting_state()

            while True:
                # Loop until the agent finishes the race

                # Choose an action using an epsilon greedy policy
                action = choose_action_epsilon_greedy(q, state, episode_num)

                # Get new state and reward using action derived from Epsilon greedy
                new_state, new_position, new_velocity, crashed, out_of_bounds = update_state(state, action)
                reward = get_reward(previous_position=get_position(state),
                                  new_position=new_position,
                                  crashed=crashed,
                                  out_of_bounds=out_of_bounds)

                # Update Q-Values
                future_action = choose_action_epsilon_greedy(q, new_state, episode_num)
                q[state][action] += learning_rate * ( reward + discount_factor * q[new_state][future_action] - q[state][action])

                # Update state
                state = new_state

                # Detect whether or not the agent finished
                if is_in_finish(new_position):
                    break

            safe_print(f"Policy after {episode_num} episodes for experiment: {algorithm_name} on {track_name} with crash scenario {crash_scenario_name}")
            # Visualize resulting agent behavior at the end of each episode
            visualization_title = f"Policy after {episode_num} episodes for experiment: {algorithm_name} on {track_name} with crash scenario {crash_scenario_name}"
            race_with_q_table(q, visualize=True, title=visualization_title)


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
                        choices=['value-iteration', 'q-learning', 'SARSA'],
                        type=str,
                        default=['value-iteration', 'q-learning', 'SARSA'])
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