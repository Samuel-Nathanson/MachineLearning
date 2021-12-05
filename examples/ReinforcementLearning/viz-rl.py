
import matplotlib.pyplot as plt
import numpy as np
experimental_results = {}

ALGO_KEY_IDX = 0
TRACK_KEY_IDX = 1
DISCOUNT_KEY_IDX = 2
LEARNING_RATE_KEY_IDX = 3
EPSILON_KEY_IDX = 4
CRASH_SCENARIO_KEY_IDX = 5
EPISODE_KEY_IDX = 6

CRASH_IDX = 0
ACTIONS_IDX = 1

with open('runs-2.data', 'r') as data:
    lines = data.readlines()

    for line in lines:
        values = line.split(",")
        algorithm_name = values[0]
        track_name = values[1]
        discount_factor = values[2]
        learning_rate = values[3]
        initial_epsilon = values[4]
        crash_scenario_name = values[5]
        episode = values[6]
        num_crashes = values[7]
        actions = values[8]

        experiment = (algorithm_name, track_name, discount_factor, learning_rate, initial_epsilon, crash_scenario_name, episode)
        results = [num_crashes, actions]

        experimental_results[experiment] = results

track_names = np.unique([x[TRACK_KEY_IDX] for x in experimental_results.keys()])
algorithm_names = np.unique([x[ALGO_KEY_IDX] for x in experimental_results.keys()])
epsilon_values = np.unique([x[DISCOUNT_KEY_IDX] for x in experimental_results.keys()])
learning_rate_values = np.unique([x[LEARNING_RATE_KEY_IDX] for x in experimental_results.keys()])
discount_values = np.unique([x[DISCOUNT_KEY_IDX] for x in experimental_results.keys()])
crash_scenario_values = np.unique([x[CRASH_SCENARIO_KEY_IDX] for x in experimental_results.keys()])

line_colors = ['b', 'g', 'r', 'c', 'm']


track = track_names[0]
algorithm = algorithm_names[0]
epsilon = epsilon_values[0]
learning_rate = learning_rate_values[0]
discount = discount_values[0]
crash_scenario = crash_scenario_values[0]

def viz_epsilon():
    #  hold all variables constant except for epsilon and episode #
    # filter for keys with learning_rate[0], discount[0], crash_scenario[0], track_name[0], algorithm[0]

    iv = "Epsilon"
    # check if we have all results
    filtered_results = {
        key:value for (key, value) in experimental_results.items() if
                            key[LEARNING_RATE_KEY_IDX] == learning_rate if
                             key[CRASH_SCENARIO_KEY_IDX] == crash_scenario if
                             key[DISCOUNT_KEY_IDX] == discount if
                             key[TRACK_KEY_IDX] == track if
                             key[ALGO_KEY_IDX] == algorithm
    }

    legend_elements = []

    # Create two subplots and unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.set_title('# Actions')
    ax2.set_title('# Crashes')
    ax1.set_ylabel('# Actions')
    ax2.set_ylabel('# Crashes')
    ax1.set_xlabel('# Episodes')
    ax2.set_xlabel('# Episodes')

    plt.suptitle(f'Experiment with variable epsilon: track={track}, algorithm={algorithm}, discount={discount}, crash_scenario={crash_scenario}, learning_rate={learning_rate}')

    # Create four polar axes and access them through the returned array

    for i, epsilon in enumerate(epsilon_values):
        filtered_results_epsilon = {
            key:value for (key, value) in filtered_results.items() if key[EPSILON_KEY_IDX] == epsilon
        }
        x_episodes = sorted(np.unique([int(x[EPISODE_KEY_IDX]) for x in filtered_results_epsilon.keys()]))

        if(len(x_episodes) == 0):
            continue

        make_key = lambda episode: (algorithm, track, discount, learning_rate, epsilon, crash_scenario, episode)

        y_crashes = [int(filtered_results_epsilon[make_key(str(e))][CRASH_IDX]) for e in x_episodes]
        y_actions = [int(filtered_results_epsilon[make_key(str(e))][ACTIONS_IDX]) for e in x_episodes]

        m_actions, b_actions = np.polyfit(x_episodes, y_actions, 1)
        m_crashes, b_crashes = np.polyfit(x_episodes, y_crashes, 1)
        plt.legend()
        ax1.plot(x_episodes, y_crashes, line_colors[i])
        ax2.plot(x_episodes, y_actions, line_colors[i])
        ax1.plot(x_episodes, m_crashes*np.array(x_episodes) + b_crashes, f"{line_colors[i]}--")
        ax2.plot(x_episodes, m_actions*np.array(x_episodes) + b_actions, f"{line_colors[i]}--")

        legend_elements.append(plt.Line2D([0], [0], color=line_colors[i], lw=4, label=f"{iv}={epsilon}"))
        legend_elements.append(plt.Line2D([0], [0], color=line_colors[i], linestyle='--', lw=1, label=f"Linear Regression for {iv}={epsilon}"))


    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.show()


def viz_learning_rate():
    #  hold all variables constant except for epsilon and episode #
    # filter for keys with learning_rate[0], discount[0], crash_scenario[0], track_name[0], algorithm[0]

    # check if we have all results

    filtered_results = {
        key: value for (key, value) in experimental_results.items() if
        # key[LEARNING_RATE_KEY_IDX] == learning_rate if
        key[EPSILON_KEY_IDX] == epsilon if
        key[CRASH_SCENARIO_KEY_IDX] == crash_scenario if
        key[DISCOUNT_KEY_IDX] == discount if
        key[TRACK_KEY_IDX] == track if
        key[ALGO_KEY_IDX] == algorithm
    }

    for i, learning_rate in enumerate(learning_rate_values):
        filtered_results_2 = { # easier re-use
            key: value for (key, value) in filtered_results.items() if key[LEARNING_RATE_KEY_IDX] == learning_rate
        }
        x_episodes = sorted(np.unique([int(x[EPISODE_KEY_IDX]) for x in filtered_results_2.keys()]))

        make_key = lambda episode: (algorithm, track, discount, learning_rate, epsilon, crash_scenario, episode)

        y_crashes = [int(filtered_results_2[make_key(str(e))][CRASH_IDX]) for e in x_episodes]
        y_actions = [int(filtered_results_2[make_key(str(e))][ACTIONS_IDX]) for e in x_episodes]

        print(y_crashes)
        plt.plot(x_episodes, y_crashes)
        plt.show()

def viz_discount_factor():
    pass

def viz_crash_scenario():
    pass

viz_epsilon()


