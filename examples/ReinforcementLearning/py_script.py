
import os
import subprocess
import sys
import time
import signal

epsilons = [0.1, 0.25, 0.5, 0.75, 0.9]
crash_scenarios = ['near-wall', 'restart']
algorithms = ['q-learning', 'SARSA']
learning_rates = [0.001, 0.01, 0.1, 0.5, 0.75]
discount_factors = [0.01, 0.1, 0.5, 0.75, 0.99]

for e in epsilons:
    for c in crash_scenarios:
        for alg in algorithms:
            for l in learning_rates:
                for d in discount_factors:
                    completed = False
                    proc = None
                    while not completed:
                        try:
                            proc = subprocess.Popen(["C:\\Users\\Samuel Nathanson\\PycharmProjects\\MachineLearning\\venv\\Scripts\\python.exe",
                                            "C:\\Users\\Samuel Nathanson\\PycharmProjects\\MachineLearning\\examples\\ReinforcementLearning\\Racetrack.py",
                                            "-algorithms",
                                            alg,
                                            "-learning_rate",
                                            str(l),
                                            "-discount-factor",
                                            str(d),
                                            "-epsilon",
                                            str(e),
                                            "-crash-scenarios",
                                            str(c)])
                            time.sleep(400)
                            if(proc.poll() is None):
                                os.kill(proc.pid, signal.CTRL_C_EVENT)
                                continue
                            completed = True
                        except:
                            continue


