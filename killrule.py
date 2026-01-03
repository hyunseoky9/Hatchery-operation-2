import numpy as np

def killrule(scores, epi, env):
    """A function to determine whether to stop training early based on performance.

    Args:
        inttestscores (list): List of average test scores from recent evaluations.
        envID (int): Identifier for the environment being trained.

    Returns:
        bool: True if training should be stopped, False otherwise.
    """
    kill = False
    scores = np.array(scores)
    if env.envID == 'Hatchery3.3.7':
        if env.c == 2.5:
            # rule 1: if the performance hasn't improved by more than 2 points in the last 500 episodes.
            if len(scores) >= 6:
                diff = scores[-5:] - scores[-6:-1]
                if np.all(diff < 2):
                    print(f'violated rule1 at episode {epi}, if no improvement in last 5 tests')
                    kill = True
            # rule 2: by episode 600, if the run never exceeded 50, kill. 
            if epi == 600:
                if np.all(scores < 50):
                    print(f'violated rule2 at episode {epi}, never exceeded 50 by episode 600')
                    kill = True
            # rule 3: by episode 1000, if the run never exceeded 68, kill.
            if epi == 1000:
                if np.all(scores < 68):
                    print(f'violated rule3 at episode {epi}, never exceeded 68 by episode 1000')
                    kill = True
        elif env.c == 1.5:
            if len(scores) >= 3:
                # rule 1: if the performance is below 11 for the last 3 tests, kill.
                if np.all((scores[-3:]<11)):
                    print(f'violated rule1 at episode {epi}, if no improvement in last 3 tests')
                    kill = True
                # rule 2: if the performance is stuck between 16 and 20 for the last 3 tests, kill.
                if np.all((scores[-3:]>16) & (scores[-3:]<20)):
                    print(f'violated rule2 at episode {epi}, if stuck between 30 and 35 in last 3 tests')
                    kill = True
        elif env.c == 2:
            # rule 1: if the performance hasn't improved by more than 2 points in the last 500 episodes.
            if len(scores) >= 6:
                diff = scores[-5:] - scores[-6:-1]
                if np.all(diff < 2):
                    print(f'violated rule1 at episode {epi}, if no improvement in last 5 tests')
                    kill = True
            # rule 2: by episode 600, if the run never exceeded 45, kill.
            if epi == 500:
                if np.all(scores < 45):
                    print(f'violated rule2 at episode {epi}, never exceeded 45 by episode 600')
                    kill = True
            

        else:
            return kill
    return kill            
