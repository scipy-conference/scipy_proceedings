
# The 'run' function, which is your existing computational model
def example_run(die1, die2):
    sum = die1 + die2
    return (sum, )

# The 'preprocess' function grabs the random input values for each case and
# structures it with any other data in the format your 'run' function expects
def example_preprocess(case):
    die1 = case.invals['die1'].val
    die2 = case.invals['die2'].val
    return (die1, die2)

# The 'postprocess' function takes the output from your 'run' function and
# saves off the outputs for each case
def example_postprocess(case, sum):
    case.addOutVal(name='Sum', val=sum)
    case.addOutVal(name='Roll Number', val=case.ncase)
    return None


import monaco as mc
from scipy.stats import randint
import matplotlib.pyplot as plt

def main():
    fcns = {'preprocess' : example_preprocess,
            'run'        : example_run,
            'postprocess': example_postprocess}
    
    ndraws = 1024
    seed = 123456  # Recommended for repeatability

    # Initialize the simulation
    sim = mc.Sim(name='Dice Roll', singlethreaded=True, savecasedata=False,
                 ndraws=ndraws, fcns=fcns, seed=seed)

    # Generate the input variables
    sim.addInVar(name='die1', dist=randint, distkwargs={'low': 1, 'high': 7})
    sim.addInVar(name='die2', dist=randint, distkwargs={'low': 1, 'high': 7})

    # Run the Simulation
    sim.runSim()

    # Calculate the mean and 5-95th percentile statistics for the dice sum
    sim.outvars['Sum'].addVarStat('mean')
    sim.outvars['Sum'].addVarStat('percentile', {'p':[0.95, 0.05]})
    
    # Plots a histogram of the dice sum
    fig, axs = plt.subplots(3,1)
    mc.plot(sim.outvars['Sum'], plotkwargs={'bins':10}, ax=axs[0])
    
    # Creates a scatter plot of the dum vs the roll number, showing randomness
    mc.plot(sim.outvars['Sum'], sim.outvars['Roll Number'], ax=axs[1])
    axs[1].get_shared_x_axes().join(axs[0], axs[1])
 
    # Calculate the sensitivity of the dice sum to each of the input variables
    sim.calcSensitivities('Sum')
    sim.outvars['Sum'].plotSensitivities(ax=axs[2])
   
    plt.show(block=True)
    

if __name__ == '__main__':
    main()
