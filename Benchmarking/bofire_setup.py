
import bofire 
import botorch
import torch
import summit
import numpy as np
import pandas as pd
import time
import os
import pickle
from setup_files_alt import evaluate_candidates
import multiprocessing
import importlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.exceptions
import bofire.strategies.api as strategies
from bofire.data_models.domain.api import Domain
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
    CategoricalInput,
    CategoricalDescriptorInput,)

from pprint import pprint as pp
from bofire.data_models.objectives.api import MinimizeObjective, MaximizeObjective
from bofire.data_models.strategies.api import MoboStrategy
from bofire.data_models.acquisition_functions.api import qEHVI, qLogEHVI
from bofire.data_models.strategies.api import SoboStrategy
from bofire.data_models.acquisition_functions.api import qEI
from bofire.data_models.strategies.api import (
    RandomStrategy as RandomStrategyModel,
)

from initialising_points import bofire_initial_conditions
#from benchmarking_alt import bofire_initial_conditions

# We wish the temperature of the reaction to be between 30 and 110 °C
temperature_feature = ContinuousInput(
    key="Temperature", bounds=[30.0, 110.0], unit="°C"
)

# Catalyst Loading
catalyst_loading_feature = ContinuousInput(
    key="Catalyst Loading", bounds=[0.5, 2], unit="%"
)

# Residence Time
residence_time_feature = ContinuousInput(
    key="Residence Time", bounds=[1 * 60, 10 * 60], unit="minutes"
)

# Catalyst choice
catalyst_feature = CategoricalInput(
    key="Catalyst",
    categories=[
        "P1-L1",
        "P2-L1",
        "P1-L2",
        "P1-L3",
        "P1-L4",
        "P1-L5",
        "P1-L6",
        "P1-L7",
    ],
)

# gather all individual features
input_features = Inputs(
    features=[
        temperature_feature,
        catalyst_loading_feature,
        residence_time_feature,
        catalyst_feature,
    ]
)

name_map = {
    "Catalyst Loading": "catalyst_loading",
    "Residence Time": "t_res",
    "Temperature": "temperature",
    "Catalyst": "catalyst",
    "Yield": "yld",
}
candidates = pd.DataFrame(
    {
        "Catalyst Loading": [0.498],
        "Residence Time": [600],
        "Temperature": [30],
        "Catalyst": ["P1-L3"],
    }
).rename(columns=name_map)
#print(candidates)
emulator = summit.get_pretrained_reizman_suzuki_emulator(case=1)
conditions = summit.DataSet.from_df(candidates)
results = emulator.run_experiments(conditions, rtn_std=True).rename(
    columns=dict(zip(name_map.values(), name_map.keys())),
)
experiments = pd.DataFrame(
    {
        "Catalyst Loading": results["Catalyst Loading"],
        "Residence Time": results["Residence Time"],
        "Temperature": results["Temperature"],
        "Catalyst": results["Catalyst"],
        "Yield": results["Yield"],
        "valid_Yield": 1,
    }
)


max_objective = MaximizeObjective(w=1.0)
#min_objective = MinimizeObjective(w=1.0, bounds=[0, 200])
max_objective_2 = MaximizeObjective(w=1.0, bounds=[0, 200])


yield_feature = ContinuousOutput(key="Yield", objective=max_objective)
ton_feature = ContinuousOutput(key="TON", objective=max_objective_2)
# create an output feature
output_features = Outputs(features=[yield_feature, ton_feature])
domain = Domain(
    inputs=input_features,
    outputs=output_features,
)
# a multi objective BO strategy

#qExpectedImprovement = qEHVI()
qLogExpectedImprovement = qLogEHVI()
mobo_strategy_data_model = MoboStrategy(
    domain=domain,
    acquisition_function=qLogExpectedImprovement,
)

# map the strategy data model to the actual strategy that has functionality
mobo_strategy = strategies.map(mobo_strategy_data_model)


# a random strategy
#random_strategy_model = RandomStrategyModel(domain=domain)
# we have to provide the strategy with our optimization problem so it knows where to sample from = initial 5 random samples?
#random_strategy = strategies.map(random_strategy_model)
#candidates = random_strategy.ask(5)
#experiments = evaluate_candidates(candidates)
'''
candidates = bofire_initial_conditions
experiments = evaluate_candidates(candidates)
mobo_strategy.tell(experiments, replace=True, retrain=True)



experimental_budget = 5 # edit as needed
i = 0
done = False

results_df = pd.DataFrame(columns=["Catalyst Loading", "Residence Time", "Temperature", 
                                   "Catalyst", "Yield", "valid_Yield", "TON", "valid_TON"])

while not done:
    i += 1
    t1 = time.time()
    # ask for a new experiment
    new_candidate = mobo_strategy.ask(1)
    new_experiment = evaluate_candidates(new_candidate)
    mobo_strategy.tell(new_experiment)
    print(f"Iteration took {(time.time()-t1):.2f} seconds")
    # inform the strategy about the new experiment
    # experiments = pd.concat([experiments,new_experiment],ignore_index=True)

    # Add new results to DataFrame
    results_df = pd.concat([results_df, new_experiment], ignore_index=True)
    
    # Calculate cumulative max for Yield and TON
    results_df["Cumulative_Max_Yield"] = results_df["Yield"].cummax()
    results_df["Cumulative_Max_TON"] = results_df["TON"].cummax()
    
    print(f"Iteration {i} took {(time.time() - t1):.2f} seconds")
    print("Current Results:")
    print(results_df[["Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])
    

    if i > experimental_budget:
        done = True


results= mobo_strategy.experiments
print(mobo_strategy.experiments)

print("results incl. cumulative max:", results_df[["Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])
'''

#with open('benchmark_results_bofire.pkl', 'wb') as f:
        #pickle.dump(results, f)

emulator = summit.get_pretrained_reizman_suzuki_emulator(case=1)


def run_mobo_optimization(emulator: summit.benchmarks.experimental_emulator.ReizmanSuzukiEmulator, mobo_strategy,  bofire_initial_conditions, experimental_budget=5):
    """
    Runs a multi-objective Bayesian optimization loop.
    
    Parameters:
        mobo_strategy: The optimization strategy object with ask and tell methods.
        evaluate_candidates: Function to evaluate candidate experiments.
        bofire_initial_conditions: Initial conditions for optimization.
        experimental_budget: Number of iterations for the optimization loop.
        
    Returns:
        results_df: DataFrame containing experiment results and cumulative metrics.
        results: Final experiments stored in the strategy object.
    """


    # We wish the temperature of the reaction to be between 30 and 110 °C
    temperature_feature = ContinuousInput(
    key="Temperature", bounds=[30.0, 110.0], unit="°C"
    )

    # Catalyst Loading
    catalyst_loading_feature = ContinuousInput(
        key="Catalyst Loading", bounds=[0.5, 2], unit="%"
    )

    # Residence Time
    residence_time_feature = ContinuousInput(
        key="Residence Time", bounds=[1 * 60, 10 * 60], unit="minutes"
    )

    # Catalyst choice
    catalyst_feature = CategoricalInput(
        key="Catalyst",
        categories=[
            "P1-L1",
            "P2-L1",
            "P1-L2",
            "P1-L3",
            "P1-L4",
            "P1-L5",
            "P1-L6",
            "P1-L7",
        ],
    )

    # gather all individual features
    input_features = Inputs(
        features=[
            temperature_feature,
            catalyst_loading_feature,
            residence_time_feature,
            catalyst_feature,
        ]
    )

    name_map = {
        "Catalyst Loading": "catalyst_loading",
        "Residence Time": "t_res",
        "Temperature": "temperature",
        "Catalyst": "catalyst",
        "Yield": "yld",
    }
    candidates = pd.DataFrame(
        {
            "Catalyst Loading": [0.498],
            "Residence Time": [600],
            "Temperature": [30],
            "Catalyst": ["P1-L3"],
        }
    ).rename(columns=name_map)
#print(candidates)
    #emulator = summit.get_pretrained_reizman_suzuki_emulator(case=1)
    conditions = summit.DataSet.from_df(candidates)
    results = emulator.run_experiments(conditions, rtn_std=True).rename(
        columns=dict(zip(name_map.values(), name_map.keys())),
    )
    experiments = pd.DataFrame(
        {
            "Catalyst Loading": results["Catalyst Loading"],
            "Residence Time": results["Residence Time"],
            "Temperature": results["Temperature"],
            "Catalyst": results["Catalyst"],
            "Yield": results["Yield"],
            "valid_Yield": 1,
        }
    )


    max_objective = MaximizeObjective(w=1.0)
    #min_objective = MinimizeObjective(w=1.0, bounds=[0, 200])
    max_objective_2 = MaximizeObjective(w=1.0, bounds=[0, 200])


    yield_feature = ContinuousOutput(key="Yield", objective=max_objective)
    ton_feature = ContinuousOutput(key="TON", objective=max_objective_2)
    # create an output feature
    output_features = Outputs(features=[yield_feature, ton_feature])
    domain = Domain(
        inputs=input_features,
        outputs=output_features,
    )
    # a multi objective BO strategy

    #qExpectedImprovement = qEHVI()
    qLogExpectedImprovement = qLogEHVI()
    mobo_strategy_data_model = MoboStrategy(
        domain=domain,
        acquisition_function=qLogExpectedImprovement,
    )

    # map the strategy data model to the actual strategy that has functionality
    mobo_strategy = strategies.map(mobo_strategy_data_model)



    # Initialize variables
    candidates = bofire_initial_conditions

    print('initial conditions', candidates)

    experiments = evaluate_candidates(candidates)
    
    mobo_strategy.tell(experiments, replace=True, retrain=True)

    results_df = pd.DataFrame(columns=["Iteration","Catalyst Loading", "Residence Time", "Temperature", 
                                       "Catalyst", "Yield", "valid_Yield", "TON", "valid_TON"])
    
    times_df_bofire = pd.DataFrame(columns=["Iteration", "Time_taken"])

    i = 0
    done = False

    while not done:
        print('Starting Bofire loop')
        i += 1
        t1 = time.time()

        # Ask for a new experiment
        new_candidate = mobo_strategy.ask(1)
        new_experiment = evaluate_candidates(new_candidate)
        
        # Inform the strategy about the new experiment
        mobo_strategy.tell(new_experiment)

        # Add new results to DataFrame
        results_df = pd.concat([results_df, new_experiment], ignore_index=True)

        # Calculate cumulative max for Yield and TON
        results_df["Cumulative_Max_Yield"] = results_df["Yield"].cummax()
        results_df["Cumulative_Max_TON"] = results_df["TON"].cummax()
        results_df.loc[results_df["Iteration"].isna(), "Iteration"] = i

        time_taken = time.time() - t1
        print(f"Iteration {i} took {time_taken:.2f} seconds")

        times_df_bofire = pd.concat(
        [times_df_bofire, pd.DataFrame({"Iteration": [i], "Time_taken": [time_taken]})],
        ignore_index=True
    )

        print("Current Results:")
        print(results_df[["Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])

        if i > experimental_budget:
            done = True

    results = mobo_strategy.experiments
    print("Final experiments:")
    print(results)

    print("Results including cumulative max metrics:")
    print(results_df[["Yield", "Cumulative_Max_Yield", "TON", "Cumulative_Max_TON"]])

    return results_df, results, times_df_bofire