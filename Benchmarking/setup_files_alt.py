import summit
from summit.benchmarks import get_pretrained_reizman_suzuki_emulator
from summit.benchmarks.experimental_emulator import ReizmanSuzukiEmulator 
from summit.utils.dataset import DataSet
import pandas as pd
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from baybe.parameters import CategoricalParameter, NumericalContinuousParameter, SubstanceParameter
from baybe.targets import NumericalTarget
from baybe.recommenders import RandomRecommender, SequentialGreedyRecommender, TwoPhaseMetaRecommender, BotorchRecommender
from baybe.surrogates import GaussianProcessSurrogate
from baybe import Campaign
from baybe.objective import Objective
#from baybe.acquisition import ExpectedImprovement # see others at https://emdgroup.github.io/baybe/_autosummary/baybe.acquisition.acqfs.html#module-baybe.acquisition.acqfs
#from baybe.acquisition import debotorchize
from baybe.searchspace import SearchSpace, SearchSpaceType, SubspaceDiscrete
#from baybe.acquisition.acqfs import ExpectedImprovement
from botorch import acquisition



import bofire 
import botorch
import torch
import summit
import numpy as np
import pandas as pd
import time
import os
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


#Baybe set-up

catalyst_smiles = pd.read_csv(pathlib.Path.cwd() / pathlib.Path("suzuki_miyaura_catalysts.csv"))
ligand_smiles = pd.read_csv(pathlib.Path.cwd() / pathlib.Path("suzuki_miyaura_ligands.csv"))



#Transforming categorical variable (catalyst) into chemical variable = need smiles representation for each catalyst

emulator = get_pretrained_reizman_suzuki_emulator(case=1)

available_catalysts = {
    "P1-L1": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L1']['smiles'].values[0]}",
    "P1-L2": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L2']['smiles'].values[0]}",
    "P1-L3": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L3']['smiles'].values[0]}",
    "P1-L4": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L4']['smiles'].values[0]}",
    "P1-L5": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L5']['smiles'].values[0]}",
    "P1-L6": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L6']['smiles'].values[0]}",
    "P1-L7": f"{catalyst_smiles[catalyst_smiles['name'] == 'P1']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L7']['smiles'].values[0]}",
    "P2-L1": f"{catalyst_smiles[catalyst_smiles['name'] == 'P2']['smiles'].values[0]}.{ligand_smiles[ligand_smiles['name'] == 'L1']['smiles'].values[0]}",
}

#Defining parameter space
parameters = [
    SubstanceParameter(
        name="catalyst",
        data=available_catalysts,
        encoding="MORDRED"
    ),
    NumericalContinuousParameter(
        name="catalyst_loading",
        bounds=(0.5,2.0),
    ),
    NumericalContinuousParameter(
        name="temperature",
        bounds=(30,110),
    ),
    NumericalContinuousParameter(
        name="t_res",
        bounds=(1,10),
    )
]
#print('Parameters defined')

#Defining search space
searchspace = SearchSpace.from_product(parameters)
#print('Search space defined')


target_1 = NumericalTarget(name="yld", mode=f"MAX", bounds=(0,100), transformation="LINEAR")
target_2 = NumericalTarget(name="ton", mode=f"MAX", bounds=(0,100), transformation="LINEAR")

targets_sobo = [target_1]
targets_mobo = [target_1, target_2]


objective_sobo = Objective(mode="SINGLE", targets = targets_sobo)
#print('Objective defined (sobo)')
objective_mobo = Objective(mode="DESIRABILITY", targets = targets_mobo, weights=[50,50], combine_func="GEOM_MEAN")
#print('Objective defined (mobo)')

recommender = TwoPhaseMetaRecommender(
    initial_recommender=RandomRecommender(),
    recommender=BotorchRecommender()
)
#print('Recommender defined')

campaign_sobo = Campaign(
    searchspace=searchspace,
    objective=objective_sobo,
    recommender=recommender
    
)
#print('Campaign (sobo) defined')

campaign_mobo = Campaign(
    searchspace=searchspace,
    objective=objective_mobo,
    recommender=recommender
    
)

#print('Campaign (mobo) defined')

#BoFire set-up


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
min_objective = MinimizeObjective(w=1.0, bounds=[0, 200])

yield_feature = ContinuousOutput(key="Yield", objective=max_objective)
ton_feature = ContinuousOutput(key="TON", objective=min_objective)
# create an output feature
output_features = Outputs(features=[yield_feature, ton_feature])
domain_bofire = Domain( 
    inputs=input_features,
    outputs=output_features,
)
# a multi objective BO strategy

#qExpectedImprovement = qEHVI()
qLogExpectedImprovement = qLogEHVI()
mobo_strategy_data_model_bofire = MoboStrategy(
    domain=domain_bofire,
    acquisition_function=qLogExpectedImprovement,
)

# map the strategy data model to the actual strategy that has functionality
mobo_strategy_bofire = strategies.map(mobo_strategy_data_model_bofire)




#this function should be general (i.e. mobo & sobo) due to the dynamic target value extraction
def perform_df_experiment(data_df: pd.DataFrame, emulator: ReizmanSuzukiEmulator, objective) -> dict:
    conditions = DataSet.from_df(data_df)
    #print(conditions)
    
    
    emulator_output = emulator.run_experiments(conditions, return_std=True)
    
    result_df = data_df.copy()

    for target in objective.targets:
        target_name = target.name  # Get the name of the target 
        
        # Find the column corresponding to the target_name in the emulator_output
        if target_name in emulator_output.columns:
            target_value = emulator_output[target_name].values[0]  
            result_df[target_name] = target_value # Add the target to the result DataFrame
        else:
            raise ValueError(f"Target column '{target_name}' not found in emulator output.")

    #print(result_df)

    return result_df

#this function should be general (i.e. mobo & sobo) due to the dynamic target value extraction
def perform_df_experiment_multi(data_df: pd.DataFrame, emulator: ReizmanSuzukiEmulator, objective) -> dict:
    conditions = DataSet.from_df(data_df)
    #print(conditions)
    
    #emulator = get_pretrained_reizman_suzuki_emulator(case=1)
    emulator_output = emulator.run_experiments(conditions, return_std=True)
    
    result_df = data_df.copy()

    for target in objective.targets:
        target_name = target.name  # Get the name of the target 
        
        # Find the column corresponding to the target_name in the emulator_output
        if target_name in emulator_output.columns:
            target_values = emulator_output[target_name].values 
            
            #result_df[target_name] = emulator_output[target_name].values # Add the target to the result DataFrame
            result_df[target_name] = pd.to_numeric(target_values, errors='coerce')
        else:
            raise ValueError(f"Target column '{target_name}' not found in emulator output.")

    #print(result_df)

    return result_df


def evaluate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the candidates using the Reizman-Suzuki emulator.

    Parameters:
        candidates: A DataFrame with the experiments.

    Returns:
        A DataFrame with the experiments and the predicted yield.
    """
    name_map = {
        "Catalyst Loading": "catalyst_loading",
        "Residence Time": "t_res",
        "Temperature": "temperature",
        "Catalyst": "catalyst",
        "Yield": "yld",
        "TON": "ton",
    }
    candidates = candidates.rename(columns=name_map)
    #emulator = summit.get_pretrained_reizman_suzuki_emulator(case=1)
    conditions = summit.DataSet.from_df(candidates)
    emulator_output = emulator.run_experiments(
        conditions, rtn_std=True
    ).rename(columns=dict(zip(name_map.values(), name_map.keys())))

     # Check if 'TON' exists in the output
    if 'TON' not in emulator_output.columns:
        print("Warning: 'TON' column not found in emulator output.")
        # Optionally, add a default value for TON or raise an error
        emulator_output['TON'] = np.nan  # Or handle as appropriate
    
    return pd.DataFrame(
        {
            "Catalyst Loading": emulator_output["Catalyst Loading"],
            "Residence Time": emulator_output["Residence Time"],
            "Temperature": emulator_output["Temperature"],
            "Catalyst": emulator_output["Catalyst"],
            "Yield": emulator_output["Yield"],
            "valid_Yield": np.ones(len(emulator_output.index)),
            "TON": emulator_output["TON"],
            "valid_TON": np.ones(len(emulator_output.index)),
        }
    )
    '''return pd.DataFrame(
        {
            "Catalyst Loading": emulator_output["Catalyst Loading"],
            "Residence Time": emulator_output["Residence Time"],
            "Temperature": emulator_output["Temperature"],
            "Catalyst": emulator_output["Catalyst"],
            "Yield": emulator_output["Yield"],
            "valid_Yield": np.ones(len(emulator_output.index)),
            "TON": emulator_output["TON"],
            "valid_TON": np.ones(len(emulator_output.index)),
        }
    )'''
#print('sobo baybe campaign',campaign_sobo)
#print('mobo baybe campaign',campaign_mobo)
#print('bofire:', mobo_strategy_data_model_bofire)

