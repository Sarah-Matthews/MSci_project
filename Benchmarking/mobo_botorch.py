import numpy as np
import pandas as pd
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
import matplotlib, matplotlib_inline
import os
import torch

import summit
from summit.benchmarks import get_pretrained_reizman_suzuki_emulator
from summit.benchmarks.experimental_emulator import ReizmanSuzukiEmulator 
from summit.utils.dataset import DataSet

from ax.core.parameter import ParameterType, RangeParameter, ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax import Arm, BatchTrial

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.report_utils import exp_to_df

# BoTorch acquisition class for ParEGO
from botorch.acquisition.multi_objective.parego import qLogNParEGO

# Plotting imports and initialization
from ax.utils.notebook.plotting import init_notebook_plotting, render
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume

# Model registry for creating multi-objective optimization models.
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.plot.contour import plot_contour
from ax.plot.diagnostic import tile_cross_validation
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.report_utils import exp_to_df

# Plotting imports and initialization
from ax.utils.notebook.plotting import init_notebook_plotting, render
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

init_notebook_plotting()
SMOKE_TEST = os.environ.get("SMOKE_TEST")

BATCH_SIZE = 4

if SMOKE_TEST:
    N_BATCH = 1
    num_samples = 128
    warmup_steps = 256
else:
    N_BATCH = 10
    BATCH_SIZE = 4
    num_samples = 256
    warmup_steps = 512


d = 10
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
problem = DTLZ2(num_objectives=2, dim=d, negate=True).to(**tkwargs)

#defining search space with categorical variable
available_catalysts = ["P1-L1", "P1-L2", "P1-L3", "P1-L4", "P1-L5", "P1-L6", "P1-L7", "P2-L1"]

catalyst = ChoiceParameter(name='catalyst', parameter_type=ParameterType.STRING, values=available_catalysts, is_ordered=False, sort_values=False)
catalyst_loading = RangeParameter(name='catalyst_loading', parameter_type=ParameterType.FLOAT, lower=0.5, upper=2.0)
temperature = RangeParameter(name='temperature', parameter_type=ParameterType.FLOAT, lower=30.0, upper=110.0)
t_res = RangeParameter(name='t_res', parameter_type=ParameterType.FLOAT, lower=1.0, upper=10.0)

searchspace = SearchSpace(parameters=[catalyst, catalyst_loading, temperature, t_res])
print(searchspace)


emulator = get_pretrained_reizman_suzuki_emulator(case=1)

class YieldMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        #target_name = 'yld'
        target_name = ('yld', 'DATA')
        if isinstance(trial, BatchTrial):
            arms = trial.arms_by_name.items()
        else:
            arms = [(trial.arm.name, trial.arm)]

        for arm_name, arm in arms: #trial.arms_by_name.items():
            
            params = arm.parameters
            # Run experiments with the current trial's parameters
            
            data_df = pd.DataFrame([params])
            conditions = DataSet.from_df(data_df)
            emulator_output = emulator.run_experiments(conditions, return_std=False)
            #print("Emulator output columns:", emulator_output.columns)
            #print("Emulator output sample:", emulator_output.head())
            # Find the column corresponding to the target_name in the emulator_output
            if target_name in emulator_output.columns:
                target_value = emulator_output[target_name].values[0]  
            
            else:
                raise ValueError(f"Target column '{target_name}' not found in emulator output.")

            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": target_value,  
                    "sem": 0.0,  
                }
            )

           
        return Data(df=pd.DataFrame.from_records(records))

    def is_available_while_running(self) -> bool:
        return True
    
class TONMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        #target_name = 'ton'
        target_name = ('ton', 'DATA')

        if isinstance(trial, BatchTrial):
            arms = trial.arms_by_name.items()
        else:
            arms = [(trial.arm.name, trial.arm)]

        for arm_name, arm in arms: #trial.arms_by_name.items():
            params = arm.parameters
            # Run experiments with the current trial's parameters
            data_df = pd.DataFrame([params])
            conditions = DataSet.from_df(data_df)
            emulator_output = emulator.run_experiments(conditions, return_std=False)
            # Find the column corresponding to the target_name in the emulator_output
            if target_name in emulator_output.columns:
                target_value = emulator_output[target_name].values[0]  
            
            else:
                raise ValueError(f"Target column '{target_name}' not found in emulator output.")

            records.append(
                {
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "trial_index": trial.index,
                    "mean": target_value, 
                    "sem": 0.0,  
                }
            )
        return Data(df=pd.DataFrame.from_records(records))
    
    def is_available_while_running(self) -> bool:
        return True





# https://ax.dev/tutorials/gpei_hartmann_developer.html#8.-Defining-custom-metrics - want to interact with the emulator here
metric_yld = YieldMetric(name="yld",lower_is_better= False )
metric_ton = TONMetric(name="ton", lower_is_better= False )


mo = MultiObjective(
    objectives=[Objective(metric=metric_yld, minimize=False), Objective(metric=metric_ton, minimize=False)],
)

problem_ref_point = [0.0, 100.0]

objective_thresholds = [
    ObjectiveThreshold(metric=metric, bound=val, relative=False)
    for metric, val in zip(mo.metrics, problem_ref_point)
]

optimization_config = MultiObjectiveOptimizationConfig(
    objective=mo,
    objective_thresholds=objective_thresholds,
)

# Create an experiment
def build_experiment():
    experiment = Experiment(
        name="exp_bo",
        search_space=searchspace,  
        optimization_config= optimization_config, 
        runner=SyntheticRunner(),  
    )
    return experiment


#N_INIT = 2 * (d + 1)
N_INIT = 5

experiment = build_experiment()





#initialise with sobol before filling a GP model to the initial points - skip this?
def initialize_experiment(experiment):
    #print("Experiment type:", type(experiment))
    
    if isinstance(experiment, Experiment):
        print('pre sobol')
        sobol = Models.SOBOL(search_space=experiment.search_space, seed=1234)
        print('post sobol')
        for _ in range(N_INIT):
            experiment.new_trial(sobol.gen(1)).run()
            print('for loosp')
        #generator_run = sobol.gen(N_INIT)
        #trial = experiment.new_batch_trial(generator_run)

        #print("Experiment metrics:", experiment.metrics)
        #print("Trial created:", trial)
        #print("Experiment trials:", experiment.trials)
        #print("Trial status:", trial.status)

        # Fetch data for both metrics
        #yield_metric_data = metric_yld.fetch_trial_data(trial)
        #ton_metric_data = metric_ton.fetch_trial_data(trial)
        
        # Combine or process both metrics data
        #print("Yield Metric Data:")
        #print(yield_metric_data.df)
        #print("TON Metric Data:")
        #print(ton_metric_data.df)

        

        #print("Initial data fetched:", experiment.fetch_data().df)
        print('pre-fetch')
        print(experiment.metrics)
        experiment.fetch_data()

        #data = experiment.fetch_data(metrics=experiment.metrics)
        
        #data = experiment.fetch_trials_data(trial_indices=trial.index)
 
        #print("Fetched data for all metrics:", data.df)
        
        #trial.run()
        #trial.mark_completed()
        
        #print("Trial status:", trial.status)

        #if data.df.empty:
            #print("No data available yet.")
        #else:
            #print("Initial data fetched:", data.df)
        
        
        print('post-fetch')
        exit()
        return experiment.fetch_data()
    else:
        raise ValueError("The provided experiment is not an instance of 'Experiment'") 
# using https://ax.dev/tutorials/saasbo_nehvi.html


data = initialize_experiment(experiment)



hv_list = [] #hypervolume
model = None
#swtup model & loop over iterations
for i in range(N_BATCH):
    model = Models.BOTORCH_MODULAR(
        experiment=experiment,
        data=data,
        surrogate=Surrogate(
            botorch_model_class=SaasFullyBayesianSingleTaskGP,
            mll_options={
                "num_samples": num_samples,  # Increasing this may result in better model fits
                "warmup_steps": warmup_steps,  # Increasing this may result in better model fits
            },
        )
        )
    generator_run = model.gen(BATCH_SIZE)
    trial = experiment.new_batch_trial(generator_run=generator_run)
    trial.run()
    data = Data.from_multiple_data([data, trial.fetch_data()])
    exp_df = exp_to_df(experiment)
    outcomes = torch.tensor(exp_df[["yld", "ton"]].values, **tkwargs)
    partitioning = DominatedPartitioning(ref_point=problem.ref_point, Y=outcomes)
    try:
        hv = partitioning.compute_hypervolume().item()
    except:
        hv = 0
        print("Failed to compute hv")
    hv_list.append(hv)
    print(f"Iteration: {i}, HV: {hv}")

df = exp_to_df(experiment).sort_values(by=["trial_index"])

outcomes = df[["yld", "ton"]].values



matplotlib.rcParams.update({"font.size": 16})


fig, axes = plt.subplots(1, 1, figsize=(8, 6))
algos = ["qNEHVI"]
train_obj = outcomes
cm = matplotlib.colormaps["viridis"]

n_results = N_INIT + N_BATCH * BATCH_SIZE

batch_number = df.trial_index.values
sc = axes.scatter(train_obj[:, 0], train_obj[:, 1], c=batch_number, alpha=0.8)
axes.set_title(algos[0])
axes.set_xlabel("Objective 1")
axes.set_ylabel("Objective 2")
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.ax.set_title("Iteration")