#from .models_v2 import * #note this is v2!!!

from setup_files_alt import *
from sklearn.preprocessing import MinMaxScaler
import pickle

'''Using the random strategy model from bofire to initialise points for all 3 models!'''

def initialise_random_point(domain):
    random_strategy_model = RandomStrategyModel(domain=domain)
    random_strategy = strategies.map(random_strategy_model)
    candidates = random_strategy.ask(10)

    return candidates

random_initialised_points = initialise_random_point(domain=domain_bofire)
df_random_initialised_point = pd.DataFrame(random_initialised_points)
df_random_initialised_point_renamed = df_random_initialised_point.rename(columns=name_map)


initial_conditions = df_random_initialised_point_renamed
bofire_initial_conditions = random_initialised_points

