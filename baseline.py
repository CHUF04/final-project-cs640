# Adapted from citylearn.py to compute the baseline cost

import gym
from common.citylearn import CityLearn, RBC_Agent
from pathlib import Path
gym.logger.set_level(40)

# Load environment
costs = []
for climate_zone in range(1, 6):
    buildings = ["Building_1", "Building_2", "Building_3", 
                "Building_4", "Building_5", "Building_6", 
                "Building_7", "Building_8", "Building_9"]

    params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
            'building_attributes':'building_attributes.json', 
            'weather_file':'weather_data.csv', 
            'solar_profile':'solar_generation_1kW.csv', 
            'carbon_intensity':'carbon_intensity.csv',
            'building_ids':buildings,
            'buildings_states_actions':'common/buildings_state_action_space.json', 
            'simulation_period': (0, 8760*1-1),
            'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
            'central_agent': False,
            'save_memory': False }

    baseline = CityLearn(**params)

    _, actions_spaces = baseline.get_state_action_spaces()

    #Instantiatiing the control agent(s)
    agent_rbc = RBC_Agent(actions_spaces)

    state = baseline.reset()
    done = False
    while not done:
        action = agent_rbc.select_action([list(baseline.buildings.values())[0].sim_results['hour'][baseline.time_step]])
        next_state, rewards, done, _ = baseline.step(action)
        state = next_state
    costs.append(baseline.cost())

for climate_zone in range(1, 6):
    print("Climate Zone " + str(climate_zone) + " cost:")
    print(costs[climate_zone-1])
