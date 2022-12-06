from common.citylearn import CityLearn
from pathlib import Path
import pickle

costs = []
for climate_zone in range(1, 5):
    # Load environment
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

    env = CityLearn(**params)

    # Contains the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
    observations_spaces, actions_spaces = env.get_state_action_spaces()

    # Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
    building_info = env.get_building_information()

    # Instantiating the control agent(s)
    agent_path = "models/sac_model_multiagent"
    try:
        agents = pickle.load(open(agent_path, 'rb'))
        print("loaded agent")
    except:
        raise ValueError("no agent found in: " + agent_path)

    state = env.reset()
    done = False

    action, coordination_vars = agents.select_action(state)   
    i = 0
    while not done:
        next_state, reward, done, _ = env.step(action)
        action_next, coordination_vars_next = agents.select_action(next_state)
        coordination_vars = coordination_vars_next
        state = next_state
        action = action_next
    costs.append(env.cost())

    sim_period = (5000, 5500)

avg_costs = {}
for key in costs[0].keys():
    avg_costs[key] = 0.0
    for climate_zone in range(1, 5):
        avg_costs[key] += costs[climate_zone-1][key]
    avg_costs[key] /= 4

print(avg_costs)
