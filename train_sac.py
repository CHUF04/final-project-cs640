from common.citylearn import CityLearn
from pathlib import Path
from sac import SAC as Agent
import matplotlib.pyplot as plt
import pickle

# Load environment
climate_zone = 5
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

params_agent = {'building_ids':buildings,
                 'buildings_states_actions':'common/buildings_state_action_space.json', 
                 'building_info':building_info,
                 'observation_spaces':observations_spaces, 
                 'action_spaces':actions_spaces}

# Instantiating the control agent(s)
agent_path = "models/sac_model_multiagent"
try:
    agents = pickle.load(open(agent_path, 'rb'))
    print("loaded agent")
except:
    agents = Agent(**params_agent)
    pickle.dump(agents, open(agent_path, 'wb'))
    print("created new agent")


# Number of episodes we want to run
epochs = 4
for epoch in range(epochs):
  state = env.reset()
  done = False

  action, coordination_vars = agents.select_action(state)   
  i = 0
  while not done:
      next_state, reward, done, _ = env.step(action)
      action_next, coordination_vars_next = agents.select_action(next_state)
      agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
      coordination_vars = coordination_vars_next
      state = next_state
      action = action_next

  pickle.dump(agents, open(agent_path, 'wb'))
  print("saved agent at epoch " + str(epoch))
  print(env.cost())
  
  sim_period = (5000, 5500)
  # Plotting electricity consumption breakdown
#   interval = range(sim_period[0], sim_period[1])
#   plt.figure(figsize=(30,8))
#   plt.plot(env.net_electric_consumption_no_pv_no_storage[interval])
#   plt.plot(env.net_electric_consumption_no_storage[interval])
#   plt.plot(env.net_electric_consumption[interval], '--')
#   plt.xlabel('time (hours)', fontsize=24)
#   plt.ylabel('kW', fontsize=24)
#   plt.xticks(fontsize= 24)
#   plt.yticks(fontsize= 24)
#   plt.legend(['Electricity demand without storage or generation (kW)', 'Electricity demand with PV generation and without storage(kW)', 'Electricity demand with PV generation and using RL for storage(kW)'], fontsize=24)
#   plt.show()