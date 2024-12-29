from setup_files_alt import *



'''
Making a new version to attempt to incorporate the util function into the baybe loops!
'''

class Models:
    """Class containing 3 bayesian objective models:
    Baybe SOBO, Baybe MOBO & BoTorch MOBO"""
    
    @staticmethod
    def run_sobo_loop(
    emulator: summit.benchmarks.experimental_emulator.ReizmanSuzukiEmulator,  
    campaign,  
    iterations: int, 
    initial_conditions_df, 
    ):
        """
        Single-objective bayesian optimisation using the BayBe back end

        emulator: Summit experimental emulator  
        campaign: the campaign defined for the optimisation 
        iterations: the number of cycles/iterations to be completed
        """

        #clear the stored measurements between each trial
        campaign._measurements_exp = pd.DataFrame()

        results_baybe_sobo = []
        cumulative_max_df = pd.DataFrame(columns=["Iteration", "Cumulative Max YLD"])
        times_df_sobo = pd.DataFrame(columns=["Iteration", "Time_taken"])

        print("Starting the SOBO loop...")

        parameter_columns = [param.name for param in searchspace.parameters]
        data_df = pd.DataFrame(columns=parameter_columns)

        #print(f"Initial conditions - randomly generated: {initial_conditions_df}")
   
        target_measurement = perform_df_experiment_multi(initial_conditions_df, emulator, objective=objective_sobo)
        campaign.add_measurements(target_measurement)
 
        
        # Record the first step
        results_baybe_sobo.append({
            "iteration": 0,
            "measurements": target_measurement
        })
        #print(results_baybe_sobo)

        #initialising a max.
        cumulative_max_yld = float('-inf')

        for i in range(1, iterations+1):
            #print(campaign)
            print(f"Running experiment {i }/{iterations}")
            t1 = time.time()
        
            recommended_conditions = campaign.recommend(batch_size=1)
            #print(f"Recommended conditions: {recommended_conditions}")

            data_df = pd.concat([data_df, recommended_conditions], ignore_index=True)

            target_measurement = perform_df_experiment(recommended_conditions, emulator, objective=objective_sobo)
            
            campaign.add_measurements(target_measurement)
            print('measurements in campaign!',campaign.measurements)
            time_taken = time.time() - t1
            #print(f"Iteration {i} took {(time.time() - t1):.2f} seconds")
        
            #eval_df_sobo = evaluate_candidates(target_measurement)
            new_yld = target_measurement['yld'].values[0]
            print(new_yld)

            if new_yld >  cumulative_max_yld:
                cumulative_max_yld = new_yld
            print(cumulative_max_yld)

            cumulative_max_df = pd.concat([cumulative_max_df, pd.DataFrame([{
            "Iteration": i,
            "Cumulative Max YLD": cumulative_max_yld
            }])], ignore_index=True)

            results_baybe_sobo.append({
                "iteration": i ,
                "measurements": target_measurement
            })

            
            print(f"Iteration {i} took {time_taken:.2f} seconds")

            times_df_sobo = pd.concat(
        [times_df_sobo, pd.DataFrame({"Iteration": [i], "Time_taken": [time_taken]})],
        ignore_index=True
    )
       
            
        
        return campaign.measurements, cumulative_max_df, times_df_sobo # cumulative_max_yld,  #results_baybe_sobo,
    
    @staticmethod
    def run_mobo_loop(
    emulator: summit.benchmarks.experimental_emulator.ReizmanSuzukiEmulator,  
    campaign,  
    iterations: int, 
    initial_conditions_df,
    ):
        """
        Multi-objective bayesian optimisation using the BayBe back end

        emulator: Summit experimental emulator  
        campaign: the campaign defined for the optimisation 
        iterations: the number of cycles/iterations to be completed
        """
        
        results_baybe_mobo = []
        cumulative_max_df = pd.DataFrame(columns=["Iteration", "Cumulative Max YLD", "Cumulative Max TON"])
        times_df_mobo = pd.DataFrame(columns=["Iteration", "Time_taken"])
        print("Starting the BayBE MOBO loop...")

        parameter_columns = [param.name for param in searchspace.parameters]
        data_df = pd.DataFrame(columns=parameter_columns)

        #print(f"Initial conditions - randomly generated: {initial_conditions_df}")
        
        target_measurement = perform_df_experiment_multi(initial_conditions_df, emulator, objective=objective_mobo)
        campaign.add_measurements(target_measurement)


        # Record the first step
        results_baybe_mobo.append({
            "iteration": 0,
            "measurements": target_measurement
        })
        #print(results_baybe_mobo)

        cumulative_max_yld = float('-inf')
        cumulative_max_ton = float('-inf')

        for i in range(1, iterations+1):
            #print(campaign)
            print(f"Running experiment {i }/{iterations}")
            t1 = time.time()
        
            recommended_conditions = campaign.recommend(batch_size=1)
            #print(f"Recommended conditions: {recommended_conditions}")

            data_df = pd.concat([data_df, recommended_conditions], ignore_index=True)

            #target_measurement = perform_df_experiment_multi(recommended_conditions, emulator, objective=objective_mobo)
            target_measurement = perform_df_experiment(recommended_conditions, emulator, objective=objective_mobo)
            campaign.add_measurements(target_measurement)
            print('measurements in campaign!',campaign.measurements)
            time_taken = time.time() - t1
        

            new_yld = target_measurement['yld'].values[0]
            print('new yld',new_yld)
            new_ton = target_measurement['ton'].values[0]
            print('new ton',new_ton)

            if new_yld >  cumulative_max_yld:
                cumulative_max_yld = new_yld
            print('cumulative yld',cumulative_max_yld)

            if new_ton >  cumulative_max_ton:
                cumulative_max_ton = new_ton
            print('cumulative ton',cumulative_max_ton)

            
            cumulative_max_df = pd.concat([cumulative_max_df, pd.DataFrame([{
             "Iteration": i,
            "Cumulative Max YLD": cumulative_max_yld,
            "Cumulative Max TON": cumulative_max_ton
        }])], ignore_index=True)

            results_baybe_mobo.append({
                "iteration": i ,
                "measurements": target_measurement
            })

            
            print(f"Iteration {i} took {time_taken:.2f} seconds")

            times_df_mobo = pd.concat(
        [times_df_mobo, pd.DataFrame({"Iteration": [i], "Time_taken": [time_taken]})],
        ignore_index=True
    )
       

        
        return  campaign.measurements, cumulative_max_df, times_df_mobo #cumulative_max_yld, cumulative_max_ton #results_baybe_mobo,
   