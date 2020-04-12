# ml-assignment-4

## Download code and dependencies
1. Make sure you have _python_ and _pip_ available.
2. Clone the project repository from https://github.com/fedme/ml-assignment-4
3. Open a terminal in the folder where you cloned the project repository
4. Run the following command to install the dependencies with _pip_:
    ```
    pip install -r ./requirements.txt
    ```

## Running the code

### Frozen Lake

1. The following command will execute the `frozen_lake.py` script which will run policy iteration, value iteration and Q-learning on several maps of the Frozen Lake problem and collect statistics.

    _Note_: The script will take a long time to execute, so it is recommended to directly look at the generated files instead.
    
    ```
    python ./frozen_lake/frozen_lake.py 
    ```
    
    The script will collect statistic in the following json files:
    - ./frozen_lake/policy_iteration_stats_0.8.json
    - ./frozen_lake/policy_iteration_stats_0.9.json
    - ./frozen_lake/value_iteration_stats_0.8.json
    - ./frozen_lake/value_iteration_stats_0.9.json
    - ./frozen_lake/qlearning_stats_0.8.json
    - ./frozen_lake/qlearning_stats_0.9.json
    
2. The following command will execute the `frozen_lake_analysis.py` script which will generate all the plots found inside the `./frozen_lake/plots` folder:
    ```
    python ./frozen_lake/frozen_lake_analysis.py 
    ```

### Forest Management

1. The following command will execute the `forest_management.py` script which will run policy iteration, value iteration and Q-learning on several sizes of the Forest Management problem and collect statistics.

    _Note_: The script will take a long time to execute, so it is recommended to directly look at the generated files instead.
    
    ```
    python ./forest_management/forest_management.py 
    ```
    
    The script will collect statistic in the following json files:
    - ./forest_management/forest_management_stats.json
    
2. The following command will execute the `forest_management_analysis.py` script which will generate all the plots found inside the `./forest_management/plots` folder:
    ```
    python ./forest_management/forest_management_analysis.py 
    ```