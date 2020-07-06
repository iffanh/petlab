# petlab

## flash_calculation.py

Calculate liquid and gas composition (flash calculation) given fluid composition, pressure and temperature condition. (Now it works only on field unit)

Run flash_calculation.py with the following command:

```python flash_calculation.py /path/to/fluid_dataset.json/ /path/to/result.csv/```

Sample data set is given under the json_files folder. Run the sample data as follows:

``` python flash_calculation.py ./json_files/pvt_dataset_2.json ./results/pvt_dataset_2_result.csv ```

## drawdown_test.py

Calculate the permeability of a reservoir given a data set of drawdown test (pressure and time). (Now it works only on SI unit)

Run drawdown_test.py with the following command:

```python flash_calculation.py /path/to/drawdown_dataset.json/ /path/to/result.csv/```

Sample data set is given under the json_files folder. Run the sample data as follows:

``` python flash_calculation.py ./json_files/dd_dataset_1.json ./results/dd_dataset_1_result.csv ```
