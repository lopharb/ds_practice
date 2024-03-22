## Structure
- all the executalbes are stored in `src/ds_practice_lopharb`
- the `notebooks` folder contains `.ipynb` files that are usually used for manual analysis, showing some data etc.

## Usage
There are two ways of using `transform.py`:
- you can put the data into a folder called "data" directly in the repo's root, then run `transform.py`, which will initialize automatically and process the data. in that case the result is stored in a subfolder inside "data"
- you can put the data anywhere else, then import `transform` into your own script. next you need to specify the data location and the destination folder when initizlizing an `ETL` object. after than you can call needed methods

## Pipeline
Currently the pipeline includes the following steps:
1. download the data from external cloud storage
2. performs preprocessing if needed
3. performs the default feature extraction steps
    - you can modify or add some additional steps
    - all the feature extraction is done via the `FeatureExtractor` class
4. trains and validates the model
    - by default the model does not have any specified hyperparams
    - it uses 8 folds for training and 1 for validation
5. creates a test submission with the resulting model and stores it into a `.csv` file

In case you want to modify any of those, feel free to manually use other provided classes.
    
## Data
The data is now fetched from google drive, you can specify a value for `needs_preprocessing` when running the pipeline from `pipeline.py` to control whether you want to preprocess data once more, or just use the presaved version.


However, if you want to manually download the data, there are several limitations to it:
1. for the script to work, your data needs to include at least 4 different `.csv` files, representing:
    - sales
    - items
    - shops
    - item catrgories
2. if you're using the default data location, the filenames are also resticted:
    - `sales_train.csv` for sales
    - `items.csv` for items
    - `shops.csv` for shops
    - `item_categories.csv` for item categories
    
  You can find the data in desired format in the [kaggle contest](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data)
