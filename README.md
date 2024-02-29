## Structure
- all the executalbes are stored in `src`
- the `notebooks` folder contains `.ipynb` files that are usually used for manual analysis, showing some data etc.

## Usage
There are two ways of using `transform.py`:
- you can put the data into a folder called "data" directly in the repo's root, then run `transform.py`, which will initialize automatically and process the data. in that case the result is stored in a subfolder inside "data"
- you can put the data anywhere else, then import `transform` into your own script. next you need to specify the data location and the destination folder when initizlizing an `ETL` object. after than you can call needed methods

## Data
There are several limitations to the data:
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
    
