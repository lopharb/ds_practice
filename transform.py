import pandas as pd
from scipy.stats import zscore
from os import path, mkdir


def fetch_data(files: dict[str, str], dir='data'):
    """
    files - dict with the following structure:
    {'table_name' : 'file_name'}
    """
    data = dict()
    for table_name, file_name in files.items():
        # prob add a way to check if file format is specified
        name, _ = path.splitext(file_name)
        data[table_name] = pd.read_csv(path.join(dir, f'{name}.csv'))
    return data


def remove_outliers(df: pd.DataFrame, column_names: list[str], zscore_threshold: float = 3):
    for col in column_names:
        mask = abs(zscore(df[col])) > zscore_threshold
        df = df.drop(df[mask].index)
    return df


def find_preferable_id(df: pd.DataFrame, along: str, value: str, id_col_name: str):
    """
    this is supposed to take in a "corrupted" shop or item name,
    and then return the ID of a shop/item, that has the same name in normal format
    so, for example, given the input "!!!Some item 123##", it shoud return the id
    of an item called as smth similar to "Some item 123"

    df - a DataFrame to look through

    along - name of a processed column

    value - the "corrupted" item name

    id_col_name - name of the index column
    """
    pass


def resolve_duplication(data: dict[str, pd.DataFrame]):
    # have no idea
    # maybe make a function that takes in a value and
    pass


def process(data: dict[str, pd.DataFrame]):
    # 1) remove empty values in case they appear
    # 2) make table-specific changes (like format tranformations etc.)
    # 3) resolve duplicate values
    # 4) remove outliers
    template = '{:2.2%} of data had empty values and was removed in {}'
    for name, table in data.items():
        before = table.shape[0]
        data[name] = table.dropna()  # :D
        after = data[name].shape[0]
        print(template.format(1-after/before, name))

    # remove sales with negative prices
    mask = data['sales']['item_price'] >= 0
    data['sales'] = data['sales'][mask]

    # fix data formats where necessary
    data['sales']['date'] = pd.to_datetime(
        data['sales']['date'], format='%d.%m.%Y')
    data['sales']['item_cnt_day'] = data['sales']['item_cnt_day'].astype(int)

    resolve_duplication(None)  # does nothing currantly

    colums_to_clear = ['item_id', 'item_price', 'item_cnt_day']
    data['sales'] = remove_outliers(data['sales'], colums_to_clear)


def save_data(data: dict[str, pd.DataFrame], dir: str):
    if not path.exists(dir):
        mkdir(dir)
    for name, table in data.items():
        file_path = path.join(dir, f'{name}.csv')
        table.to_csv(file_path, index=False)


if __name__ == "__main__":
    files = {
        'item_categories': 'item_categories.csv',
        'items': 'items.csv',
        'sales': 'sales_train.csv',
        'shops': 'shops.csv',
        'test': 'test'
    }
    data = fetch_data(files)

    process(data)
    save_data(data, './data/processed_files/')
