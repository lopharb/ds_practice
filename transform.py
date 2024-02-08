import pandas as pd
from scipy.stats import zscore
from os import path, mkdir
import re


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


def clear_bad_shops(text):
    patterns = [r"[^ЁёА-Яа-яA-Za-z0-9\s\"\.\,\-\(\)]+",  # any non-typical characters
                r"([\d\"])[^\d\"]+$"]  # whatever's after the complete address since it usually ends with a number
    text = re.sub(patterns[0], '', text)
    text = re.sub(patterns[1], r'\1', text)
    return text


def find_preferable_id(id, pairs):
    if id in pairs['shop_id_old']:
        return pairs[pairs['shop_id_old'] == id]['shop_id_new'].item()
    return id  # leaving the id the same in case there's no suitable value


def resolve_duplication(data: dict[str, pd.DataFrame]):
    # resolving shops first
    pattern = r"^[А-Яа-яA-Za-zЁё][ЁёА-Яа-яA-Za-z0-9\s\"\.,\-\(\)]*$"
    weird = data['shops'][~data['shops']
                          ['shop_name'].str.match(pattern)].copy()
    weird['shop_name'] = weird['shop_name'].apply(clear_bad_shops)

    # precalculate the table that matches an old id with a new one
    pairs = weird.merge(data['shops'], on='shop_name', how='inner',
                        suffixes=('_old', '_new'))
    data['shops']['shop_id'] = data['shops']['shop_id'].apply(
        lambda id: find_preferable_id(id, pairs))

    # polishing values that did not find a pair
    data['shops']['shop_name'] = data['shops']['shop_name'].apply(
        clear_bad_shops)


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

    resolve_duplication(data)  # does nothing currently

    colums_to_clear = ['item_id', 'item_price', 'item_cnt_day']
    data['sales'] = remove_outliers(data['sales'], colums_to_clear)


def save_data(data: dict[str, pd.DataFrame], dir: str):
    if not path.exists(dir):
        mkdir(dir)
    for name, table in data.items():
        file_path = path.join(dir, f'{name}.csv')
        print(f'saving {name}...')
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
