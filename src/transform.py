import re
import pandas as pd
from scipy.stats import zscore
from os import path, mkdir


# TODO add docstrings

def fetch_data(file_names: dict[str, str], data_folder='data'):
    data = dict()
    for table_name, file_name in file_names.items():
        # prob add a way to check if file format is specified
        name, _ = path.splitext(file_name)
        data[table_name] = pd.read_csv(path.join(data_folder, f'{name}.csv'))
    return data


def remove_outliers(df: pd.DataFrame, column_names: list[str], zscore_threshold: float = 3):
    for col in column_names:
        mask = abs(zscore(df[col])) > zscore_threshold
        df = df.drop(df[mask].index)
    return df


def clear_noisy_names(text, mode):
    if mode == 'shop':
        patterns = []

        # any non-typical characters
        patterns.append(r"[^ЁёА-Яа-яA-Za-z0-9\s\"\.\,\-\(\)]+")

        # whatever's after the complete address since it usually ends with a number
        patterns.append(r"([\d\"])[^\d\"]+$")
        text = re.sub(patterns[0], '', text)
        text = re.sub(patterns[1], r'\1', text)

    if mode == 'item':
        pattern = r"[\/\*\?\!]"
        text = re.sub(pattern, '', text)

    return text


def find_preferable_id(entry_id, pairs, mode):
    if entry_id in pairs[f'{mode}_id_old'].values:
        return pairs[pairs[f'{mode}_id_old'] == entry_id][f'{mode}_id_new'].item()
    return entry_id  # leaving the id the same in case there's no suitable value


def resolve_noisy_text(data_frames: dict[str, pd.DataFrame], mode: str):
    print(f'resolving noisy values in {mode}s...')
    pattern = ''
    if mode == 'shop':
        pattern = r"^[А-Яа-яA-Za-zЁё][ЁёА-Яа-яA-Za-z0-9\s\"\.,\-\(\)]*$"
    if mode == 'item':
        pattern = r"^[А-Яа-яA-Za-zЁё0-9].*$"

    weird = data_frames[f'{mode}s'][~data_frames[f'{mode}s']
                                    [f'{mode}_name'].str.match(pattern)].copy()
    weird[f'{mode}_name'] = weird[f'{mode}_name'].apply(
        lambda x: clear_noisy_names(x, mode))  # not sure if that's the best way to to that

    # precalculate the table that matches an old id with a new one
    pairs = weird.merge(data_frames[f'{mode}s'], on=f'{mode}_name', how='inner',
                        suffixes=('_old', '_new'))
    data_frames['sales'][f'{mode}_id'] = data_frames['sales'][f'{mode}_id'].apply(
        lambda id: find_preferable_id(id, pairs, mode))

    # polishing values that did not find a pair
    data_frames[f'{mode}s'][f'{mode}_name'] = data_frames[f'{mode}s'][f'{mode}_name'].apply(
        lambda x: clear_noisy_names(x, mode))


def process(data_frames: dict[str, pd.DataFrame]):
    template = '{:2.2%} of data had empty values and was removed in {}'
    for name, table in data_frames.items():
        before = table.shape[0]
        data_frames[name] = table.dropna()  # :D
        after = data_frames[name].shape[0]
        print(template.format(1-after/before, name))

    # remove sales with negative prices
    mask = data_frames['sales']['item_price'] >= 0
    data_frames['sales'] = data_frames['sales'][mask]

    # fix data formats where necessary
    data_frames['sales']['date'] = pd.to_datetime(
        data_frames['sales']['date'], format='%d.%m.%Y')
    data_frames['sales']['item_cnt_day'] = data_frames['sales']['item_cnt_day'].astype(
        int)

    resolve_noisy_text(data_frames, 'shop')
    resolve_noisy_text(data_frames, 'item')

    colums_to_clear = ['item_id', 'item_price', 'item_cnt_day']
    data_frames['sales'] = remove_outliers(
        data_frames['sales'], colums_to_clear)


def save_data(data: dict[str, pd.DataFrame], destination: str):
    if not path.exists(destination):
        mkdir(destination)
    df_to_save = data['sales'].merge(data['items'], on='item_id', how='left')
    df_to_save = df_to_save.merge(data['shops'], on='shop_id', how='left')
    df_to_save = df_to_save.merge(
        data['item_categories'], on='item_category_id', how='left')
    print(df_to_save.head(5))
    print('saving...')
    file_path = path.join(destination, 'processed_data.csv')
    df_to_save.to_csv(file_path, index=False)


if __name__ == "__main__":
    files = {
        'item_categories': 'item_categories.csv',
        'items': 'items.csv',
        'sales': 'sales_train.csv',
        'shops': 'shops.csv',
        'test': 'test'
    }
    data_frames = fetch_data(files)

    process(data_frames)
    save_data(data_frames, './data/processed_files/')
