import re
import pandas as pd
from scipy.stats import zscore
from os import path, mkdir


class ETL:
    """
    Class that implements an ETL pipeline for the given task

    Properties:

    self.data -- a dictionary, that stores the extracted data.
    initializes with an empty dict. has table names as keys 
    and DataFrames as values.

    self.filenames -- a dictionary describing the data source.
    has table names as keys and paths to the respective files
    as values.
    """

    def __init__(self, source: str, filenames: list[str]):
        """
        Constructs a new ETL object, setting a data source for it.

        Keyword arguments:

        source -- path a directory containing the .csv files with the data.
        the path should be relative to the location, from which the script is running

        filenames -- list of filenames with the data. the names should be passed in 
        a specific order: sales -> items -> shops -> item_categories
        """
        self.data = dict()
        files = {
            'sales': None,
            'items': None,
            'shops': None,
            'item_categories': None
        }
        for key, file_name in zip(files.keys(), filenames):
            name, _ = path.splitext(file_name)
            files[key] = path.join(source, name+'.csv')
        self.filenames = files

    def remove_outliers(self, columns: list[str], threshold: float = 3):
        """ Removes the outiers from the sales table
        using a regular z-score.

        Keyword arguments:

        columns -- list of column names to clear

        threshold -- outlier detection boundary.
        the value is consifered an outlier if
        its z-score is greater than threshold, 
        or lower than -threshold. (default: 3)
        """
        for col in columns:
            mask = abs(zscore(self.data['sales'][col])) > threshold
            self.data['sales'] = self.data['sales'].drop(
                self.data['sales'][mask].index)

    @staticmethod
    def clear_noisy_names(text: str, mode: str) -> str:
        """
        Method to remove noise from text values. 
        Has two modes: for clearing items and shops respectively.

        Keyword argsuments:

        text -- the text value to be cleared.
        is modified in-place

        mode -- either 'shop' or 'item'. sets the clearing mode
        for the method. does not modify the text if any other value is passed
        """
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

    @staticmethod
    def find_preferable_id(entry_id: int, lookup: pd.DataFrame, mode: str) -> int:
        """
        Finds the ID of an entry similar to teh given one.
        Returns the original ID in case the new one is not found.

        Keyword arguments:

        entry_id -- the id of the entry in table that is supposed to be replaced

        lookup -- a DataFrame, that contains paired (old-new) IDs of
        similar entries in a table. is used to look up the new ID

        mode -- either 'shop' or 'item'. sets the  column names 
        for the lookup-table. 
        """
        if entry_id in lookup[f'{mode}_id_old'].values:
            return lookup[lookup[f'{mode}_id_old'] == entry_id][f'{mode}_id_new'].item()
        return entry_id  # leaving the id the same in case there's no suitable value

    def resolve_noisy_text(self, mode: str):
        """
        Method for replacing IDs for the sales with noisy item/shop names.
        Clears up the noise in case a suitable new value is not found.

        Keyword arguments:

        mode -- either 'shop' or 'item'. sets the  column names 
        for several tables used in process. also sets a regex for deteciong the noise
        """
        print(f'resolving noisy values in {mode}s...')
        pattern = ''
        if mode == 'shop':
            pattern = r"^[А-Яа-яA-Za-zЁё][ЁёА-Яа-яA-Za-z0-9\s\"\.,\-\(\)]*$"
        if mode == 'item':
            pattern = r"^[А-Яа-яA-Za-zЁё0-9].*$"

        weird = self.data[f'{mode}s'][~self.data[f'{mode}s']
                                      [f'{mode}_name'].str.match(pattern)].copy()
        weird[f'{mode}_name'] = weird[f'{mode}_name'].apply(
            lambda x: ETL.clear_noisy_names(x, mode))  # not sure if that's the best way to to that

        # precalculate the table that matches an old id with a new one
        pairs = weird.merge(self.data[f'{mode}s'], on=f'{mode}_name', how='inner',
                            suffixes=('_old', '_new'))
        self.data['sales'][f'{mode}_id'] = self.data['sales'][f'{mode}_id'].apply(
            lambda id: ETL.find_preferable_id(id, pairs, mode))

        # polishing values that did not find a pair
        self.data[f'{mode}s'][f'{mode}_name'] = self.data[f'{mode}s'][f'{mode}_name'].apply(
            lambda x: ETL.clear_noisy_names(x, mode))

    def extract(self) -> None:
        """
        Extracts data from sources specified in __init__
        and stores them into the data property.
        """
        self.data = dict()
        for table_name, file_path in self.filenames.items():
            self.data[table_name] = pd.read_csv(file_path)

    def tarnsform(self):
        """
        Transorms the extracted data. This process includes
        removing empty values, fixing the data types, 
        resolving noise and removing outliers.
        """
        template = '{:2.2%} of data had empty values and was removed in {}'
        for name, table in self.data.items():
            before = table.shape[0]
            self.data[name] = table.dropna()  # :D
            after = self.data[name].shape[0]
            print(template.format(1-after/before, name))

        # remove sales with negative prices
        mask = self.data['sales']['item_price'] >= 0
        self.data['sales'] = self.data['sales'][mask]

        # fix data formats where necessary
        self.data['sales']['date'] = pd.to_datetime(
            self.data['sales']['date'], format='%d.%m.%Y')
        self.data['sales']['item_cnt_day'] = self.data['sales']['item_cnt_day'].astype(
            int)

        self.resolve_noisy_text(mode='shop')
        self.resolve_noisy_text(mode='item')

        colums_to_clear = ['item_id', 'item_price', 'item_cnt_day']
        # skipping this for now?
        # self.remove_outliers(colums_to_clear)

    def load(self, destination: str):
        """
        Merges the processed data to a single table, then
        writes it to a .csv file.

        Keyword arguments:

        destination -- path to the new file location
        """
        if not path.exists(destination):
            mkdir(destination)
        df_to_save = self.data['sales'].merge(
            self.data['items'], on='item_id', how='left')
        df_to_save = df_to_save.merge(
            self.data['shops'], on='shop_id', how='left')
        df_to_save = df_to_save.merge(
            self.data['item_categories'], on='item_category_id', how='left')
        print('saving...')
        file_path = path.join(destination, 'processed_data.csv')
        df_to_save.to_csv(file_path, index=False)


if __name__ == "__main__":
    pipeline = ETL('data/', ['sales_train.csv',
                             'items', 'shops', 'item_categories'])

    pipeline.extract()
    pipeline.tarnsform()
    pipeline.load('./data/processed_files/')
