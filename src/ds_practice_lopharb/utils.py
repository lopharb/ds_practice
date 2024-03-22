import pandas as pd


class FoldLoader:
    """
    Provides tools for loading time series data, processing it and dividing into folds.
    Is iterable

    Example:

    >>> loader = FoldLoader(your_data, 3, folding_mode='over')
    >>> for train, val in loader:
    >>>     # whatever we wanna do with the folds

    This creates a FoldLoader with overlap folding mode.
    After that, you can iterate through it using a for loop, where on each iteration
    you recieve training and validation sets insize a 2-element tuple
    """

    def __init__(self, data: pd.DataFrame,  n_train_folds: int, n_valid_folds: int = 1, folding_mode: str = 'sequence') -> None:
        """
        Constructs a FoldLoader object, sets its internal properties 
        (data source, training and validation sets size, folding mode)

        Keyword arguments:

        data -- a pandas dataframe containing the training data

        n_train_folds -- the amount of months used to train the model.
        must be less or equal to the amount of months in `data` minus n_valid_folds. 
        if `folding_mode` is 'stack', only affects the train size on the first iteration

        n_valid_folds -- the amount of months used to validate the model.
        must be less or equal to the amount of months in `data` minus n_train_folds
        Default: 1

        folding_mode -- one of "sequence", "overlap", "stack"
        a string denoting the data splitting mode:
            - `sequence` for `1,2train+3valid` -> `4,5train+6valid` -> `7,8train+9valid` -> ...
            - `overlap` for `1,2train+3valid` -> `2,3train+4valid` -> `3,4train+5valid` -> ...
            - `stack` for `1,2train+3valid` -> `1,2,3train+4valid` -> `1,2,3,4train+5valid` -> ...
        Default: `sequence`
        """

        # evaluate some internal parameters
        self.min_fold = data['date_block_num'].unique().min()
        self.max_fold = data['date_block_num'].unique().max()
        self.current_fold = self.min_fold
        self.train_size = n_train_folds
        self._start_train_size = n_train_folds
        self.val_size = n_valid_folds
        self.mode = folding_mode
        self.len = 0
        self.data = data
        if self.mode.startswith('seq'):
            self.len = (self.max_fold-self.min_fold +
                        1)//(self.train_size+self.val_size)
        elif self.mode.startswith('over'):
            self.len = self.max_fold - self.min_fold - self.train_size - self.val_size + 2
        elif self.mode.startswith('stack'):
            self.len = (self.max_fold-self.min_fold + 1) - \
                self.val_size - (self.train_size-1)

        if (self.max_fold - self.min_fold + 1) < self.train_size + self.val_size:
            raise ValueError(
                f'too many folds in train and valid for the given data')

    def __len__(self) -> int:
        """
        Dunder method for `len()`
        Returns:
            int: number of iterations with given fold sizes
            until the `data` runs out
        """
        return self.len

    def _get_item_internal(self, start_idx) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        iterates through the dataset to form the training and validation sets
        for current iteration

        Keyword arguments:

        start_idx -- index of the fold (month) to start from

        Returns:

        tuple(train, val) -- training and vaidation sets in form of a pandas DataFrame
        """
        train = pd.DataFrame(columns=self.data.columns)
        val = pd.DataFrame(columns=self.data.columns)

        for idx in range(start_idx, start_idx + self.train_size):
            train = pd.concat(
                [train, self.data[self.data['date_block_num'] == idx]])
        start_idx += self.train_size
        for idx in range(start_idx, start_idx + self.val_size):
            val = pd.concat(
                [val, self.data[self.data['date_block_num'] == idx]])
        return (train, val)

    def __getitem__(self, idx) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        calls `_get_item_internal` to get training and validation sets.
        adjusts internal counters corresponding to the selected folding mode.
        Raises IndexError once the iteration is over

        Keyword arguments:

        idx -- not used in there since the data has to be consecutive.
        however, it's mandatory to let __getitem__ take an index as input, 
        so there it is

        Returns:

        tuple(train, val) -- training and vaidation sets in form of a pandas DataFrame
        """
        if (self.current_fold+self.train_size+self.val_size - 1) > self.max_fold:
            raise IndexError
        start_idx = self.current_fold
        if self.mode.startswith('seq'):
            result = self._get_item_internal(start_idx)
            self.current_fold += self.train_size+self.val_size
            return result
        elif self.mode.startswith('over'):
            result = self._get_item_internal(start_idx)
            self.current_fold += 1
            return result
        elif self.mode.startswith('stack'):
            result = self._get_item_internal(start_idx)
            self.train_size += 1
            return result

    def reset_folds(self) -> None:
        """
        sets the current fold counter to point to the first fold 
        of the dataset. this effectively means that the iteration will be started over
        """
        self.current_fold = self.min_fold
        self.train_size = self._start_train_size


class FeatureExtractor:
    """
    Class that provides some utilities for feature extraction

    Attributes:

    features (list[str]): list of features contained in data.
    It's updated automatically if the feature is exctracted via
    the class methods 
    """

    def __init__(self,
                 train: pd.DataFrame,
                 test: pd.DataFrame,
                 items: pd.DataFrame,
                 shops: pd.DataFrame,
                 categories: pd.DataFrame,
                 test_month_num: int) -> None:
        """
        Initiates a new FoldLoader object. This also includes some data
        preparing steps like grouping it and joining with the names of
        items, shops and categories.

        Args:

            train (pd.DataFrame): train data set

            test (pd.DataFrame): test data set

            items (pd.DataFrame): table for matching items with their names and
            categories

            shops (pd.DataFrame): table for matching shops with their names

            categories (pd.DataFrame): table for matching categories with their names

            test_month_num (int): sets the date_block_num value for the test set
        """
        self.train = train.copy()
        self.test = test.copy()
        self.items = items.copy()
        self.shops = shops.copy()
        self.categories = categories.copy()

        self.features = ['date_block_num', 'shop_id', 'item_id']

        # group the data
        agg_cols = ['date_block_num', 'shop_id', 'item_id']
        self.train = self.train.groupby(agg_cols, as_index=False).sum()
        self.test['date_block_num'] = test_month_num
        self.test = self.test.groupby(agg_cols, as_index=False).sum()

        # average out the price
        self.train['item_price'] = self.train['item_price'] / \
            self.train['item_cnt_day']

        # add named values from other tables
        self._join_names('train')
        self._join_names('test')

    def _join_names(self, mode: str):
        _df = None
        if mode == 'train':
            _df = self.train
        else:
            _df = self.test

        _df = _df.merge(self.items, how='inner', on='item_id')
        _df = _df.merge(self.categories, how='inner', on='item_category_id')
        _df = _df.merge(self.shops, how='inner', on='shop_id')

        if mode == 'train':
            self.train = _df
        else:
            self.test = _df

    def extract_features(self,
                         from_cols: list[str],
                         to_cols: list[str],
                         using: list[callable]) -> None:
        """
        Generates desired features using a provided function.
        Is only suitable for features that depend on a single column.
        Modifies the self.train/test property

        Args:
            from_cols (list[str]): list of column names in `data` that will 
            be used as sources for the features

            to_cols (list[str]): list of new column names that will be used
            to store the features

            using (list[callable]): list of one-argument functions that will
            be used to calculate new values
        """
        if not (len(from_cols) == len(to_cols) == len(using)):
            raise ValueError('list sizes have to match')
        for f, t, u in zip(from_cols, to_cols, using):
            print(f'Extracting {t} from {f}...')
            if t in self.train.columns:
                continue
            self.train[t] = self.train[f].apply(u)
            self.test[t] = self.test[f].apply(u)
            print('Done!')
        self.features += to_cols

    def add_lags(self, column: str, lag: int, index_cols: list[str]) -> None:
        """
        Adds lagged values to the train and test sets

        Args:
            column (str): column in the data to take the lags from

            lag (int): the amount of timestamps to step back

            index_cols (list[str]): columns that are used to group the data
            (date_block_num is always explicitly included in this list)

        Raises:
            ValueError if a lag <= 0 is passed
        """
        if lag < 1:
            raise ValueError('lag should be 1 or higher')
        print(f'Adding lag {lag} for {column}...')
        _source = self.train[['date_block_num', column]+index_cols].copy()
        _source['date_block_num'] += lag
        new_col_name = f'{column}_lag_{lag}'

        _col = column
        joined = self.train.merge(_source, how='left',
                                  on=index_cols+['date_block_num'])
        if _col in self.train.columns:
            _col += '_y'
        joined[new_col_name] = joined[_col].fillna(0)
        self.train[new_col_name] = joined[new_col_name]

        _col = column
        joined = self.test.merge(_source, how='left',
                                 on=index_cols+['date_block_num'])
        if _col in self.test.columns:
            _col += '_y'
        joined[new_col_name] = joined[_col].fillna(0)
        self.test[new_col_name] = joined[new_col_name]
        print('Done!')
        self.features.append(new_col_name)
