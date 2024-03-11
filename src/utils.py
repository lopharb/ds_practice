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

        target_col -- the name of a column in `data` contains the target values

        n_train_folds -- the amount of months used to train the model.
        must be less or equal to the amount of months in `data` minus n_valid_folds. 
        if `folding_mode` is 'stack', only affects the train size on the first iteration

        n_valid_folds -- the amount of months used to validate the model.
        must be less or equal to the amount of months in `data` minus n_train_folds
        Default: 1

        folding_mode -- one of "sequence", "overlap", "stack"
        a string denoting the data splitting mode:
            - `sequence` for `1,2train+3valid` -> `4,5train+6valid` -> `7,8train+9valid`
            - `overlap` for `1,2train+3valid` -> `2,3train+4valid` -> `3,4train+5valid`
            - `stack` for `1,2train+3valid` -> `1,2,3train+4valid` -> `1,2,3,4train+5valid`
        Default: `sequence`
        """
        self.min_fold = data['date_block_num'].unique().min()
        self.max_fold = data['date_block_num'].unique().max()
        self.current_fold = self.min_fold
        self.train_size = n_train_folds
        self._start_train_size = n_train_folds
        self.val_size = n_valid_folds
        self.mode = folding_mode
        self.len = 0
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
        self.data = data.groupby(
            ['date_block_num', 'shop_id', 'item_id'], as_index=False).sum()

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
    def __init__(self) -> None:
        raise NotImplementedError
