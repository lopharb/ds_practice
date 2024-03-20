import json
import catboost as cb
import numpy as np
import pandas as pd
from os import path
from .utils import FoldLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


class Validator:
    def __init__(self,
                 in_features: list[str],
                 cat_features: list[str],
                 target: str,
                 data_loader: FoldLoader,
                 model_args: dict[str, any],
                 early_stopping: bool | int = 50) -> None:
        self.model = cb.CatBoostRegressor(**model_args)
        self.loader = data_loader
        self.in_features = in_features
        self.cat_features = cat_features
        self.target = target
        self.args = model_args
        self.early_stopping = early_stopping
        self.params = {'model_params': model_args,
                       'loader_params': {'train_size': data_loader.train_size,
                                         'val_size': data_loader.val_size},
                       'early_stopping': early_stopping,
                       'scores': {'fold': [],
                                  'train_rmse': [],
                                  'val_rmse': []},
                       'features': in_features
                       }

    def _save_params(self, filename: str):
        json_object = json.dumps(self.params, indent=4)

        file_name = path.basename(filename)
        with open(f'{file_name}.json', 'w') as outfile:
            outfile.write(json_object)

    def validate(self, verbose: bool = True, save_params: bool = True, filename: str = None):
        if save_params and filename is None:
            raise ValueError(
                'make sure to specify the filename when save_params is set to True')

        template = 'fold: [{:2} out of {:2}]\tRMSE-train: [{:3.3f}]\tRMSE-val: [{:3.3f}]'

        self.params['scores']['fold'] = []
        self.params['scores']['train_rmse'] = []
        self.params['scores']['val_rmse'] = []

        fold = 1
        self.loader.reset_folds()
        for train, val in tqdm(self.loader):
            # fetch data
            train_data = cb.Pool(train[self.in_features],
                                 train[self.target], cat_features=self.cat_features)
            val_data = cb.Pool(val[self.in_features],
                               val[self.target], cat_features=self.cat_features)
            # reset the model
            if self.early_stopping:
                self.model.fit(train_data, eval_set=val_data, use_best_model=True,
                               verbose=False, early_stopping_rounds=self.early_stopping)
            else:
                self.model.fit(train_data, eval_set=val_data,
                               use_best_model=True, verbose=False)
            # validate
            preds = self.model.predict(train[self.in_features])
            rmse_train = (
                np.sqrt(mean_squared_error(train[self.target], preds)))
            preds = self.model.predict(val[self.in_features])
            rmse_val = (np.sqrt(mean_squared_error(val[self.target], preds)))
            self.params['scores']['fold'].append(fold)
            self.params['scores']['train_rmse'].append(rmse_train)
            self.params['scores']['val_rmse'].append(rmse_val)
            if verbose:
                print(template.format(fold, len(
                    self.loader), rmse_train, rmse_val))
            fold += 1

        if save_params:
            self._save_params(filename)

        return self.params

    def create_submission(self, test_df: pd.DataFrame, filename: str, round=False) -> list[float | int]:
        predicts = self.model.predict(test_df[self.in_features])
        if round:
            predicts = [int(x) for x in predicts]
        submission = pd.DataFrame(
            {'ID': range(len(predicts)), 'item_cnt_month': predicts})
        submission.to_csv(path_or_buf=filename, index=False)
        return predicts
