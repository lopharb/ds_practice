from ds_practice_lopharb.transform import ETL
from ds_practice_lopharb.utils import FoldLoader, FeatureExtractor
from ds_practice_lopharb.validate_model import Validator
import pandas as pd
import re
import os
from math import sin, cos, pi


def download_data(folder_id: str, output_dir: str = 'downloaded_data'):
    command = f"gdown --folder https://drive.google.com/drive/folders/{folder_id} -O {output_dir}"
    os.system(command)
    return os.path.join('./', output_dir)


def process_data(data_dir: str) -> str:
    # this needs to be replaced with fetching from google drive
    processor = ETL(data_dir,
                    [
                        'sales_train',
                        'items',
                        'shops',
                        'item_categories'
                    ])
    processor.extract()
    processor.tarnsform()
    new_path = os.path.join(data_dir, '/processed/')
    processor.load(new_path)


def extract_features(extractor: FeatureExtractor):
    item_cats = extractor.items.merge(
        extractor.categories, how='inner', on='item_category_id')

    lookup = {}
    shop_types = ['ТЦ', 'ТРК', 'ТРЦ', 'ЧС', 'МТРЦ', 'ТК', 'Выездная']

    def month_wrapper(func: callable):
        def ex_month(x):
            month = x % 12 + 1
            return func(2*pi*month/12)
        return ex_month

    def fetch_cat(id):
        if id not in lookup:
            lookup[id] = item_cats[item_cats['item_id'] == id]['item_category_name'].unique()[
                0]
        return lookup[id]

    def gen_cat(id):
        cat = fetch_cat(id)
        return 'Игры' if cat.startswith('Игры') \
            else re.split(r'\s*-\s*', cat)[0]

    def get_shop_type(shop_name):
        for st in shop_types:
            if st in shop_name:
                return st
        return 'Other'

    new_features = [
        'month_sin',
        'month_cos',
        'item_cat',
        'general_cat',
        'city',
        'shop_type'
    ]
    sources = [
        'date_block_num',
        'date_block_num',
        'item_id',
        'item_id',
        'shop_name',
        'shop_name'
    ]
    extractors = [
        month_wrapper(sin),
        month_wrapper(cos),
        fetch_cat,
        gen_cat,
        lambda x: x.split()[0],
        get_shop_type
    ]

    extractor.extract_features(sources, new_features, extractors)

    for lag in range(1, 5):
        extractor.add_lags('item_price', lag, ['shop_id', 'item_id'])
        extractor.add_lags('item_cnt_day', lag, ['shop_id', 'item_id'])


def run_pipeline(folder_id: str, needs_preprocessing: bool = False, data_storage: str = 'data') -> None:
    path = download_data(
        folder_id, data_storage)
    target = 'item_cnt_day'
    cat_cols = ['shop_id', 'item_id', 'item_cat',
                'general_cat', 'city', 'shop_type']
    if needs_preprocessing:
        process_data(data_storage)
    train = pd.read_csv(os.path.join(
        data_storage, '/processed/processed_data.csv'))
    test = pd.read_csv(os.path.join(data_storage, 'test.csv'))
    shops = pd.read_csv(os.path.join(data_storage, 'shops.csv'))
    items = pd.read_csv(os.path.join(data_storage, 'items.csv'))
    item_cats = pd.read_csv(os.path.join(data_storage, 'item_categories.csv'))
    extractor = FeatureExtractor(train=train,
                                 test=test,
                                 test_month_num=34,
                                 items=items,
                                 shops=shops,
                                 categories=item_cats)
    extract_features(extractor)
    extractor.train[target] = extractor.train[target].clip(0, 30)

    loader = FoldLoader(data=extractor.train,
                        n_train_folds=8, folding_mode='seq')
    validator = Validator(loader, extractor.features,
                          cat_cols, target, early_stopping=50)
    validator.validate(verbose=True, save_params=False)
    validator.create_submission(
        extractor.test, 'pipeline_check.csv', round=True)


if __name__ == '__main__':
    run_pipeline(folder_id='<FOLDER_ID_PLACEHOLDER>', data_storage='data')
