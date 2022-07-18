import numpy as np
import pandas as pd

# global variables
from edudata import NUM_COLS_DTYPES
from edudata.method import EMPTY_METHOD, SAMPLE_METHOD
from edudata.method import DEFAULT_METHODS_MAP, INIT_METHODS_MAP, CONT_TO_CAT_METHODS_MAP
from edudata.method import ALL_METHODS, INIT_METHODS, DEFAULT_METHODS, NA_METHODS
from edudata.processor import NAN_KEY


INIT_STEP = 'init'
PROCESSOR_STEP = 'processor'
FIT_STEP = 'fit'
GENERATE_STEP = 'generate'

NONE_TYPE = type(None)

DENSITY = 'density'


class Validator:
    def __init__(self, spop):
        self.spop = spop
        self.attributes_types = {'method': (NONE_TYPE, str, list),
                                 'visit_sequence': (NONE_TYPE, np.ndarray, list),
                                 # 'predictor_matrix': (NONE_TYPE,),
                                 'proper': (bool,),
                                 'cont_na': (NONE_TYPE, dict),
                                 'smoothing': (bool, str, dict),
                                 'default_method': (str,),
                                 'numtocat': (NONE_TYPE, list),
                                 'catgroups': (NONE_TYPE, int, dict),
                                 'seed': (NONE_TYPE, int),
                                 'k': (NONE_TYPE, int),
                                 'missing': (bool, float),
                                 'outliers': (bool, float),
                                 'numtype': (bool, str),
                                 'save': (bool, str)}

    def check_init(self):
        step = INIT_STEP

        self.default_method_validator(step=step)
        self.method_validator(step=step)
        self.visit_sequence_validator(step=step)
        # self.predictor_matrix_validator(step=step)
        self.proper_validator(step=step)
        self.cont_na_validator(step=step)
        self.smoothing_validator(step=step)
        self.numtocat_validator(step=step)
        self.catgroups_validator(step=step)
        self.seed_validator(step=step)
        self.etc_validator(step=step)


    def check_processor(self):
        step = PROCESSOR_STEP

        self.visit_sequence_validator(step=step)
        self.method_validator(step=step)
        self.predictor_matrix_validator(step=step)
        self.smoothing_validator(step=step)

        self.cont_na_validator(step=step)
        self.numtocat_validator(step=step)
        self.catgroups_validator(step=step)

    def check_fit(self):
        step = FIT_STEP

        self.method_validator(step=step)
        self.visit_sequence_validator(step=step)
        self.predictor_matrix_validator(step=step)
        self.smoothing_validator(step=step)

    def check_generate(self):
        step = GENERATE_STEP

        self.k_validator(step=step)

    def check_valid_type(self, attribute_name, return_type=False):
        attribute_type = getattr(self.spop, attribute_name)
        expected_types = self.attributes_types[attribute_name]
        assert isinstance(attribute_type, expected_types), \
            "Synthpop(합성 옵션)의 각 옵션에 정확한 형태 입력해주세요."

        if return_type:
            return attribute_type

    def method_validator(self, step=None):
        if step == INIT_STEP:
            # validate method type is allowed
            method_type = self.check_valid_type('method', return_type=True)

            if isinstance(method_type, str):
                # if method type is str
                # validate method is in allowed init methods
                assert self.spop.method in INIT_METHODS, \
                    "method를 직접 입력 시 'sample', 'cart', 'randomforest', 'parametric' 중에 입력해주세요."

            elif isinstance(method_type, list):
                # if method type is list
                # validate all methods are allowed
                assert all(m in ALL_METHODS for m in self.spop.method), \
                    "method에 list 입력시 성하고자 하는 변수와 같은 수로\n'', ‘sample’, 'cart’, 'randomforest', ‘parametric’, ‘norm’, ‘normrank’, ‘polyreg'중에 입력해 주세요."

        if step == PROCESSOR_STEP:
            first_visited_col = self.spop.visit_sequence.index[self.spop.visit_sequence == 0].values[0]

            if self.spop.method is None:
                self.spop.method = [DEFAULT_METHODS_MAP[self.spop.default_method][self.spop.df_dtypes[col]] if col != first_visited_col else SAMPLE_METHOD
                                    for col in self.spop.df_columns]

            elif isinstance(self.spop.method, str):
                self.spop.method = [INIT_METHODS_MAP[self.spop.method][self.spop.df_dtypes[col]] if col != first_visited_col else SAMPLE_METHOD
                                    for col in self.spop.df_columns]

            else:
                for col, visit_order in self.spop.visit_sequence.sort_values().iteritems():
                    col_method = self.spop.method[self.spop.df_columns.index(col)]
                    if col_method != EMPTY_METHOD:
                        assert col_method == SAMPLE_METHOD, \
                            "첫번째로 합성하는 변수는 'sample'방법으로 해주세요."
                        break

            assert len(self.spop.method) == self.spop.n_df_columns, \
                "합성하고자 하는 원데이터의 변수 갯수와 방법의 수를 일치시켜 주세요."
            self.spop.method = pd.Series(self.spop.method, index=self.spop.df_columns)

        if step == FIT_STEP:
            for col in self.spop.method.index:
                if col in self.spop.numtocat:
                    self.spop.method[col] = CONT_TO_CAT_METHODS_MAP[self.spop.method[col]]
                #(수정)
                elif col in self.spop.processor.processing_dict[NAN_KEY] and self.spop.method[col] in NA_METHODS:
                    nan_col_index = self.spop.method.index.get_loc(col)
                    index_list = self.spop.method.index.tolist()
                    index_list.insert(nan_col_index, self.spop.processed_df_columns[nan_col_index])
                    self.spop.method = self.spop.method.reindex(index_list, fill_value=CONT_TO_CAT_METHODS_MAP[self.spop.method[col]])

    def visit_sequence_validator(self, step=None):
        if step == INIT_STEP:
            # validate visit_sequence type is allowed
            visit_sequence_type = self.check_valid_type('visit_sequence', return_type=True)

            if isinstance(visit_sequence_type, np.ndarray):
                # if visit_sequence type is numpy array
                # transform visit_sequence into a list
                self.spop.visit_sequence = [col.item() for col in self.spop.visit_sequence]
                visit_sequence_type = list

            if isinstance(visit_sequence_type, list):
                # if visit_sequence type is list
                # validate all visits are unique
                assert len(set(self.spop.visit_sequence)) == len(self.spop.visit_sequence), \
                    "visit_sequence의 합성하고자 하는 변수의 이름이나 index 한번씩만 입력해주세요."
                # validate all visits are either type int or type str
                assert all(isinstance(col, int) for col in self.spop.visit_sequence) or all(isinstance(col, str) for col in self.spop.visit_sequence), \
                    "visit_sequence에는 합성하고자 하는 변수의 index나 이름을 써주세요."

        if step == PROCESSOR_STEP:
            if self.spop.visit_sequence is None:
                self.spop.visit_sequence = [col.item() for col in np.arange(self.spop.n_df_columns)]

            if isinstance(self.spop.visit_sequence[0], int):
                assert set(self.spop.visit_sequence).issubset(set(np.arange(self.spop.n_df_columns))), \
                    "visit_sequence에는 원데이터의 변수 범위 안의 index를 입력해주세요."
                self.spop.visit_sequence = [self.spop.df_columns[i] for i in self.spop.visit_sequence]
            else:
                assert set(self.spop.visit_sequence).issubset(set(self.spop.df_columns)), \
                    "visit_sequence에는 원데이터의 변수 이름을 입력해주세요."

            self.spop.visited_columns = [col for col in self.spop.df_columns if col in self.spop.visit_sequence]
            self.spop.visit_sequence = pd.Series([self.spop.visit_sequence.index(col) for col in self.spop.visited_columns], index=self.spop.visited_columns)

        if step == FIT_STEP:
            #(수정_추가_오류)
            for col in self.spop.visit_sequence.index:
                if col in self.spop.processor.processing_dict[NAN_KEY] and self.spop.method[col] in NA_METHODS:
                    visit_step = self.spop.visit_sequence[col]
                    self.spop.visit_sequence.loc[self.spop.visit_sequence >= visit_step] += 1

                    nan_col_index = self.spop.visit_sequence.index.get_loc(col)
                    index_list = self.spop.visit_sequence.index.tolist()
                    index_list.insert(nan_col_index, self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1])
                    self.spop.visit_sequence = self.spop.visit_sequence.reindex(index_list, fill_value=visit_step)

    def predictor_matrix_validator(self, step=None):
        if step == PROCESSOR_STEP:
            self.spop.predictor_matrix = np.zeros([len(self.spop.visit_sequence), len(self.spop.visit_sequence)], dtype=int)
            self.spop.predictor_matrix = pd.DataFrame(self.spop.predictor_matrix, index=self.spop.visit_sequence.index, columns=self.spop.visit_sequence.index)
            visited_columns = []
            for col, _ in self.spop.visit_sequence.sort_values().iteritems():
                self.spop.predictor_matrix.loc[col, visited_columns] = 1
                visited_columns.append(col)

        if step == FIT_STEP:
            for col in self.spop.predictor_matrix:
                #(수정_추가문_오류)
                if col in self.spop.processor.processing_dict[NAN_KEY] and self.spop.method[col] in NA_METHODS:
                    nan_col_index = self.spop.predictor_matrix.columns.get_loc(col)
                    self.spop.predictor_matrix.insert(nan_col_index, self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1], self.spop.predictor_matrix[col])

                    index_list = self.spop.predictor_matrix.index.tolist()
                    index_list.insert(nan_col_index, self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1])
                    self.spop.predictor_matrix = self.spop.predictor_matrix.reindex(index_list, fill_value=0)
                    self.spop.predictor_matrix.loc[self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1]] = self.spop.predictor_matrix.loc[col]

                    self.spop.predictor_matrix.loc[:, self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1]] = 0

    def proper_validator(self, step=None):
        if step == INIT_STEP:
            # validate proper type is allowed
            self.check_valid_type('proper')

    def cont_na_validator(self, step=None):
        if step == INIT_STEP:
            # validate cont_na type is allowed
            self.check_valid_type('cont_na')

        if step == PROCESSOR_STEP:
            if self.spop.cont_na is None:
                self.spop.cont_na = {}
            else:
                assert all(col in self.spop.df_columns for col in self.spop.cont_na), \
                "cont_na에는 원데이터의 변수명을 key로 결측치로 처리하고 싶은 값의 리스트를 value로 하는 dictionary 형태로 입력해주세요.\n (예: {'colname':[10]})"
                #(수정_추가)
                assert all(type(self.spop.cont_na[keys]) is list for keys in self.spop.cont_na), \
                "cont_na의 dictionary value에는 list를 입력해주세요.(단일 value도 list로 입력, 예:{'colname':[10]})"
                assert all(self.spop.df_dtypes[col] in NUM_COLS_DTYPES for col in self.spop.cont_na), \
                    "cont_na의 key 값에는 원데이터의 숫자형 변수 이름을 입력해주세요."
                self.spop.cont_na = {col: col_cont_na for col, col_cont_na in self.spop.cont_na.items() if self.spop.method[col] in NA_METHODS}

    def smoothing_validator(self, step=None):
        if step == INIT_STEP:
            self.check_valid_type('smoothing')

        if step == PROCESSOR_STEP:
            if self.spop.smoothing is False:
                self.spop.smoothing = {col: False for col in self.spop.df_columns}
            #(수정문_추가_오류)
            elif self.spop.smoothing is True:
                assert self.spop.smoothing == True, \
                    "smoothing에는 bool과 'density'만 입력 가능합니다."
                self.spop.smoothing = {col: self.spop.df_dtypes[col] in NUM_COLS_DTYPES for col in self.spop.df_columns}

            elif isinstance(self.spop.smoothing, str):
                assert self.spop.smoothing == DENSITY, \
                    "smoothing에는 bool과 'density'만 입력 가능합니다."
                self.spop.smoothing = {col: self.spop.df_dtypes[col] in NUM_COLS_DTYPES for col in self.spop.df_columns}
            else:
                assert all((smoothing_method == DENSITY and self.spop.df_dtypes[col] in NUM_COLS_DTYPES) or smoothing_method is False or smoothing_method is True
                           for col, smoothing_method in self.spop.smoothing.items()), \
                    "smoothing에 dictionary 입력시 합성하고자 하는 변수와 bool 값을 입력해주세요. "
                self.spop.smoothing = {col: (self. spop.smoothing.get(col, False) == DENSITY and self.spop.df_dtypes[col] in NUM_COLS_DTYPES) for col in self.spop.df_columns}


        if step == FIT_STEP:
            for col in self.spop.processed_df_columns:
                if col in self.spop.numtocat:
                    self.spop.smoothing[col] = False
                elif col in self.spop.processor.processing_dict[NAN_KEY]:
                    self.spop.smoothing[self.spop.processor.processing_dict[NAN_KEY][col]['col_nan_name']] = False

    def default_method_validator(self, step=None):
        if step == INIT_STEP:
            # validate default_method type is allowed
            self.check_valid_type('default_method')

            # validate default_method is in allowed default methods
            assert self.spop.default_method in DEFAULT_METHODS, \
                "defaualt_method에는 'cart', 'parametric'만 입력 가능합니다."

    def numtocat_validator(self, step=None):
        if step == INIT_STEP:
            self.check_valid_type('numtocat')

        if step == PROCESSOR_STEP:
            if self.spop.numtocat is None:
                self.spop.numtocat = []
            else:
                assert all(col in self.spop.df_columns for col in self.spop.numtocat)
                assert all(self.spop.df_dtypes[col] in NUM_COLS_DTYPES for col in self.spop.numtocat), \
                    "numtocat에는 그룹화하고자 하는 원데이터의 숫자형 변수 이름을 list 형태로 입력해야 합니다."

    def catgroups_validator(self, step=None):
        if step == INIT_STEP:
            catgroups_type = self.check_valid_type('catgroups', return_type=True)

            if isinstance(catgroups_type, int):
                assert self.spop.catgroups > 1, \
                    "catgroup에는 그룹화하고자하는 1 이상의 수를 입력해야 합니다."
            #수정_추가
            elif isinstance(catgroups_type, dict):
                assert set(self.spop.catgroups.keys()) == set(self.spop.numtocat)
                assert all((isinstance(col_groups, int) and col_groups > 1) for col_groups in self.spop.catgroups.values()), \
                    "catgroup에는 numtocat의 변수 이름을 key로 하며, 그룹화하고자 하는 1 이상의 수를 value로 하는 dictionary를 입력해야 합니다. "

        if step == PROCESSOR_STEP:
            if self.spop.catgroups is None:
                self.spop.catgroups = {col: 5 for col in self.spop.numtocat}
            elif isinstance(self.spop.catgroups, int):
                self.spop.catgroups = {col: self.spop.catgroups for col in self.spop.numtocat}

    def seed_validator(self, step=None):
        if step == INIT_STEP:
            # validate seed type is allowed
            self.check_valid_type('seed')

    def k_validator(self, step=None):
        if step == GENERATE_STEP:
            # validate k type is allowed
            self.check_valid_type('k')

            if self.spop.k is None:
                self.spop.k = self.spop.n_df_rows

    def etc_validator(self, step=None):
        if step == INIT_STEP:
            assert type(self.spop.missing) is bool or type(self.spop.missing) is float and self.spop.missing > 0 and self.spop.missing < 1, \
                "missing은 Bool(원데이터의 결측치 반영) 혹은 임의로 만들고자 하는 결측치의 비율을 1보다 작은 float 형태로 입력해야 합니다."

            assert type(self.spop.outliers) is bool or type(self.spop.outliers) is float and self.spop.outliers > 0 and self.spop.outliers < 1, \
                "outliers는 Bool(True: 1%) 혹은 임의로 만들고자 하는 outliers의 비율을 1보다 작은 float 형태로 입력해야 합니다."

            assert self.spop.numtype == False or self.spop.numtype == 'int', \
                "numtype은 'int'(정수로 데이터 생성) 혹은 False(소수로 데이터 생성)를 입력해야 합니다."

            assert type(self.spop.save) is bool or type(self.spop.save) == str, \
                "save는 True(synth.csv 저장), False(저장하지 않음), 문자열(문자열.csv 저장)을 입력해야 합니다."

