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
                                 'k': (NONE_TYPE, int)}

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
        assert isinstance(attribute_type, expected_types)

        if return_type:
            return attribute_type

    def method_validator(self, step=None):
        if step == INIT_STEP:
            # validate method type is allowed
            method_type = self.check_valid_type('method', return_type=True)

            if isinstance(method_type, str):
                # if method type is str
                # validate method is in allowed init methods
                assert self.spop.method in INIT_METHODS

            elif isinstance(method_type, list):
                # if method type is list
                # validate all methods are allowed
                assert all(m in ALL_METHODS for m in self.spop.method)

        if step == PROCESSOR_STEP:
            first_visited_col = self.spop.visit_sequence.index[self.spop.visit_sequence == 0].values[0]

            if self.spop.method is None:
                self.spop.method = [DEFAULT_METHODS_MAP[self.spop.default_method][self.spop.df_dtypes[col]] if col != first_visited_col else SAMPLE_METHOD
                                    for col in self.spop.df_columns]

            elif isinstance(self.spop.method, str):                self.spop.method = [INIT_METHODS_MAP[self.spop.method][self.spop.df_dtypes[col]] if col != first_visited_col else SAMPLE_METHOD
                                    for col in self.spop.df_columns]

            else:
                for col, visit_order in self.spop.visit_sequence.sort_values().iteritems():
                    col_method = self.spop.method[self.spop.df_columns.index(col)]
                    if col_method != EMPTY_METHOD:
                        assert col_method == SAMPLE_METHOD
                        break

            assert len(self.spop.method) == self.spop.n_df_columns
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
                assert len(set(self.spop.visit_sequence)) == len(self.spop.visit_sequence)
                # validate all visits are either type int or type str
                assert all(isinstance(col, int) for col in self.spop.visit_sequence) or all(isinstance(col, str) for col in self.spop.visit_sequence)

        if step == PROCESSOR_STEP:
            if self.spop.visit_sequence is None:
                self.spop.visit_sequence = [col.item() for col in np.arange(self.spop.n_df_columns)]

            if isinstance(self.spop.visit_sequence[0], int):
                assert set(self.spop.visit_sequence).issubset(set(np.arange(self.spop.n_df_columns)))
                self.spop.visit_sequence = [self.spop.df_columns[i] for i in self.spop.visit_sequence]
            else:
                assert set(self.spop.visit_sequence).issubset(set(self.spop.df_columns))

            self.spop.visited_columns = [col for col in self.spop.df_columns if col in self.spop.visit_sequence]
            self.spop.visit_sequence = pd.Series([self.spop.visit_sequence.index(col) for col in self.spop.visited_columns], index=self.spop.visited_columns)

        if step == FIT_STEP:
            #(수정_추가)
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
                #(수정_추가)
                if col in self.spop.processor.processing_dict[NAN_KEY] and self.spop.method[col] in NA_METHODS:
                    nan_col_index = self.spop.predictor_matrix.columns.get_loc(col)
                    self.spop.predictor_matrix.insert(nan_col_index, self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1], self.spop.predictor_matrix[col])

                    index_list = self.spop.predictor_matrix.index.tolist()
                    index_list.insert(nan_col_index, self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1])
                    self.spop.predictor_matrix = self.spop.predictor_matrix.reindex(index_list, fill_value=0)
                    self.spop.predictor_matrix.loc[self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1]] = self.spop.predictor_matrix.loc[col]

                    self.spop.predictor_matrix.loc[col, self.spop.processed_df_columns[self.spop.processed_df_columns.index(col)-1]] = 1

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
                assert all(col in self.spop.df_columns for col in self.spop.cont_na)
                assert all(self.spop.df_dtypes[col] in NUM_COLS_DTYPES for col in self.spop.cont_na)
                self.spop.cont_na = {col: col_cont_na for col, col_cont_na in self.spop.cont_na.items() if self.spop.method[col] in NA_METHODS}

    def smoothing_validator(self, step=None):
        if step == INIT_STEP:
            self.check_valid_type('smoothing')

        if step == PROCESSOR_STEP:
            if self.spop.smoothing is False:
                self.spop.smoothing = {col: False for col in self.spop.df_columns}
            elif isinstance(self.spop.smoothing, str):
                assert self.spop.smoothing == DENSITY
                self.spop.smoothing = {col: self.spop.df_dtypes[col] in NUM_COLS_DTYPES for col in self.spop.df_columns}
            else:
                assert all((smoothing_method == DENSITY and self.spop.df_dtypes[col] in NUM_COLS_DTYPES) or smoothing_method is False
                           for col, smoothing_method in self.spop.smoothing.items())
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
            assert self.spop.default_method in DEFAULT_METHODS

    def numtocat_validator(self, step=None):
        if step == INIT_STEP:
            self.check_valid_type('numtocat')

        if step == PROCESSOR_STEP:
            if self.spop.numtocat is None:
                self.spop.numtocat = []
            else:
                assert all(col in self.spop.df_columns for col in self.spop.numtocat)
                assert all(self.spop.df_dtypes[col] in NUM_COLS_DTYPES for col in self.spop.numtocat)

    def catgroups_validator(self, step=None):
        if step == INIT_STEP:
            catgroups_type = self.check_valid_type('catgroups', return_type=True)

            if isinstance(catgroups_type, int):
                assert self.spop.catgroups > 1

            elif isinstance(catgroups_type, dict):
                assert set(self.spop.catgroups.keys()) == set(self.spop.numtocat)
                assert all((isinstance(col_groups, int) and col_groups > 1) for col_groups in self.spop.catgroups.values())

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
