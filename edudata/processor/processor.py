import numpy as np
import pandas as pd
import random

# global variables
from edudata import NUM_COLS_DTYPES, CAT_COLS_DTYPES
from edudata.method import SAMPLE_METHOD

NAN_KEY = 'nan'
NUMTOCAT_KEY = 'numtocat'


class Processor:
    def __init__(self, spop):
        self.spop = spop
        self.processing_dict = {NUMTOCAT_KEY: {},
                                NAN_KEY: {}
                                }

    def preprocess(self, df, dtypes):

        for col in self.spop.visited_columns:
            #(수정_추가_오류)
            if self.spop.method[col] == SAMPLE_METHOD:
                df.dropna(subset=[col], inplace=True)
                df.index = range(len(df))
            else:
                col_nan_indices = df[col].isna()
                cont_nan_indices = {v: df[col] == v for v in self.spop.cont_na.get(col, [])}
                col_nan_series = [(np.nan, col_nan_indices)] + list(cont_nan_indices.items())

                col_all_nan_indices = pd.DataFrame({index: value[1] for index, value in enumerate(col_nan_series)}).max(axis=1)
                col_not_nan_indices = np.invert(col_all_nan_indices)

                if col in self.spop.numtocat:
                    self.processing_dict[NUMTOCAT_KEY][col] = {'dtype': self.spop.df_dtypes[col],
                                                               'categories': {}
                                                               }

                    not_nan_values = df.loc[col_not_nan_indices, col].copy()

                    #(수정_추가_오류)
                    df_cut = pd.cut(df.loc[col_not_nan_indices, col], self.spop.catgroups[col],
                                    labels=range(self.spop.catgroups[col]), include_lowest=True)
                    df_col = df.loc[col_not_nan_indices, col]

                    #(수정_추가)
                    for i in df_col.index:
                        df.loc[i, col] = df_cut[i]

                    grouped = pd.DataFrame({'grouped': df.loc[col_not_nan_indices, col], 'real': not_nan_values}).groupby('grouped')
                    self.processing_dict[NUMTOCAT_KEY][col]['categories'] = grouped['real'].apply(np.array).to_dict()

                    for index, (_, bool_series) in enumerate(col_nan_series):
                        nan_cat = self.spop.catgroups[col] + index
                        self.processing_dict[NUMTOCAT_KEY][col]['categories'][nan_cat] = df.loc[bool_series, col].to_numpy()
                        df.loc[bool_series, col] = nan_cat

                    df[col] = df[col].astype('category')
                    self.spop.df_dtypes[col] = 'category'

                else:
                    #(수정_추가)
                    if self.spop.df_dtypes[col] in 'category':
                        if col_nan_indices.any():
                            NaN_value = 'cat_NaN'
                            col_nan_name = col + '_NaN'
                            df.insert(df.columns.get_loc(col), col_nan_name, 0)
                            self.processing_dict[NAN_KEY][col] = {'col_nan_name': col_nan_name,
                                                                  'dtype': self.spop.df_dtypes[col],
                                                                  'nan_flags': {}
                                                                  }

                            for index, (cat, bool_series) in enumerate(col_nan_series):
                                cat_index = index + 1
                                self.processing_dict[NAN_KEY][col]['nan_flags'][cat_index] = NaN_value
                                df.loc[bool_series, col_nan_name] = cat_index
                            df[col].cat.add_categories(NaN_value, inplace=True)
                            df.loc[col_all_nan_indices, col] = NaN_value
                            self.spop.df_dtypes[col_nan_name] = 'category'

                    elif self.spop.df_dtypes[col] in NUM_COLS_DTYPES:
                        if col_all_nan_indices.any():
                            col_nan_name = col + '_NaN'
                            df.insert(df.columns.get_loc(col), col_nan_name, 0)

                            self.processing_dict[NAN_KEY][col] = {'col_nan_name': col_nan_name,
                                                                  'dtype': self.spop.df_dtypes[col],
                                                                  'nan_flags': {}
                                                                  }

                            for index, (cat, bool_series) in enumerate(col_nan_series):
                                cat_index = index + 1
                                self.processing_dict[NAN_KEY][col]['nan_flags'][cat_index] = 0
                                df.loc[bool_series, col_nan_name] = cat_index
                            df.loc[col_all_nan_indices, col] = 0

                            df[col_nan_name] = df[col_nan_name].astype('category')
                            self.spop.df_dtypes[col_nan_name] = 'category'

        return df

    def postprocess(self, synth_df, random_state=None):
        #(수정)

        if self.spop.missing:
            for col, processing_nan_col_dict in self.processing_dict[NAN_KEY].items():
                if processing_nan_col_dict['dtype'] in CAT_COLS_DTYPES:
                    col_nan_value = processing_nan_col_dict['nan_flags'][1]
                    synth_df[col] = synth_df[col].astype(object)
                    synth_df.loc[synth_df[col] == col_nan_value, col] = np.nan
                    synth_df[col] = synth_df[col].astype('category')

                    synth_df.drop(columns=processing_nan_col_dict['col_nan_name'], inplace=True)

                    # 수정 df(원본데이터)도 NaN column 삭제
                    # df.drop(columns=processing_nan_col_dict['col_nan_name'], inplace=True)

                elif processing_nan_col_dict['dtype'] in NUM_COLS_DTYPES:
                    for col_nan_flag, col_nan_value in processing_nan_col_dict['nan_flags'].items():
                        nan_flag_indices = synth_df[processing_nan_col_dict['col_nan_name']] == col_nan_flag
                        synth_df.loc[nan_flag_indices, col] = np.nan
                    synth_df.drop(columns=processing_nan_col_dict['col_nan_name'], inplace=True)

            # synth_df = synth_df.replace(0, np.nan)

                    # df.drop(columns=processing_nan_col_dict['col_nan_name'], inplace=True)
            #(수정_추가)
            if self.spop.missing is not True and self.spop.missing > 0:
                count = 0
                np.random.seed(random_state)
                while round(len(synth_df)*len(synth_df.columns)*self.spop.missing) != count:
                    r_l = np.random.randint(0, len(synth_df) - 1)
                    c_l = np.random.randint(0, len(synth_df.columns) - 1)
                    synth_df.iloc[r_l, c_l] = np.nan
                    count += 1


        # else:
        #     for col, processing_nan_col_dict in spop.processing_dict[NAN_KEY].items():
        #         if processing_nan_col_dict['dtype'] in CAT_COLS_DTYPES or processing_nan_col_dict[
        #             'dtype'] in NUM_COLS_DTYPES:
        #             df.drop(columns=processing_nan_col_dict['col_nan_name'], inplace=True)

        #(수정_추가)
        if self.spop.outliers is True:
            temp_df = synth_df.copy()
            np.random.seed(random_state)
            for i in range(round(len(synth_df)*len(synth_df.columns)*0.01)):
                r_l = np.random.randint(0, len(synth_df)-1)
                c_l = np.random.randint(0, len(synth_df.columns)-1)
                if synth_df.iloc[:, c_l].dtypes in['int64', 'float64']:
                    q1 = synth_df.iloc[:, c_l].quantile(0.25)
                    q3 = synth_df.iloc[:, c_l].quantile(0.75)
                    if self.spop.numtype == 'int':
                        if i%2 == 0:
                            temp_df.iloc[r_l, c_l] = round(q1 - np.random.uniform(1.5, 2)*(q3-q1))
                        else:
                            temp_df.iloc[r_l, c_l] = round(q3 + np.random.uniform(1.5, 2)*(q3-q1))
                    else :
                        if i%2 == 0:
                            temp_df.iloc[r_l, c_l] = round(q1 - np.random.uniform(1.5, 2)*(q3-q1), 2)
                        else:
                            temp_df.iloc[r_l, c_l] = round(q3 + np.random.uniform(1.5, 2)*(q3-q1), 2)

            synth_df = temp_df.copy()

        elif self.spop.outliers > 0:
            temp_df = synth_df.copy()
            np.random.seed(random_state)
            for i in range(round(len(synth_df)*len(synth_df.columns)*self.spop.outliers)):
                r_l = random.randint(0, len(synth_df)-1)
                c_l = random.randint(0, len(synth_df.columns)-1)
                if synth_df.iloc[:, c_l].dtypes in['int64', 'float64']:
                    q1 = synth_df.iloc[:, c_l].quantile(0.25)
                    q3 = synth_df.iloc[:, c_l].quantile(0.75)
                    if self.spop.numtype == 'int':
                        if i % 2 == 0:
                            temp_df.iloc[r_l, c_l] = round(q1 - random.uniform(1.5, 2) * (q3 - q1))
                        else:
                            temp_df.iloc[r_l, c_l] = round(q3 + random.uniform(1.5, 2) * (q3 - q1))
                    else:
                        if i % 2 == 0:
                            temp_df.iloc[r_l, c_l] = round(q1 - random.uniform(1.5, 2) * (q3 - q1), 2)
                        else:
                            temp_df.iloc[r_l, c_l] = round(q3 + random.uniform(1.5, 2) * (q3 - q1), 2)

            synth_df = temp_df.copy()


        if self.spop.save is True:
            synth_df.to_csv('synth.csv')
        elif self.spop.save is not False:
            synth_df.to_csv('%s.csv' % (self.spop.save))

        return synth_df
