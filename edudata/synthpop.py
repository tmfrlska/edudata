import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from edudata.validator import Validator
from edudata.processor import Processor

from edudata.processor import NAN_KEY
from edudata import NUM_COLS_DTYPES, CAT_COLS_DTYPES
from edudata.method import CART_METHOD, METHODS_MAP, NA_METHODS

from table_evaluator import load_data, TableEvaluator

class Synthpop:
    def __init__(self,
                 method=None,
                 visit_sequence=None,
                 predictor_matrix=None,
                 proper=False,
                 cont_na=None,
                 smoothing=False,
                 default_method=CART_METHOD,
                 numtocat=None,
                 catgroups=None,
                 seed=None,
                 missing=False,
                 outliers=False,
                 school=False,
                 save=False):

        # initialise the validator and processor
        self.validator = Validator(self)
        self.processor = Processor(self)

        # initialise arguments
        self.method = method
        self.visit_sequence = visit_sequence
        self.predictor_matrix = predictor_matrix
        self.proper = proper
        self.cont_na = cont_na
        self.smoothing = smoothing
        self.default_method = default_method
        self.numtocat = numtocat
        self.catgroups = catgroups
        self.seed = seed
        self.missing = missing
        self.outliers = outliers
        self.school = school
        self.save = save

        self.validator.check_init()

    def fit(self, df, dtypes = None):

        self.df_columns = df.columns.tolist()
        self.n_df_rows, self.n_df_columns = np.shape(df)
        dtypes = dtypes

        # (수정_추가)
        if dtypes is None:
            dtypes = {col: df.dtypes[col].name for col in self.df_columns}
            # for col, dtype in dtypes.items():
            #     if dtype == 'object':
            #         dtypes[col] = 'category'

        df = df.astype(dtypes)
        self.df_dtypes = dtypes

        # check processor
        self.validator.check_processor()
        # preprocess
        processed_df = self.processor.preprocess(df, self.df_dtypes)
        self.processed_df_columns = processed_df.columns.tolist()
        self.n_processed_df_columns = len(self.processed_df_columns)
        # check fit
        self.validator.check_fit()
        # fit
        self._fit(processed_df)

    def _fit(self, df):
        self.saved_methods = {}

        # (수정_추가)
        if self.missing:
            None
        else:
            NaN_col = [self.processor.processing_dict[NAN_KEY][col]['col_nan_name'] for col in
                       self.processor.processing_dict[NAN_KEY].keys()]

            for col in self.processor.processing_dict[NAN_KEY]:
                nan_indices = df[self.processor.processing_dict[NAN_KEY][col]['col_nan_name']] != 0
                df.loc[nan_indices, col] = np.nan
            self.visit_sequence.drop(NaN_col, inplace=True)
            self.predictor_matrix.drop(index=NaN_col, columns=NaN_col, inplace=True)
            df.dropna(inplace=True)

        self.predictor_matrix_columns = self.predictor_matrix.columns.to_numpy()

        for col, visit_step in self.visit_sequence.sort_values().iteritems():
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col], smoothing=self.smoothing[col],
                                                       proper=self.proper, random_state=self.seed)
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            col_method.fit(X_df=df[col_predictors], y_df=df[col])
            self.saved_methods[col] = col_method
            # (수정)
            print('{}_trained'.format(col))

    def generate(self, k=None):
        self.k = k

        # check generate
        self.validator.check_generate()
        # generate
        synth_df = self._generate()
        # postprocess
        processed_synth_df = self.processor.postprocess(synth_df)

        return processed_synth_df

    def _generate(self):
        synth_df = pd.DataFrame(data=np.zeros([self.k, len(self.visit_sequence)]), columns=self.visit_sequence.index)

        for col, visit_step in self.visit_sequence.sort_values().iteritems():
            col_method = self.saved_methods[col]
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]

            synth_df[col] = col_method.predict(synth_df[col_predictors])
            #(수정_추가)
            if col in self.numtocat:
                #확인필요
               synth_df[col] = synth_df[col].astype('category')

            if self.missing :
                if col in self.processor.processing_dict[NAN_KEY] and self.df_dtypes[col] in NUM_COLS_DTYPES and self.method[col] in NA_METHODS:
                    nan_indices = synth_df[self.processor.processing_dict[NAN_KEY][col]['col_nan_name']] != 0
                    synth_df.loc[nan_indices, col] = 0

            if self.df_dtypes[col] in NUM_COLS_DTYPES and synth_df[col].notna().any():
                if self.school == 'e':
                    synth_df[col] = synth_df[col].astype(int)
                else:
                    synth_df[col] = synth_df[col].astype(self.df_dtypes[col])
                    # 경계에 있는 값들을 변화 시켜 pmse ratio에 큰 영향을 미침
                    # synth_df[col] = round(synth_df[col], 2)

            print('{}_generated'.format(col))

        return synth_df

    def compare(self, df, synth, detail=False):
        #(수정_추가)
        assert set(synth.columns).issubset(set(df.columns)), "원데이터셋과 합성데이터셋을 확인해주세요."

        for col in df.columns:
            df.dropna(subset=[col], inplace=True)
            df.index = range(len(df))
            if self.missing:
                synth.dropna(subset=[col], inplace=True)
                synth.index = range(len(synth))

        synth_df_propen = synth.copy()
        df_propen = df.copy()

        n1, n2 = (len(df_propen), len(synth_df_propen))
        N = n1 + n2
        cc = n2 / N

        df_propen['util_prop'] = [0] * n1
        synth_df_propen['util_prop'] = [1] * n2
        df_score = pd.concat([df_propen, synth_df_propen], ignore_index=True)
        df_score = df_score.astype({'util_prop': 'category'})
        df_score = df_score[synth_df_propen.columns.tolist()]

        X_df = df_score.iloc[:, :-1]
        y_df = df_score['util_prop']

        cat_cols = X_df.select_dtypes(CAT_COLS_DTYPES).columns.to_list()
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True)

        logistic = LogisticRegression(max_iter=10000)
        logistic.fit(X_df, y_df)

        proba_predict = logistic.predict_proba(X_df)
        logi_score = pd.Series([i[1] for i in proba_predict])
        logi_pmse = sum((logi_score - cc) ** 2) / N

        km = len(synth_df_propen.columns.tolist())

        pmseexp = km * ((1 - cc) ** 2) * cc / N
        logi_s_pmse = logi_pmse / pmseexp

        print('pMSE Ratio : %.4f' % (logi_s_pmse))

        # 분포 시각화
        features = synth.columns.tolist()
        if len(features) < 3:
            nrows = 1
            ncols = len(features)
        else:
            nrows = round(len(features) / 3) + 1
            ncols = 3

        plt.figure(figsize=(18, nrows * 6), dpi=300)

        for index, feature in enumerate(features):
            plt.subplot(nrows, ncols, index + 1)
            plt.hist((df[feature], synth[feature]), histtype='bar', density=True, label=('Raw', 'Synth'))
            plt.xticks(rotation=90, size=7)
            plt.legend()
            plt.title(feature)

        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.show()

        if detail :
            table_evaluator = TableEvaluator(df, synth)
            table_evaluator.visual_evaluation()