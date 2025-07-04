import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from edudata.validator import Validator
from edudata.processor import Processor

from edudata.processor import NAN_KEY
from edudata import NUM_COLS_DTYPES, CAT_COLS_DTYPES
from edudata.method import RANDOMFOREST_METHOD, CART_METHOD, METHODS_MAP, NA_METHODS


class Synthpop:
    def __init__(self,
                 method=None,
                 visit_sequence=None,
                 predictor_matrix=None,
                 proper=False,
                 cont_na=None,
                 smoothing=False,
                 default_method=RANDOMFOREST_METHOD,
                 numtocat=None,
                 catgroups=None,
                 seed=None,
                 missing=False,
                 outliers=False,
                 numtype=False,
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
        self.numtype = numtype
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

        self.validator.check_processor()
        processed_df = self.processor.preprocess(df, self.df_dtypes)
        self.processed_df_columns = processed_df.columns.tolist()
        self.n_processed_df_columns = len(self.processed_df_columns)
        self.validator.check_fit()
        self._fit(processed_df)
        print('training complete')

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

        for col, visit_step in self.visit_sequence.sort_values().items():
            col_method = METHODS_MAP[self.method[col]](dtype=self.df_dtypes[col], smoothing=self.smoothing[col],
                                                       proper=self.proper, random_state=self.seed)
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]
            col_method.fit(X_df=df[col_predictors], y_df=df[col])
            self.saved_methods[col] = col_method
            # (수정)
            # print('{}_trained'.format(col))

    def generate(self, k=None):
        self.k = k
        # check generate
        self.validator.check_generate()
        # generate
        synth_df = self._generate()
        # postprocess
        processed_synth_df = self.processor.postprocess(synth_df, self.seed)
        print('generating complete')

        return processed_synth_df

    def _generate(self):
        synth_df = pd.DataFrame(data=np.zeros([self.k, len(self.visit_sequence)]), columns=self.visit_sequence.index)

        for col, visit_step in self.visit_sequence.sort_values().items():
            col_method = self.saved_methods[col]
            col_predictors = self.predictor_matrix_columns[self.predictor_matrix.loc[col].to_numpy() == 1]

            synth_df[col] = col_method.predict(synth_df[col_predictors])

            if col in self.numtocat:
                synth_df[col] = synth_df[col].astype('category')

            if self.df_dtypes[col] in NUM_COLS_DTYPES and synth_df[col].notna().any():
                if self.numtype == 'int':
                    synth_df[col] = synth_df[col].astype(int)
                else:
                    synth_df[col] = synth_df[col].astype(self.df_dtypes[col])
            elif self.df_dtypes[col] in CAT_COLS_DTYPES and synth_df[col].notna().any():
                synth_df[col] = synth_df[col].astype(self.df_dtypes[col])

            if self.missing :
                if col in self.processor.processing_dict[NAN_KEY] and self.df_dtypes[col] in NUM_COLS_DTYPES and self.method[col] in NA_METHODS:
                    nan_indices = synth_df[self.processor.processing_dict[NAN_KEY][col]['col_nan_name']] != 0
                    synth_df.loc[nan_indices, col] = np.nan

        return synth_df

    def missing_df(self, df):
        assert self.missing is not False, 'missing attribute에 0보다 크고 1보다 작은 비율값을 입력해주세요.'
        count = 0
        np.random.seed(self.seed)
        temp_df = df.copy()
        if self.missing == 1:
            return temp_df
        else:
            while round(len(df) * len(df.columns) * self.missing) >= count:
                r_l = np.random.randint(0, len(df))
                c_l = np.random.randint(0, len(df.columns))
                if temp_df.iloc[r_l, c_l] is not np.nan:
                    temp_df.iloc[r_l, c_l] = np.nan
                    count += 1
            return temp_df

    def outliers_df(self, df):
        assert self.outliers is not False, 'outliers attribute에 0보다 크고 1보다 작은 비율값을 입력해주세요.'
        temp_df = df.copy()
        np.random.seed(self.seed)
        if self.outliers is True:
            for i in range(round(len(df)*len(df.columns)*0.01)):
                r_l = np.random.randint(0, len(df))
                c_l = np.random.randint(0, len(df.columns))
                if df.iloc[:, c_l].dtypes in['int64', 'float64']:
                    q1 = df.iloc[:, c_l].quantile(0.25)
                    q3 = df.iloc[:, c_l].quantile(0.75)
                    if self.numtype == 'int':
                        if i%2 == 0:
                            temp_df.iloc[r_l, c_l] = round(q1 - np.random.uniform(1.5, 2)*(q3-q1))
                        else:
                            temp_df.iloc[r_l, c_l] = round(q3 + np.random.uniform(1.5, 2)*(q3-q1))
                    else :
                        if i%2 == 0:
                            temp_df.iloc[r_l, c_l] = round(q1 - np.random.uniform(1.5, 2)*(q3-q1), 2)
                        else:
                            temp_df.iloc[r_l, c_l] = round(q3 + np.random.uniform(1.5, 2)*(q3-q1), 2)
            return temp_df

        elif self.outliers > 0:
            for i in range(round(len(df) * len(df.columns) * self.outliers)):
                r_l = np.random.randint(0, len(df) - 1)
                c_l = np.random.randint(0, len(df.columns) - 1)
                if df.iloc[:, c_l].dtypes in ['int64', 'float64']:
                    q1 = df.iloc[:, c_l].quantile(0.25)
                    q3 = df.iloc[:, c_l].quantile(0.75)
                    if self.numtype == 'int':
                        if i % 2 == 0:
                            temp_df.iloc[r_l, c_l] = round(q1 - np.random.uniform(1.5, 2) * (q3 - q1))
                        else:
                            temp_df.iloc[r_l, c_l] = round(q3 + np.random.uniform(1.5, 2) * (q3 - q1))
                    else:
                        if i % 2 == 0:
                            temp_df.iloc[r_l, c_l] = round(q1 - np.random.uniform(1.5, 2) * (q3 - q1), 2)
                        else:
                            temp_df.iloc[r_l, c_l] = round(q3 + np.random.uniform(1.5, 2) * (q3 - q1), 2)
            return temp_df

    def compare(self, df, synth, visualize=True, **kwargs):
        """
        원본 데이터와 합성 데이터를 비교하여 pMSE 및 pMSE Ratio를 계산합니다.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            원본 데이터셋
        synth : pandas.DataFrame
            합성 데이터셋
        visualize : bool, default=True
            분포 시각화 여부
        **kwargs : dict
            추가 매개변수 (하위 호환성을 위해 유지)
            
        Returns:
        --------
        tuple : (pMSE, pMSE_ratio)
            
        Note:
        -----
        table_evaluator 라이브러리는 의존성 충돌로 인해 비활성화되었습니다.
        상세한 평가가 필요한 경우 별도로 table_evaluator를 설치하여 사용하세요.
        """
        
        # detail 매개변수 사용 시 경고
        if 'detail' in kwargs:
            print("경고: detail 매개변수는 table_evaluator 의존성 제거로 인해 비활성화되었습니다.")
            print("기본 시각화만 제공됩니다.")
        
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

        print('pMSE : %.4f' % (logi_pmse))
        print('pMSE Ratio : %.4f' % (logi_s_pmse))

        # 기본 분포 시각화 (table_evaluator 대신 matplotlib 사용)
        if visualize:
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

        # table_evaluator 기능은 의존성 충돌로 인해 비활성화됨
        # 필요시 별도 설치 후 다음 코드 활성화:
        # 
        # try:
        #     from table_evaluator import TableEvaluator
        #     if kwargs.get('detail', False):
        #         for col in df.columns:
        #             if df.dtypes[col].name == 'category':
        #                 df[col] = df[col].astype('object')
        #             if synth.dtypes[col].name == 'category':
        #                 synth[col] = synth[col].astype('object')
        #         table_evaluator = TableEvaluator(df, synth)
        #         table_evaluator.visual_evaluation()
        # except ImportError:
        #     print("table_evaluator가 설치되지 않았습니다. 기본 시각화만 제공됩니다.")

        return logi_pmse, logi_s_pmse