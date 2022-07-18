from edudata.method.base import Method
from edudata.method.helpers import proper, smooth
from edudata.method.empty import EmptyMethod
from edudata.method.sample import SampleMethod
from edudata.method.cart import CARTMethod
from edudata.method.norm import NormMethod
from edudata.method.normrank import NormRankMethod
from edudata.method.polyreg import PolyregMethod
from edudata.method.ramdomforest import RandomforestMethod


EMPTY_METHOD = ''
SAMPLE_METHOD = 'sample'
CART_METHOD = 'cart'
PARAMETRIC_METHOD = 'parametric'
NORM_METHOD = 'norm'
NORMRANK_METHOD = 'normrank'
POLYREG_METHOD = 'polyreg'
RANDOMFOREST_METHOD = 'randomforest'


METHODS_MAP = {EMPTY_METHOD: EmptyMethod,
               SAMPLE_METHOD: SampleMethod,
               CART_METHOD: CARTMethod,
               NORM_METHOD: NormMethod,
               NORMRANK_METHOD: NormRankMethod,
               POLYREG_METHOD: PolyregMethod,
               RANDOMFOREST_METHOD: RandomforestMethod
               }

ALL_METHODS = (EMPTY_METHOD, SAMPLE_METHOD, CART_METHOD, PARAMETRIC_METHOD, NORM_METHOD, NORMRANK_METHOD, POLYREG_METHOD, RANDOMFOREST_METHOD)
DEFAULT_METHODS = (CART_METHOD, PARAMETRIC_METHOD, RANDOMFOREST_METHOD)
INIT_METHODS = (SAMPLE_METHOD, CART_METHOD, PARAMETRIC_METHOD, RANDOMFOREST_METHOD)
NA_METHODS = (SAMPLE_METHOD, CART_METHOD, NORM_METHOD, NORMRANK_METHOD, POLYREG_METHOD, RANDOMFOREST_METHOD)

#(수정_추가)
PARAMETRIC_METHOD_MAP = {'int64': NORMRANK_METHOD,
                         'float64': NORMRANK_METHOD,
                         'datetime64[ns]': NORMRANK_METHOD,
                         'bool': POLYREG_METHOD,
                         'category': POLYREG_METHOD,
                         'object': POLYREG_METHOD
                         }

CART_METHOD_MAP = {'int64': CART_METHOD,
                   'float64': CART_METHOD,
                   'datetime64[ns]': CART_METHOD,
                   'bool': CART_METHOD,
                   'category': CART_METHOD,
                   'object': CART_METHOD
                   }

RANDOMFOREST_METHOD_MAP = {'int64': RANDOMFOREST_METHOD,
                   'float64': RANDOMFOREST_METHOD,
                   'datetime64[ns]': RANDOMFOREST_METHOD,
                   'bool': RANDOMFOREST_METHOD,
                   'category': RANDOMFOREST_METHOD,
                   'object': RANDOMFOREST_METHOD
                   }

SAMPLE_METHOD_MAP = {'int64': SAMPLE_METHOD,
                     'float64': SAMPLE_METHOD,
                     'datetime64[ns]': SAMPLE_METHOD,
                     'bool': SAMPLE_METHOD,
                     'category': SAMPLE_METHOD,
                     'object': SAMPLE_METHOD
                     }

DEFAULT_METHODS_MAP = {CART_METHOD: CART_METHOD_MAP,
                       PARAMETRIC_METHOD: PARAMETRIC_METHOD_MAP,
                       RANDOMFOREST_METHOD: RANDOMFOREST_METHOD_MAP
                       }


INIT_METHODS_MAP = DEFAULT_METHODS_MAP.copy()
INIT_METHODS_MAP[SAMPLE_METHOD] = SAMPLE_METHOD_MAP

CONT_TO_CAT_METHODS_MAP = {SAMPLE_METHOD: SAMPLE_METHOD,
                           CART_METHOD: CART_METHOD,
                           NORM_METHOD: POLYREG_METHOD,
                           NORMRANK_METHOD: POLYREG_METHOD,
                           POLYREG_METHOD: POLYREG_METHOD,
                           RANDOMFOREST_METHOD: RANDOMFOREST_METHOD
                           }
