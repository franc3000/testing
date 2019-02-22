import logging
import json
import numpy as np
import pandas as pd
from typing import Union, List

from sklearn.base import TransformerMixin, BaseEstimator


def is_numeric(numpy_array):
    if not np.issubdtype(numpy_array.dtype, np.number):
        try:
            tmp = numpy_array.astype(np.float32)
        except (ValueError, TypeError) as error:
            return False

    return True


class ExtractorBase(TransformerMixin, BaseEstimator):
    """
    Base class of the extractors.  Inherit this base class to all types of extractors so that the extractors
    have `get_feature_names` and `get_feature_types` methods which are helpful for pipeline and feature union
    construction.
    """

    def __init__(self, input_features: Union[str, List[str]], output_feature_names=None, output_feature_types=None):
        """

        :param input_features: the name or a list of names of the input features that need to be transformed
        :param output_feature_names: the name or a list of names of the output/transformed features
        :param output_feature_types: a list of the types (numeric or categorical) of the transformed features.
        """
        if isinstance(input_features, str):
            input_features = [input_features]

        if output_feature_names:
            if isinstance(output_feature_names, str):
                output_feature_names = [output_feature_names]
            assert isinstance(output_feature_names, type(input_features))

        if output_feature_types:
            assert isinstance(output_feature_types, str) or isinstance(output_feature_types, list)

        self.input_features = input_features
        self.output_feature_names = output_feature_names
        self.output_feature_types = output_feature_types

        self.cached_items = None

    @property
    def log(self):
        return logging.getLogger(self.__class__.__name__)

    def _check_input_features(self, X: pd.DataFrame):
        """
        check if the input features are contained in X.  This can be put at the beginning of fit method.
        :param X:
        :return:
        """
        for column in self.input_features:
            msg = "column {} does not exist in X".format(column)
            assert column in X.columns, msg

    def _post_transform(self, _X: pd.DataFrame):
        """
        automatically check the types of the transformed features if output_feature_types is not specified
        when initializing the extractor.  This can be put at the end of the transform method.
        :param _X:
        :return:
        """
        if not self.output_feature_names:
            self.output_feature_names = _X.columns.tolist()

        if not self.output_feature_types:
            self.output_feature_types = []

            for column in _X.columns:
                if is_numeric(_X[column].values):
                    self.output_feature_types.append("numeric")
                else:
                    self.output_feature_types.append("categorical")
        elif isinstance(self.output_feature_types, str) and _X.shape[1] > 1:
            f_type = self.output_feature_types
            self.output_feature_types = [f_type for i in range(_X.shape[1])]
        elif isinstance(self.output_feature_types, list):
            assert len(self.output_feature_types) == _X.shape[1]

        self.log.info("finished transforming {} to {}".format(self.input_features, self.output_feature_names))

    def get_feature_names(self):
        """
        output a list of the names of the transformed features
        :return: list of transformed feature names
        """
        if isinstance(self.output_feature_names, str):
            self.output_feature_names = [self.output_feature_names]
        return self.output_feature_names

    def get_feature_types(self):
        if isinstance(self.output_feature_types, str):
            self.output_feature_types = [self.output_feature_types]
        return self.output_feature_types

    def fit(self, X, y=None, **fit_params):
        return self


class BuyerClass(ExtractorBase):

    builtin_features = [
        "is_flipper",
        "bad_targets",
        "is_builder",
        "is_ibuyer",
        "is_fund",
        "is_wholesaler",
        "is_landlord",
        "is_service",
    ]

    def __init__(
        self,
        input_features=None,
        buyer_class_table="info_table.json",
        output_feature_names=None,
        output_feature_types="categorical"
    ):

        if input_features is None and output_feature_names is None:
            input_features = self.builtin_features
            output_feature_names = self.builtin_features
        else:
            assert (isinstance(input_features, str) or isinstance(input_features, list))
            assert (isinstance(output_feature_names, str) or isinstance(output_feature_names, list))

        super().__init__(
            input_features=input_features,
            output_feature_names=output_feature_names,
            output_feature_types=output_feature_types
        )

        # verify inputs
        for feature in self.input_features:
            if feature not in self.builtin_features:
                msg = "feature {} not available, expected to be in {}".format(feature, self.builtin_features)
                raise ValueError(msg)

        self.buyer_class_table = buyer_class_table

    def _transform(self, df: pd.DataFrame):

        for column in ["grantee", "grantee_mail_address_line_1", "grantee_mail_address_last_line"]:
            df[column] = df[column].fillna("NULL")

        with open(self.buyer_class_table, "r") as fp:
            df_buyer_class = pd.DataFrame.from_dict(json.load(fp))

        # apply merge by buyer name and address
        df_merge = pd.merge(
            left=df,
            right=df_buyer_class,
            on=["grantee", "grantee_mail_address_line_1", "grantee_mail_address_last_line"],
            how="left",
        )
        df_merge = df_merge.fillna(-1.)
        return df_merge[self.get_feature_names()]

    def fit(self, X: pd.DataFrame, y=None, **fit_params):
        # check if "grantee", "grantee_mail_address_line_1", and "grantee_mail_address_last_line" are
        # included in the input dataframe
        for column in ["grantee", "grantee_mail_address_line_1", "grantee_mail_address_last_line"]:
            if column not in X.columns:
                msg = f"feature `{column}` not found in input dataframe"
                raise ValueError(msg)

        return self

    def transform(self, X: pd.DataFrame):
        """transform input df"""
        _X = self._transform(X)
        self._post_transform(_X)
        return _X
