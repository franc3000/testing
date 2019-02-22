import pandas as pd
from buyer_class import BuyerClass

input_dict = {
    "grantee": [
        "Situation Solutions One Inc",
        "Property Owner 5 LLC",
        "Show Me the Money LLC",
        "Dr Horton Crown LLC",
        "Evervest LLC",
    ],
    "grantee_mail_address_line_1": [
        "5701 Princess Anne Rd Ste G",
        "PO Box 4090",
        "2212 E Williams Field Rd Ste 220",
        "1371 Dogwood Dr SW",
        "305 Saint Louis Ave # 432",
    ],
    "grantee_mail_address_last_line": [
        "Virginia Beach VA 23462-3253",
        "Scottsdale AZ 85261-4090",
        "Gilbert AZ 85295-0774",
        "Conyers GA 30012-5127",
        "Cicero IN 46034-5005",
    ],
}

df_input = pd.DataFrame.from_dict(input_dict)

buyer_class = BuyerClass(
    input_features=["is_flipper", "bad_targets", "is_builder"],
    output_feature_names=["is_flipper", "bad_targets", "is_builder"]
)


def test_is_flipper():
    df = buyer_class.fit_transform(df_input)
    assert df["is_flipper"].values.tolist() == [1, 0, -1, 0, -1]

    
def test_is_bad_targets():
    df = buyer_class.fit_transform(df_input)
    assert df["bad_targets"].values.tolist() == [0, 1, -1, 1, -1]

    
def test_is_builder():
    df = buyer_class.fit_transform(df_input)
    assert df["is_builder"].values.tolist() == [0, 0, -1, 1, -1]
    
