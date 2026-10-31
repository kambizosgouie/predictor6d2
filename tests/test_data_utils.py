import unittest
from io import BytesIO

import pandas as pd

from data_utils import excel_sheet_names, numeric_data_frame, read_tabular_data


class DataUtilsTests(unittest.TestCase):
    def test_read_csv_and_convert_numeric_columns(self):
        frame = read_tabular_data(
            b"temp,reading,note\n20,1.5,low\n30,not-a-number,high\n",
            "measurements.csv",
        )

        numeric = numeric_data_frame(frame)

        self.assertEqual(list(numeric.columns), ["temp", "reading"])
        self.assertEqual(numeric.loc[0, "temp"], 20)
        self.assertTrue(pd.isna(numeric.loc[1, "reading"]))

    def test_read_selected_excel_sheet(self):
        contents = BytesIO()
        with pd.ExcelWriter(contents, engine="openpyxl") as writer:
            pd.DataFrame({"temp": [20, 30], "density": [1.0, 0.9]}).to_excel(
                writer, sheet_name="Measurements", index=False
            )
            pd.DataFrame({"ignored": [1]}).to_excel(
                writer, sheet_name="Other", index=False
            )

        file_bytes = contents.getvalue()

        self.assertEqual(excel_sheet_names(file_bytes), ["Measurements", "Other"])
        frame = read_tabular_data(file_bytes, "data.xlsx", "Measurements")
        self.assertEqual(list(frame.columns), ["temp", "density"])
        self.assertEqual(frame["density"].tolist(), [1.0, 0.9])

    def test_duplicate_excel_headers_get_unique_names(self):
        contents = BytesIO()
        pd.DataFrame([[1, 2]], columns=["value", " value "]).to_excel(
            contents, index=False, engine="openpyxl"
        )

        frame = read_tabular_data(contents.getvalue(), "data.xlsx", "Sheet1")

        self.assertEqual(list(frame.columns), ["value", "value (2)"])
