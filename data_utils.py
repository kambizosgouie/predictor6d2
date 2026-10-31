from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


def excel_sheet_names(file_bytes: bytes) -> list[str]:
    with pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl") as workbook:
        return workbook.sheet_names


def read_tabular_data(
    file_bytes: bytes,
    file_name: str,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(BytesIO(file_bytes))
    elif suffix in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name=sheet_name,
            engine="openpyxl",
        )
    else:
        raise ValueError("Upload a CSV, XLSX, or XLSM file.")

    names: list[str] = []
    seen: dict[str, int] = {}
    for index, column in enumerate(frame.columns, start=1):
        name = str(column).strip() or f"Column {index}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        names.append(name if count == 1 else f"{name} ({count})")
    frame.columns = names
    return frame


def numeric_data_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.loc[:, numeric.notna().any()]
