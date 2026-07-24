"""
Author: Elko Gerville-Reache
Date Created: 2025-12-23
Date Modified: 2026-06-16
Description:
    Text utility functions.
"""

from textwrap import dedent

from matplotlib import font_manager

from visualastro.core.config import config, _resolve_default, _Unset, _UNSET


def pretty_table(
    headers: list[str] | None,
    data: list[list],
    precision: int | _Unset = _UNSET,
    sci_notation: bool | _Unset = _UNSET,
    pad: int | _Unset = _UNSET
) -> str:
    """
    Format a pretty table given a 2D list of table cells. Each cell
    can be either a numerical value with or without units, a string, or None.

    Use print_pretty_table to print the output directly.

    Parameters
    ----------
    headers : list[str] | None
        Names of each column. If None, skips rendering the header row
        and starts directly with the separator line.
    data : list[list]
        2D list containing each table cell. Should have shape
        (Nrows, Ncols). Each cell can be:

        * Numerical value (int, float)
        * Quantity object with units (e.g., astropy.units.Quantity)
        * String
        * None(renders as empty string)

    precision : int | _Unset, optional, default=_UNSET
        Number of decimal places for numerical formatting.
        If `_UNSET`, uses `config.table_precision`.
    sci_notation : bool | _Unset, optional, default=_UNSET
        If `True`, formats numbers in scientific notation (e.g., 1.23e-04).
        If `False`, formats as fixed-point decimals (e.g., 0.000123).
        If `_UNSET`, uses `config.table_sci_notation`.
    pad : int | _Unset, optional, default=_UNSET
        Number of spaces between columns for visual separation.
        If `_UNSET`, uses `config.table_column_pad`.

    Returns
    -------
    table : str
        Formatted table string with aligned columns, ready to print.
    """

    precision = _resolve_default(precision, config.table_precision)
    sci_notation = _resolve_default(sci_notation, config.table_sci_notation)
    pad = _resolve_default(pad, config.table_column_pad)

    if not isinstance(pad, int) or pad <= 0:
        raise ValueError(
            "`pad` must be an integer > 0!"
        )
    col_pad = ' ' * pad

    def _format_table_cell(cell) -> str:
        """
        Format each table cell for consistent styling.

        Parameters
        ----------
        cell : float | int | Quantity | str | None
            Table cell to format

        Returns
        -------
        formatted_cell : str
        """
        if cell is None or cell == '':
            return ''
        elif isinstance(cell, str):
            return cell

        if hasattr(cell, 'unit'):
            value = cell.value
            unit = f' {cell.unit}' if True else ''
        else:
            value = cell
            unit = ''

        if sci_notation:
            return f'{value:.{precision}e}{unit}'
        else:
            return f'{value:.{precision}f}{unit}'

    # format all cells
    formatted_data = [[_format_table_cell(cell) for cell in row] for row in data]

    ncols = len(headers) if headers is not None else len(formatted_data[0])

    # compute column widths
    col_widths = []
    for j in range(ncols):
        if headers is not None:
            max_header = len(headers[j])
        else:
            max_header = 0
        max_data = max((len(row[j]) for row in formatted_data), default=0)
        col_widths.append(max(max_header, max_data))

    table = []
    if headers is not None:
        # left align headers and pad to column width
        header_names = [f'{headers[j]:<{col_widths[j]}}' for j in range(len(headers))]
        # add padding between each column
        header_row = col_pad.join(header_names)
        table.append(header_row)
        separator = '─' * len(header_row)
    else:
        row_len = sum(col_widths) + len(col_pad)*(ncols-1)
        separator = '─' * row_len

    table.append(separator)

    # fill table rows
    for row_data in formatted_data:
        row = [f'{row_data[j]:<{col_widths[j]}}' for j in range(ncols)]
        table.append(col_pad.join(row))

    table = '\n'.join(table)

    return table


def print_pretty_table(
    headers: list[str] | None,
    data: list[list],
    precision: int | _Unset = _UNSET,
    sci_notation: bool | _Unset = _UNSET,
    pad: int | _Unset = _UNSET
) -> None:
    """
    Format a pretty table given a 2D list of table cells and print it. Each cell
    can be either a numerical value with or without units, a string, or None.

    Parameters
    ----------
    headers : list[str] | None
        Names of each column. If None, skips rendering the header row
        and starts directly with the separator line.
    data : list[list]
        2D list containing each table cell. Should have shape
        (Nrows, Ncols). Each cell can be:

        * Numerical value (int, float)
        * Quantity object with units (e.g., astropy.units.Quantity)
        * String
        * None(renders as empty string)

    precision : int | _Unset, optional, default=_UNSET
        Number of decimal places for numerical formatting.
        If `_UNSET`, uses `config.table_precision`.
    sci_notation : bool | _Unset, optional, default=_UNSET
        If `True`, formats numbers in scientific notation (e.g., 1.23e-04).
        If `False`, formats as fixed-point decimals (e.g., 0.000123).
        If `_UNSET`, uses `config.table_sci_notation`.
    pad : int | _Unset, optional, default=_UNSET
        Number of spaces between columns for visual separation.
        If `_UNSET`, uses `config.table_column_pad`.
    """
    table = pretty_table(
        headers, data, precision=precision, sci_notation=sci_notation, pad=pad
    )

    print(table)


def find_font(fontname: str) -> None:
    """
    Find and print the path of a registered matplotlib font.

    Parameters
    ----------
    fontname : str
        Name of font. Is case sensitive.
    """
    fontname = str(fontname)
    for font in font_manager.fontManager.ttflist:
        if fontname in font.name:
            print(font.name, '->', font.fname)


def print_font_info(fontname: str) -> None:
    """
    Print a matplotlib registered font's attributes.

    Parameters
    ----------
    fontname : str
        Name of font. Is case sensitive.
    """
    for f in font_manager.fontManager.ttflist:
        if f.name == fontname:
            print(dedent(f"""\
                Family : {f.name}
                Style  : {f.style}
                Weight : {f.weight}
                Stretch: {f.stretch}
                Variant: {f.variant}
                Size   : {f.size}
                File   : {f.fname}
            """))
