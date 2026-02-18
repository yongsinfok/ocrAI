"""Data export functionality."""
from typing import Any, List, Dict, Optional
import io
import logging
import pandas as pd
import json

logger = logging.getLogger(__name__)

class ExportManager:
    """Manage data export to various formats."""

    def to_excel(self, data: Any, filename: str = "export") -> bytes:
        """Convert data to Excel format.

        Args:
            data: Data to export (list of lists, dict, or DataFrame)
            filename: Base filename for the Excel file

        Returns:
            Excel file as bytes
        """
        output = io.BytesIO()

        # Convert to DataFrame
        df = self._to_dataframe(data)

        # Write to Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=filename[:31])  # Excel sheet name limit

        output.seek(0)
        return output.read()

    def to_csv(self, data: Any) -> str:
        """Convert data to CSV format.

        Args:
            data: Data to export

        Returns:
            CSV string
        """
        df = self._to_dataframe(data)
        return df.to_csv(index=False)

    def to_json(self, data: Any) -> str:
        """Convert data to JSON format.

        Args:
            data: Data to export

        Returns:
            JSON string
        """
        if isinstance(data, str):
            return data

        df = self._to_dataframe(data)
        # Convert to dict first, then to JSON for better control
        return json.dumps(df.to_dict(orient='records'), ensure_ascii=False, indent=2)

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert various data types to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, list):
            if not data:
                return pd.DataFrame()

            # Check if list of dicts
            if isinstance(data[0], dict):
                return pd.DataFrame(data)

            # Check if list of lists (table format)
            if isinstance(data[0], list):
                # First row as headers
                return pd.DataFrame(data[1:], columns=data[0])

            # Simple list
            return pd.DataFrame(data, columns=["Value"])

        # String - try to parse as JSON
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return self._to_dataframe(parsed)
            except:
                return pd.DataFrame({"Value": [data]})

        # Fallback
        return pd.DataFrame({"Value": [str(data)]})

    def format_by_template(self, data: Any, template: str) -> str:
        """Format data using a template string.

        Args:
            data: Data to format
            template: Template string with {placeholders}

        Returns:
            Formatted string
        """
        df = self._to_dataframe(data)

        # Convert DataFrame to dict for template formatting
        if len(df) > 0:
            row = df.iloc[0].to_dict()
        else:
            row = {}

        try:
            return template.format(**row)
        except KeyError as e:
            logger.warning(f"Template key error: {e}")
            return template.format(data=str(data))
