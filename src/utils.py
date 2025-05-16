import pandas as pd
from tabulate import tabulate
from datetime import datetime
import re

def save_to_excel(data: list, filename: str):
    """
    Saves data to an Excel file.
    """
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)

def save_to_markdown(data: list, filename: str):
    """
    Saves data to a Markdown file.
    """
    md_table = tabulate(data, headers="keys", tablefmt="pipe")
    with open(filename, "w") as f:
        f.write(md_table)

def generate_filename(website: str, extension: str):
    """
    Generates a filename based on the website, date, and time.
    """
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    website = re.sub(r"[^a-zA-Z0-9]", "_", website)  # Sanitize website name
    filename = f"scrape_{website}+{date_time_str}.{extension}"
    return filename
