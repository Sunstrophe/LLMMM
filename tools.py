from langchain_core.tools import tool
from datetime import datetime
import pytz


@tool
def clock() -> str:
    """Tell what time it currently is"""
    current_time = datetime.now(pytz.timezone("Europe/Stockholm"))
    return current_time.strftime("%H:%M:%S")
