import pytz


def get_time_str(input_time):
    # Convert input time to Eastern Time
    et_tz = pytz.timezone("US/Eastern")
    current_time = input_time.astimezone(et_tz)

    # Get month name (lowercase)
    month_name = current_time.strftime("%B").lower()

    # Get day with zero padding
    day = current_time.strftime("%d").lstrip("0")

    # Get hour in 12-hour format with am/pm
    hour = current_time.strftime("%I").lstrip("0")  # Remove leading zero
    if not hour:  # If hour was "00", make it "12"
        hour = "12"
    am_pm = current_time.strftime("%p").lower()

    return f"{month_name}-{day}-{hour}{am_pm}-et"
