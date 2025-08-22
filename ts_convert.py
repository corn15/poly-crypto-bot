# import datetime
# import pytz
#
# def to_est_format(ts_ms):
#     utc_time = datetime.datetime.fromtimestamp(ts_ms, tz=pytz.utc)
#     eastern_tz = pytz.timezone('America/New_York')
#     et_time = utc_time.astimezone(eastern_tz)
#     formatted_string = et_time.strftime('%B-%d-%I%p-%Z').lower()
#     final_string = et_time.strftime('%B-%d-%I%p-%Z').replace(' 0', ' ').replace(' PM', 'pm').replace(' AM', 'am')
#     return final_string

import datetime

import pytz


def to_est_format(ts):
    utc_time = datetime.datetime.fromtimestamp(ts, tz=pytz.utc)
    eastern_tz = pytz.timezone("America/New_York")
    et_time = utc_time.astimezone(eastern_tz)

    # Format: august-13-6pm-et
    month = et_time.strftime("%B").lower()
    day = str(et_time.day)
    hour = et_time.hour

    if hour == 0:
        time_str = "12am"
    elif hour < 12:
        time_str = f"{hour}am"
    elif hour == 12:
        time_str = "12pm"
    else:
        time_str = f"{hour - 12}pm"

    formatted_string = f"{month}-{day}-{time_str}-et"
    print(f"時間格式: {formatted_string}")
    return formatted_string
