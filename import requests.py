
#import sys
#sys.path.append('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')
import requests
import datetime

# OpenWeatherMap API密钥
api_key = "YOUR_API_KEY"

# 指定经纬度和日期
latitude = 40.7128  # 纬度
longitude = -74.0060  # 经度
target_date = "2023-08-10"  # 指定日期

# 构建API请求URL
url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine?lat={latitude}&lon={longitude}&dt={target_date}&appid={api_key}"

# 发送API请求
response = requests.get(url)
data = response.json()

# 提取最高温度和湿度数据
if "hourly" in data:
    hourly_data = data["hourly"]
    for hour in hourly_data:
        timestamp = hour.get("dt")
        temperature = hour.get("temp")
        humidity = hour.get("humidity")
        
        # 将时间戳转换为日期和时间
        time_obj = datetime.datetime.fromtimestamp(timestamp)
        
        # 打印日期、时间、最高温度和湿度
        print(f"Date: {time_obj.date()}, Time: {time_obj.time()}, Max Temperature: {temperature}, Humidity: {humidity}")
else:
    print("No hourly data available.")