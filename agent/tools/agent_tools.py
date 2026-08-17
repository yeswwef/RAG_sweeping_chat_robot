import os

import httpx
from langchain_core.tools import tool

from utils.config_handler import agent_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from rag.rag_service import get_rag_service
import random
from datetime import datetime
from zoneinfo import ZoneInfo

@tool(description="在向量检索中检索参考资料")
def rag_summarize(query: str) -> str:
    return get_rag_service().rag_summarize(query)



####################################################################################
# WMO 天气代码 -> 中文（Open-Meteo 用）
WMO_WEATHER_CODE = {
    0: "晴", 1: "大部晴朗", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "小毛毛雨", 55: "强毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
}

@tool(description="联网获取指定城市的实时天气与未来三天预报，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    # 兼容"合肥市"这类带"市"的写法：先去掉后缀再查
    if city.endswith("市") and len(city) > 1:
        city = city[:-1]
    try:
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"}, timeout=10,
        )
        #geo.raise_for_status()：如果 HTTP 状态码不是 200（比如 404 或 500），主动抛出异常，进入 except 分支
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            return f"未找到城市{city}"
        #纬度(latitude) 和 经度(longitude)
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        """
        current：当前实时数据（气温、湿度、风速、天气代码）。
        daily：未来三天的数据（最高温、最低温、天气代码）。
        forecast_days=3：只取今天及后两天。
        timezone：强制使用中国时区，否则默认 UTC 时间可能会导致日期显示偏差
        """
        fc = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "forecast_days": 3, "timezone": "Asia/Shanghai",
            },
            timeout=10,
        )
        fc.raise_for_status()
        data = fc.json()
        cur = data["current"]#当前天气
        desc = WMO_WEATHER_CODE.get(cur["weather_code"], cur["weather_code"])#天气状况晴阴
        daily = data["daily"]
        
        parts = [
            #daily['time'] 是 Open-Meteo 天气 API 自动附带返回的默认字段
            f"{daily['time'][i]} {daily['temperature_2m_min'][i]}~{daily['temperature_2m_max'][i]}℃"
            for i in range(len(daily["time"]))
        ]

        return (
            f"城市{city}当前天气：{desc}，气温{cur['temperature_2m']}℃，"
            f"空气湿度{cur['relative_humidity_2m']}%，风速{cur['wind_speed_10m']}km/h。"
            f"未来三天预报：{'；'.join(parts)}"
        )
    except Exception as e:
        logger.error(f"[get_weather]查询城市{city}失败: {e}")
        return f"查询城市{city}天气失败，请稍后再试"



####################################################################################



@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


user_ids = [
    "1001", "1002", "1003", "1004", "1005",
    "1006", "1007", "1008", "1009", "1010",
]


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


from datetime import datetime
from zoneinfo import ZoneInfo

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m")


external_data = {}


def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(
                f"外部数据文件{external_data_path}不存在"
            )
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr = line.strip().split(",")
                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")
                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(
    description="从外部系统中获取指定用户在指定月份的使用记录，"
    "以纯字符串形式返回， 如果未检索到返回空字符串"
)
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()
    try:
        return str(external_data[user_id][month])
    except KeyError:
        logger.warning(
            f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据"
        )
        return ""


@tool(
    description="无入参，无返回值，调用后触发中间件自动为报告生成的场景"
    "动态注入上下文信息，为后续提示词切换提供上下文信息"
)
def fill_context_for_report():
    return "fill_context_for_report已调用"
