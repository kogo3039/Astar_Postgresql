import cv2
import pandas as pd
equator = 234 # map 상의 적도를 지나는 y 좌표
pacific_meridian = 74  # pacific_centered map 상에서의 본초자오선에 해당하는 x 좌표
atlantic_meridian = 482  # atlantic_centered map 상에서의 본초자오선에 해당하는 x 좌표

def latilong_2_xy(equator, meridian, height, width, latitude, longitude):
    if latitude > 0:
        y = equator - round(latitude * (equator / 84))
    elif latitude < 0:
        y = equator + round(-latitude * ((height - equator) / 80))
    else:
        y = equator

    if longitude < 0:
        longitude = 180 + (180 + longitude)
    distance_per_long = width / 360
    x = round(meridian + distance_per_long * longitude)
    if x > width:
        x = round(meridian - (360 - longitude) * distance_per_long)

    return x, y


# map상의 x와 y좌표를 latitude, longitude로 변환해주는 함수
def xy_2_latilong(equator, meridian, height, width, x, y):
    latitude, longitude = 0, 0
    distance_per_latitude_north = 84 / equator
    distance_per_latitude_south = 80 / (height - equator)
    if y >= equator:
        latitude = (y - equator) * distance_per_latitude_south * -1
    elif y <= equator:
        latitude = (equator - y) * distance_per_latitude_north
    else:
        latitude = 0

    dateline = meridian + (width / 2)
    distance_per_longitude = 180 / (width / 2)
    if x > meridian and x <= dateline:
        longitude = (x - meridian) * distance_per_longitude
    elif x > meridian and x >= dateline:
        longitude = (180 - ((x - dateline) * distance_per_longitude)) * -1
    elif x < meridian and x <= dateline:
        longitude = (meridian - x) * distance_per_longitude * -1
    elif x < meridian and x >= dateline:
        pass  # non-case
    else:
        longitude = 0

    return latitude, longitude

if __name__ == "__main__":

    equator = 234  # map 상의 적도를 지나는 y 좌표
    pacific_meridian = 74  # pacific_centered map 상에서의 본초자오선에 해당하는 x 좌표
    atlantic_meridian = 482  # atlantic_centered map 상에서의 본초자오선에 해당하는 x 좌표
    p_map = cv2.imread("mercato_pacific.png", cv2.IMREAD_GRAYSCALE)
    a_map = cv2.imread("mercato_atlantic.png", cv2.IMREAD_GRAYSCALE)
    height = p_map.shape[0]  # 1014
    width = p_map.shape[1]  # 475

    x1, y1 = latilong_2_xy(equator, pacific_meridian, height, width, 29.9863, 121.8401)
    x2, y2 = latilong_2_xy(equator, atlantic_meridian, height, width, 29.9863, 121.8401)

    lat, long = xy_2_latilong(equator, atlantic_meridian, height, width, 0, 0)
    print(lat, long)

    # columns = [i + 1 for i in range(1014)]
    # df = pd.DataFrame(p_map, columns = columns)
    # df.to_csv("map_csv.csv", index=False)

