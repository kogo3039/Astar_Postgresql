import os
import cv2
import pandas as pd
import sqlalchemy
import folium
import numpy as np
from sqlalchemy.sql import text

url = 'postgresql+psycopg2://ecomarine:Ecomarine1!@3.35.245.80:27720/ecomarine'

# lat, long를 map 상의 x, y 좌표료 변환
def latilong_2_xy(equator, meridian, height, width, latitude, longitude):

    if latitude > 0:
        x = equator - round(latitude * (equator / 84))
    elif latitude < 0:
        x = equator + round(-latitude * ((height - equator) / 80))
    else:
        x = equator

    if longitude < 0:
        longitude = 180 + (180 + longitude)
    distance_per_long = width / 360
    y = round(meridian + distance_per_long * longitude)
    if y > width:
        y = round(meridian - (360 - longitude) * distance_per_long)

    return x, y


# map상의 x와 y좌표를 lat, long로 변환해주는 함수
def xy_2_latilong(equator, meridian, height, width, x, y):

    latitude, longitude = 0, 0
    distance_per_latitude_north = 84 / equator
    distance_per_latitude_south = 80 / (height - equator)
    if x >= equator:
        latitude = (x - equator) * distance_per_latitude_south * -1
    elif x <= equator:
        latitude = (equator - x) * distance_per_latitude_north
    else:
        latitude = 0

    dateline = meridian + (width / 2)
    distance_per_longitude = 180 / (width / 2)
    if y > meridian and y <= dateline:
        longitude = (y - meridian) * distance_per_longitude
    elif y > meridian and y >= dateline:
        longitude = (180 - ((y - dateline) * distance_per_longitude)) * -1
    elif y < meridian and y <= dateline:
        longitude = (meridian - y) * distance_per_longitude * -1
    elif y < meridian and y >= dateline:
        pass  # non-case
    else:
        longitude = 0

    return latitude, longitude

# the shortest nodes from DB
def extract_node(source, target, loc):

    engine = sqlalchemy.create_engine(url)

    sql = f"""SELECT * FROM pgr_astar(
            'SELECT id, source, target, cost, reverse_cost, x1, y1, x2, y2 FROM {loc}_edge_table',
            {source}, {target}, heuristic := 3);"""

    with engine.connect().execution_options(autocommit=True) as conn:
            query = conn.execute(text(sql))

    df = pd.DataFrame(query.fetchall())
    if not df.empty:

        nodes = tuple(df['node'])

        return nodes
    else:
        print("There are no nodes!!!")
        exit()

# the shortest nodes x, y from DB
def extract_x_y(nodes, loc):

    engine = sqlalchemy.create_engine(url)

    sql = f"""SELECT x, y FROM {loc}_vertex_table 
            where id in {nodes};"""

    with engine.connect().execution_options(autocommit=True) as conn:
        query = conn.execute(text(sql))

    df = pd.DataFrame(query.fetchall())

    return df

# the shortest coordinate lat, long tranlation
def tranlate_the_shortest_coordinate(source, target, loc):

    equator = 234  # map 상의 적도를 지나는 y 좌표
    pacific_meridian = 74  # pacific_centered map 상에서의 본초자오선에 해당하는 x 좌표
    atlantic_meridian = 482  # atlantic_centered map 상에서의 본초자오선에 해당하는 x 좌표
    height = 475
    width = 1014
    p_loc = 'pacific'
    a_loc = 'atlantic'

    if loc == p_loc:

        nodes = extract_node(source, target, loc)
        nodes_lsts= []
        for i in range(len(nodes)):
            if i % 5 == 0:
                nodes_lsts.append(nodes[i])


        df = extract_x_y(tuple(nodes_lsts), loc)

        coordinate_lsts = []
        for i in range(df.shape[0]):
            x_y = list(df.loc[i])
            lat, long = xy_2_latilong(equator, pacific_meridian, height, width, x_y[0], x_y[1])
            coordinate_lsts.append((lat, long))

    elif loc == a_loc:

        nodes = extract_node(source, target, loc)
        df = extract_x_y(nodes, loc)

        coordinate_lsts = []
        for i in range(df.shape[0]):
            x_y = list(df.loc[i])
            lat, long = xy_2_latilong(equator, atlantic_meridian, height, width, x_y[0], x_y[1])
            coordinate_lsts.append((lat, long))

    return coordinate_lsts

# 지도 그리기
def astar_nodes_map(coordinates):

    seaMap = folium.Map(location=[35.95, 127.7], zoom_start=2)

    for coord in coordinates:

        if coord[1] < 0:
            marker= folium.Circle(location=[coord[0], coord[1] + 360],
                                   icon=folium.Icon(color='orange', icon='star')
                                   )
        else:
            marker = folium.Circle(location=[coord[0], coord[1]],
                                   icon=folium.Icon(color='orange', icon='star')
                                    )
        marker.add_to(seaMap)

    seaMap.save("nodeMap/nodes_map.html")

if __name__ == "__main__":

    # p_map = cv2.imread("worldMap/mercato_pacific.png", cv2.IMREAD_GRAYSCALE)
    # a_map = cv2.imread("worldMap/mercato_atlantic.png", cv2.IMREAD_GRAYSCALE)

    # map_size = np.array(p_map)  # map size는 1014 * 475로 동일
    # print(map_size.shape)
    # exit()
    equator = 234  # map 상의 적도를 지나는 y 좌표
    pacific_meridian = 74  # pacific_centered map 상에서의 본초자오선에 해당하는 x 좌표
    atlantic_meridian = 482  # atlantic_centered map 상에서의 본초자오선에 해당하는 x 좌표
    height = 475
    width = 1014
    p_loc = 'pacific'
    a_loc = 'atlantic'

    # lat, long = xy_2_latilong(equator, atlantic_meridian, height, width, 1014, 475)
    # print(lat, long)
    # exit()

    # s_lat_long = [36.97435897435898, 125.32544378698225]
    # e_lat_long = [37, -123]
    # panama = [206, 866]
    # italy = [137, 126]
    # x, y = latilong_2_xy(equator, pacific_meridian, height, width, panama[0], panama[1])
    # print(x, y)
    # lat, long = xy_2_latilong(equator, pacific_meridian, height, width, 137, 126)
    # print(lat, long)
    # exit()

    loc = p_loc
    source = 51024
    target = 72830

    coordinate_lsts = tranlate_the_shortest_coordinate(source, target, loc)


    astar_nodes_map(coordinate_lsts)

