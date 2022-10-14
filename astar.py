import cv2
import folium
from datetime import datetime
import numpy as np
from skimage import io
from haversine import haversine
from collections import defaultdict
from tqdm import tqdm
import math

class Worldmaps:
    def __init__(self, start_lati, start_long, end_lati, end_long):
        self.origin = (start_lati, start_long) # origin (latitude, longitude)
        self.destination = (end_lati, end_long) # destination (latitude, longitude)

        # 북위 84 ~ 남위 -80, 서경 -180 ~ 동경 180을 벗어나는지 체크
        if start_lati > 84 or start_lati < -80 or start_long > 180 or start_long < -180 or \
                end_lati > 84 or end_lati < -80 or end_long >= 180 or end_long < -180:
            raise Exception("Error: out of latitude/longitude ranges")

        self.pacific_map = cv2.imread("mercato_pacific.png", cv2.IMREAD_GRAYSCALE)
        self.route_on_pacific_map = cv2.imread("mercato_pacific.png")
        self.atlantic_map = cv2.imread("mercato_atlantic.png", cv2.IMREAD_GRAYSCALE)
        self.route_on_atlantic_map = cv2.imread("mercato_atlantic.png")

        self.map_size = np.array(self.pacific_map) # map size는 1014 * 475로 동일
        self.height = self.map_size.shape[0] # 1014
        self.width = self.map_size.shape[1] # 475

        self.equator = 234 # map 상의 적도를 지나는 y 좌표
        self.pacific_meridian = 74  # pacific_centered map 상에서의 본초자오선에 해당하는 x 좌표
        self.atlantic_meridian = 482  # atlantic_centered map 상에서의 본초자오선에 해당하는 x 좌표

        self.start_x_pacific, self.start_y_pacific = self.latilong_2_xy(self.equator, self.pacific_meridian, self.height, self.width, start_lati, start_long)
        self.end_x_pacific, self.end_y_pacific = self.latilong_2_xy(self.equator, self.pacific_meridian, self.height, self.width, end_lati, end_long)
        self.start_x_atlantic, self.start_y_atlantic = self.latilong_2_xy(self.equator, self.atlantic_meridian, self.height, self.width, start_lati, start_long)
        self.end_x_atlantic, self.end_y_atlantic = self.latilong_2_xy(self.equator, self.atlantic_meridian, self.height, self.width, end_lati, end_long)

        if np.array(self.pacific_map)[self.start_y_pacific][self.start_x_pacific] != 255:
            raise Exception("Error: Non-sea coordinates (start coordinates", self.start_x_pacific, self.start_y_pacific, ")")
        elif np.array(self.pacific_map)[self.end_y_pacific][self.end_x_pacific] != 255:
            raise Exception("Error: Non-sea coordinates (end coordinates", self.end_x_pacific, self.end_y_pacific, ")")
        if np.array(self.atlantic_map)[self.start_y_atlantic][self.start_x_atlantic] != 255:
            raise Exception("Error: Non-sea coordinates (start coordinates", self.start_x_atlantic, self.start_y_atlantic, ")")
        elif np.array(self.atlantic_map)[self.end_y_atlantic][self.end_x_atlantic] != 255:
            raise Exception("Error: Non-sea coordinates (end coordinates", self.end_x_atlantic, self.end_y_atlantic, ")")
        #if self.pacific_map[self.start_y_pacific][self.start_x_pacific] < 220:
        #     self.start_y_pacific, self.start_x_pacific = \
        #         self.coordinate_transmit(self.start_y_pacific, self.start_x_pacific)
        #     # raise Exception("Error: Non-sea coordinates (start coordinates", self.start_x_pacific, self.start_y_pacific, ")")
        # if self.pacific_map[self.end_y_pacific][self.end_x_pacific] < 220:
        #     self.end_y_pacific, self.end_x_pacific = \
        #         self.coordinate_transmit(self.end_y_pacific, self.end_x_pacific)
        # if self.atlantic_map[self.start_y_atlantic][self.start_x_atlantic] < 220:
        #     self.start_y_atlantic, self.start_x_atlantic = \
        #         self.coordinate_transmit(self.start_y_atlantic, self.start_x_atlantic)
        #     # raise Exception("Error: Non-sea coordinates (start coordinates", self.start_x_atlantic, self.start_y_atlantic, ")")
        # if self.atlantic_map[self.end_y_atlantic][self.end_x_atlantic] < 220:
        #     self.end_y_atlantic, self.end_x_atlantic = \
        #         self.coordinate_transmit(self.end_y_atlantic, self.end_x_atlantic)
        print("└. complete: load world map\n")

    def coordinate_transmit(self, cord_y, cord_x):

        print("first", cord_y, cord_x)
        minimum = 100000
        for i in range(cord_y - 5, cord_y + 6):
            for j in range(cord_x - 10, cord_x + 11):
                if i < 0 or i > 474 or j < 0 or j > 1013:
                    continue
                if 220 <= self.atlantic_map[i][j] <= 255:
                    dist = abs(i - cord_y) + abs(j - cord_x)
                    if minimum > dist:
                        minimum = dist
                        new_y, new_x = i, j

        print("second", new_y, new_x)
        return new_y, new_x

    def latilong_2_xy(self, equator, meridian, height, width, latitude, longitude):
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
    def xy_2_latilong(self, equator, meridian, height, width, x, y):
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

    # map 상의 두 지점을 대상으로 haversine distance를 반환해주는 함수
    def xy_distance_haversine(self, start, end, paci_or_atlan):
        start_latitude, start_longitude, end_latitude, end_longitude = 0, 0, 0, 0

        if paci_or_atlan == "pacific":
            start_latitude, start_longitude = self.xy_2_latilong(self.equator, self.pacific_meridian, self.height, self.width,
                                                                 start[0], start[1])
            end_latitude, end_longitude = self.xy_2_latilong(self.equator, self.pacific_meridian, self.height, self.width, end[0],
                                                             end[1])
        if paci_or_atlan == "atlantic":
            start_latitude, start_longitude = self.xy_2_latilong(self.equator, self.atlantic_meridian, self.height,
                                                                 self.width,
                                                                 start[0], start[1])
            end_latitude, end_longitude = self.xy_2_latilong(self.equator, self.atlantic_meridian, self.height,
                                                             self.width, end[0],
                                                             end[1])
        result = haversine((start_latitude, start_longitude), (end_latitude, end_longitude), unit='km')
        return result

class Astar:
    def __init__(self, worldmaps, toggle_haversine):
        self.worldmaps = worldmaps # Wolrdmap 객체 (instance)
        self.toggle_haversine = toggle_haversine # haversine을 사용할지 결정하는 toggle (Boolean)
        self.phaselist = ["pacific", "atlantic"]
        self.step_size = 1  # 탐색 길이
        self.add = ([0, self.step_size], [0, -self.step_size], [self.step_size, 0], [-self.step_size, 0],
                    [self.step_size, self.step_size], [-self.step_size, -self.step_size],
                    [-self.step_size, self.step_size], [self.step_size, -self.step_size])

        self.pacific_star = {'position': (self.worldmaps.start_x_pacific, self.worldmaps.start_y_pacific), 'cost': 0, 'parent': None,
                             'trajectory': 0}
        self.pacific_end = {'position': (self.worldmaps.end_x_pacific, self.worldmaps.end_y_pacific), 'cost': 0,
                            'parent': (self.worldmaps.end_x_pacific, self.worldmaps.end_y_pacific), 'trajectory': 0}

        self.atlantic_star = {'position': (self.worldmaps.start_x_atlantic, self.worldmaps.start_y_atlantic), 'cost': 0, 'parent': None,
                             'trajectory': 0}
        self.atlantic_end = {'position': (self.worldmaps.end_x_atlantic, self.worldmaps.end_y_atlantic), 'cost': 0,
                            'parent': (self.worldmaps.end_x_atlantic, self.worldmaps.end_y_atlantic), 'trajectory': 0}

        self.pacific_sorted_pf = [] # A* 탐색을 위한 open set (list)
        self.pacific_opendict = defaultdict(lambda : {'cost' : np.inf}) # A* 탐색을 위한 open set (list)
        self.pacific_closedict = {
            (self.worldmaps.start_x_pacific, self.worldmaps.start_y_pacific) : {'cost' : 0, 'parent': None, 'trajectory': 0}} # A* 탐색을 위한 close set (list)
        self.pacific_route_length = 0

        self.atlantic_sorted_pf = []
        self.atlantic_opendict = defaultdict(lambda: {'cost': np.inf})
        self.atlantic_closedict = {
            (self.worldmaps.start_x_atlantic, self.worldmaps.start_y_atlantic): {'cost': 0, 'parent': None, 'trajectory': 0}}
        self.atlantic_route_length = 0

        self.pacific_road = [] # map상의 좌표로 최종 경로가 담기는 변수 (map상의 좌표 list)
        self.pacific_real_road = [] # latitude, longitude로 최종 경로가 담기는 변수 (latitude, longitude list)
        self.pacific_point = [] # temp
        self.pacific_cal_time = 0 # 탐색에 소요된 시간을 저장하는 변수

        self.atlantic_road = []
        self.atlantic_real_road = []
        self.atlantic_point = []
        self.atlantic_cal_time = 0

    # A* 알고리즘을 이용하여 해역 상의 최적 경로를 탐색
    def get_route(self):
        self.start_timer = datetime.now().replace(microsecond=0)
        for phase in tqdm(self.phaselist):
            if phase == "pacific":
                while 1:
                    s_point = list(self.pacific_closedict)[-1]
                    for i in range(len(self.add)):
                        x = s_point[0] + self.add[i][0]
                        if x < 0 or x >= self.worldmaps.width:
                            continue
                        y = s_point[1] + self.add[i][1]
                        if y < 0 or y >= self.worldmaps.height:
                            continue
                        if (x, y) in self.pacific_closedict.keys():
                            continue
                        if self.worldmaps.pacific_map[y, x] < 220: # 지도상 이동 가능한 바다인지 판별
                            continue

                        G = self.pacific_closedict[s_point]["trajectory"] + self.worldmaps.xy_distance_haversine((x, y), s_point, phase) # trajectory-based + haversine distacne G
                        H = self.worldmaps.xy_distance_haversine(self.pacific_end['position'], (x, y), phase) # haversine distance H
                        F = G + H

                        if self.pacific_opendict[(x,y)]['cost'] > F :
                            self.pacific_opendict[(x,y)] = {'cost' : F, 'parent': s_point, 'trajectory': G}

                    # 비교 탐색 향후 개선 필
                    min_val = {'cost' : np.inf}
                    min_pos = None
                    for pos, val in self.pacific_opendict.items():
                        if val['cost'] < min_val['cost']:  # Search minimal cost in openlist
                            min_val = val
                            min_pos = pos
                    self.pacific_closedict[min_pos] = min_val
                    self.pacific_opendict.pop(min_pos)

                    if min_pos == self.pacific_end['position']:
                        break

                pos = min_pos
                self.pacific_road.append(pos)
                self.pacific_real_road.append(self.worldmaps.xy_2_latilong(self.worldmaps.equator,
                                                                           self.worldmaps.pacific_meridian,
                                                                           self.worldmaps.height,
                                                                           self.worldmaps.width,
                                                                           pos[0], pos[1]))

                while self.pacific_closedict[pos]['parent'] != None :
                    pos = self.pacific_closedict[pos]['parent']
                    self.pacific_route_length += self.pacific_closedict[pos]['trajectory']
                    self.pacific_road.append(pos)
                    self.pacific_real_road.append(self.worldmaps.xy_2_latilong(self.worldmaps.equator,
                                                                               self.worldmaps.pacific_meridian,
                                                                               self.worldmaps.height,
                                                                               self.worldmaps.width,
                                                                               pos[0], pos[1]))

                print("└. complete : route on pacific centered map")

            if phase == "atlantic":
                while 1:
                    s_point = list(self.atlantic_closedict)[-1]
                    for i in range(len(self.add)):
                        x = s_point[0] + self.add[i][0]
                        if x < 0 or x >= self.worldmaps.width:
                            continue
                        y = s_point[1] + self.add[i][1]
                        if y < 0 or y >= self.worldmaps.height:
                            continue
                        if (x, y) in self.atlantic_closedict.keys():
                            continue
                        if self.worldmaps.atlantic_map[y, x] < 220:  # 지도상 이동 가능한 바다인지 판별
                            continue

                        G = self.atlantic_closedict[s_point]["trajectory"] + self.worldmaps.xy_distance_haversine((x, y), s_point, phase)  # trajectory-based + haversine distacne G
                        H = self.worldmaps.xy_distance_haversine(self.atlantic_end['position'], (x, y), phase)  # haversine distance H
                        F = G + H

                        if self.atlantic_opendict[(x, y)]['cost'] > F:
                            self.atlantic_opendict[(x, y)] = {'cost': F, 'parent': s_point, 'trajectory': G}

                    # 비교 탐색 향후 개선 필
                    min_val = {'cost': np.inf}
                    min_pos = None
                    for pos, val in self.atlantic_opendict.items():
                        if val['cost'] < min_val['cost']:  # Search minimal cost in openlist
                            min_val = val
                            min_pos = pos
                    self.atlantic_closedict[min_pos] = min_val
                    self.atlantic_opendict.pop(min_pos)

                    if min_pos == self.atlantic_end['position']:
                        break

                pos = min_pos
                self.atlantic_road.append(pos)
                self.atlantic_real_road.append(self.worldmaps.xy_2_latilong(self.worldmaps.equator,
                                                                            self.worldmaps.atlantic_meridian,
                                                                            self.worldmaps.height,
                                                                            self.worldmaps.width,
                                                                            pos[0], pos[1]))

                while self.atlantic_closedict[pos]['parent'] != None:
                    pos = self.atlantic_closedict[pos]['parent']
                    self.atlantic_route_length += self.atlantic_closedict[pos]['trajectory']
                    self.atlantic_road.append(pos)
                    self.atlantic_real_road.append(
                        self.worldmaps.xy_2_latilong(self.worldmaps.equator,
                                                     self.worldmaps.atlantic_meridian,
                                                     self.worldmaps.height,
                                                     self.worldmaps.width,
                                                     pos[0], pos[1]))

                print("└. complete : route on atlantic centered map")

    def select_route(self):
        if self.pacific_route_length <= self.atlantic_route_length :
            self.road = self.pacific_road
            self.real_road = self.pacific_real_road
            self.route_on_map = self.worldmaps.route_on_pacific_map
            self.start = self.pacific_star
            self.end = self.pacific_end
            self.maptype = 'pacific'
        else:
            self.road = self.atlantic_road
            self.real_road = self.atlantic_real_road
            self.route_on_map = self.worldmaps.route_on_atlantic_map
            self.start = self.atlantic_star
            self.end = self.atlantic_end
            self.maptype = 'atlantic'

        self.end_timer = datetime.now()
        self.cal_time = self.end_timer - self.start_timer
        print("└. complete: select route on", self.maptype, "centered map (calculation time =", self.cal_time, ")\n")


    def draw_route(self):
        for i in range(len(self.road)-1):
            cv2.line(self.route_on_map, self.road[i], self.road[i+1], (0, 0, 200), 1)

        now = datetime.now()
        now = now.strftime("%m%d%H%M%S")
        self.real_road = self.real_road[::-1]

        cv2.circle(self.route_on_map, self.start['position'], 2, (0, 255, 0), -1)
        cv2.circle(self.route_on_map, self.end['position'], 2, (255, 255, 0), -1)
        io.imshow(self.route_on_map)
        saving_name = "routemap_" + now + ".png"
        cv2.imwrite(saving_name, self.route_on_map)
        print("└. complete: draw route on map and save -> '" + saving_name + "'")
        with open("./routelist_"+now+".txt", 'w')as lf:
            lf.write(str(self.worldmaps.origin[0]) + ", " + str(self.worldmaps.origin[1]) + " to " + str(self.worldmaps.destination[0]) + ", " + str(self.worldmaps.destination[1])+"\n")
            lf.write(str(self.cal_time) + "\n")
            for line in self.real_road:
                lf.write(str(line[0])+", "+str(line[1])+"\n")
        print("└. complete: save route list -> './routelist_"+now+".txt'\n")

    def draw_map(self, clr, lineClr):

        seaMap = folium.Map(location=[35, 123], zoom_start=5)
        if self.maptype:
            pre_path = [
                (self.real_road[i][0], self.real_road[i][1]) if 0 <= self.real_road[i][1] <= 180 \
                    else (self.real_road[i][0], self.real_road[i][1]+360) for i in range(len(self.real_road))
                ]
        else:

            pre_path = [
                (self.real_road[i][0], self.real_road[i][1]) if 0 <= self.real_road[i][1] <= 180 \
                    else (self.real_road[i][0], self.real_road[i][1]) for i in range(len(self.real_road))
            ]

        for i in range(len(pre_path)):
            path = list(pre_path[i])
            if i == 0:
                folium.Marker(location=path,
                                zoom_start=17,
                                icon=folium.Icon(color=clr, icon='star'),
                                popup="lat:{0:0.2f} || lon:{1:0.2f}".format(path[0], path[1])
                                 ).add_to(seaMap)
            else:
                folium.CircleMarker(location=path,
                                    zoom_start=17,
                                    color=clr,
                                    fill_color=clr,
                                    popup="lat:{0:0.2f} || lon:{1:0.2f}".format(path[0], path[1]),
                                    radius=5,
                                    ).add_to(seaMap)

        folium.PolyLine(locations=pre_path, color=lineClr, tooltip='PolyLine').add_to(seaMap)
        seaMap.save('Astar*loadMap.html')
        print("LoadMap Completion!")

if __name__ == "__main__":

    # [37, 125.5]  # 대한민국 인천
    # [37, -123]  # 미국 샌프란시스코
    # [5.5, 80]  # 스리랑카 콜롬보
    # [40, 14.15]  # 이탈리아 나폴리
    # [25, -80]  # 미국 마이애미
    # [-33, -57]  # 아르헨티나 부에노스 아이레스

    # 메르카토르 법으로 근사된 세계 지도의 픽셀을 node로 보고, 두 픽셀 사이의 최단 경로를 Astar 알고리즘으로 탐색합니다.

    # origin의 위경도는 start에 destination의 위경도는 end에 입력해주세요
    # origin과 destination은 모두 해역 위에 있어야 합니다.
        # 입력 가능한 위도 범위 : -80 ~ 84
        # 입력 가능한 경도 범위 : -180 ~ 180
        # 북위와 동경은 양수, 남위와 서경은 음수로 입력 (위 참조)

    # map은 1014*475 스케일로 근사되었기 때문에 실제 위경도와 map 상의 좌표가 2% 가량 차이날 수 있습니다.
    # 반환은 두 가지로 route list (txt) 와 map 상에 표기된 route map (png)입니다.
    # 구현 테스트 결과 (계산 시간)
        # 대한민국_인천 -> 미국 샌프란시스코 : 00:04:31.16
        # 대한민국_인천 -> 스리랑카 콜롬보 : 00:00:00.37
        # 대한민국 인천 -> 이탈리아 나폴리 : 00:00:23.99
        # 대한민국_인천 -> 미국 마이애미 : 00:00:35.78
        # 대한민국_인천 -> 아르헨티나 부에노스 아이레스 : 00:03:04.07

    start = [41.08651, 149.43457]
    end = [-2.24, -44.11]

    print("1. start: load map")
    worldmaps = Worldmaps(start[0], start[1], end[0], end[1])
    astar = Astar(worldmaps, True)

    print("2. start: path finding")
    astar.get_route()
    astar.select_route()
    clr = 'red'
    lineClr = 'orange'
    astar.draw_map(clr, lineClr)
    exit()

    print("3. start: draw route on map")
    astar.draw_route()