import os

import cv2
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
from tqdm import tqdm
from multiprocessing import Pool

def put_cost_reverse_cost(x1, y1, x2, y2,  df_edges, idx, vertex_lsts, cost, reverse_cost):

    if 0 <= x2 < 58 or 408 <= x2 < 475:

        if cost > 0 and reverse_cost > 0:

            df_edges.loc[idx] = [idx, '', 0, 0, 1*cost, 1*reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

        elif cost>0 and reverse_cost<0:

            df_edges.loc[idx] = [idx, '', 0, 0, 1 * cost, reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

        elif cost < 0 and reverse_cost > 0:

            df_edges.loc[idx] = [idx, '', 0, 0, cost, 1*reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])



    elif 58 <= x2 < 116 or 350 <= x2 < 408:

        if cost > 0 and reverse_cost > 0:

            df_edges.loc[idx] = [idx, '', 0, 0, 1 * cost, 1 * reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

        elif cost > 0 and reverse_cost < 0:

            df_edges.loc[idx] = [idx, '', 0, 0, 1 * cost, reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

        elif cost < 0 and reverse_cost > 0:

            df_edges.loc[idx] = [idx, '', 0, 0, cost, 1 * reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

    elif 116 <= x2 < 175 or 292 <= x2 < 350:

        if cost > 0 and reverse_cost > 0:

            df_edges.loc[idx] = [idx, '', 0, 0, 3 * cost, 3 * reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

        elif cost > 0 and reverse_cost < 0:

            df_edges.loc[idx] = [idx, '', 0, 0, 3 * cost, reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

        elif cost < 0 and reverse_cost > 0:

            df_edges.loc[idx] = [idx, '', 0, 0, cost, 3 * reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

    elif 175 <= x2 < 292:

        if cost > 0 and reverse_cost > 0:

            df_edges.loc[idx] = [idx, '', 0, 0, 4 * cost, 4 * reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

        elif cost > 0 and reverse_cost < 0:

            df_edges.loc[idx] = [idx, '', 0, 0, 4 * cost, reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

        elif cost < 0 and reverse_cost > 0:

            df_edges.loc[idx] = [idx, '', 0, 0, cost, 4 * reverse_cost, x1, y1, x2, y2]

            if [x1, y1] not in vertex_lsts:
                vertex_lsts.append([x1, y1])

            if [x2, y2] not in vertex_lsts:
                vertex_lsts.append([x2, y2])

    return vertex_lsts, df_edges



def make_edges_vector(arrays):


        df_edges = pd.DataFrame(columns=['id', 'dir', 'source', 'target', 'cost','reverse_cost','x1','y1','x2','y2'])
        df_vertex = pd.DataFrame(columns=['id', 'x', 'y'])
        idx = 0
        vertex_lsts = []
        pixel_color = 50

        for x in tqdm(range(arrays[0].shape[0])):

            for y in range(arrays[0].shape[1]):

                # 첫번째 행 로드는 위쪽 로드 확인하지 않고 오른쪽 로드만 확인하고 끝에서 두번째 로드에서 끝남
                if x == 0 and y<=arrays[0].shape[1]-2:

                    # 바다
                    if arrays[0][x,y] > pixel_color:

                        # 오른쪽 로드 확인
                        if arrays[0][x, y+1] > pixel_color:

                            cost = 1
                            reverse_cost = 1
                            x1 = x
                            y1 = y
                            x2 = x
                            y2 = y+1
                            vertex_lsts, df_edges = put_cost_reverse_cost(
                                x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                            idx += 1

                        else:

                            cost = 1
                            reverse_cost = -1
                            x1 = x
                            y1 = y
                            x2 = x
                            y2 = y + 1
                            vertex_lsts, df_edges = put_cost_reverse_cost(
                                x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                            idx += 1
                    # 육지
                    else:
                        # 바다
                        if arrays[0][x, y + 1] > 0:

                            cost = -1
                            reverse_cost = 1
                            x1 = x
                            y1 = y
                            x2 = x
                            y2 = y + 1
                            vertex_lsts, df_edges = put_cost_reverse_cost(
                                x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                            idx += 1

                # 두번째 행 로드부터는 오른쪽 위쪽 다 확인 마지막 열은 위쪽만 확인
                elif x > 0:

                    # 바다
                    if arrays[0][x, y] > pixel_color:

                        # 마지막 열인지 확인
                        if y == arrays[0].shape[1] - 1:

                            # 위쪽 확인 바다
                            if arrays[0][x - 1, y] > pixel_color:
                                cost = 1
                                reverse_cost = 1
                                x1=x
                                y1=y
                                x2=x-1
                                y2=y
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1
                            # 위쪽 확인 육지
                            else:

                                cost = 1
                                reverse_cost = -1
                                x1 = x
                                y1 = y
                                x2 = x-1
                                y2 = y
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1

                        # 마지막 열이 아님
                        else:
                            # 오른쪽 확인 바다
                            if arrays[0][x, y + 1] > pixel_color:

                                cost = 1
                                reverse_cost = 1
                                x1 = x
                                y1 = y
                                x2 = x
                                y2 = y + 1
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1

                            # 오른쪽 확인 육지
                            else:

                                cost = 1
                                reverse_cost = -1
                                x1 = x
                                y1 = y
                                x2 = x
                                y2 = y + 1
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1

                            # 위쪽 확인 바다
                            if arrays[0][x-1, y] > pixel_color:

                                cost = 1
                                reverse_cost = 1
                                x1 = x
                                y1 = y
                                x2 = x-1
                                y2 = y
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1

                            # 위쪽 확인 육지
                            else:

                                cost = 1
                                reverse_cost = -1
                                x1 = x
                                y1 = y
                                x2 = x-1
                                y2 = y
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1

                    else:
                        # 마지막 열인지 확인 위쪽 로드만 확인
                        if y == arrays[0].shape[1] - 1:
                            # 위쪽 로드만 확인
                            if arrays[0][x - 1, y] > pixel_color:

                                cost = -1
                                reverse_cost = 1
                                x1 = x
                                y1 = y
                                x2 = x-1
                                y2 = y
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1

                        # 마지막 행의 열로드가 아니면
                        else:
                            # 오른쪽 로드 확인 바다
                            if arrays[0][x, y+1] > pixel_color:

                                cost = -1
                                reverse_cost = 1
                                x1 = x
                                y1 = y
                                x2 = x
                                y2 = y + 1
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1

                            # 위쪽 로드 확인 바다
                            if arrays[0][x-1, y] > pixel_color:

                                cost = -1
                                reverse_cost = 1
                                x1 = x
                                y1 = y
                                x2 = x-1
                                y2 = y
                                vertex_lsts, df_edges = put_cost_reverse_cost(
                                    x1, y1, x2, y2, df_edges, idx, vertex_lsts, cost, reverse_cost)
                                idx += 1


        for i in range(len(vertex_lsts)):
            vertex_lsts[i].insert(0, i+1)
            df_vertex.loc[i] = vertex_lsts[i]


        df_vertex.to_csv(f'csv/{arrays[1]}_vertex.csv', index=False)
        df_edges.to_csv(f'csv/{arrays[1]}_edges.csv', index=False)



def insert_data_into_pg(df, edge_table):

    url = 'postgresql+psycopg2://ecomarine:Ecomarine1!@3.35.245.80:27720/ecomarine'
    engine = sqlalchemy.create_engine(url)
    for i in tqdm(range(df.shape[0])):

        sql = f"""INSERT INTO {edge_table} (cost,reverse_cost,x1,y1,x2,y2) 
                VALUES ( {df.loc[i,'cost']}, {df.loc[i,'reverse_cost']}, {df.loc[i,'x1']},
                {df.loc[i,'y1']}, {df.loc[i,'x2']}, {df.loc[i,'y2']});"""

        with engine.connect().execution_options(autocommit=True) as conn:
            conn.execute(text(sql))

if __name__ == "__main__":

    p_map = cv2.imread("worldMap/mercato_pacific.png", cv2.IMREAD_GRAYSCALE)
    a_map = cv2.imread("worldMap/mercato_atlantic.png", cv2.IMREAD_GRAYSCALE)
    df = pd.DataFrame(a_map)

    lsts = [(p_map, 'pacific'), (a_map, 'atlantic')]
    cnt = os.cpu_count()
    pool = Pool(cnt)
    pool.map(make_edges_vector, lsts)
    pool.close()
    pool.join()

