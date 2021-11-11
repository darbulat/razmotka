import json
import os

import fiona
from shapely.geometry import MultiPoint, Polygon, Point, shape


class SaveResults:
    def __init__(self, n, m):
        self.N = n
        self.M = m

        self.matrix = [[
            0 if ((x > 39 and 18 < y < 49)
                  or (x > 26 and 49 <= y < 61)) else 1
            for x in range(42)
        ] for y in range(61)]
        self.matrix_pp = [[0 for _ in range(42 + 2 * self.N)] for _ in
                          range(61 + 2 * self.M)]

        self.s_points, self.matrix_points = self.parse_mesa(
            'algorithm/mesa/s',
            len(self.matrix[0]) + 1,
            len(self.matrix) + 1,
        )
        self.r_points, self.matched_matrix = self.parse_mesa(
            'algorithm/mesa/r',
            len(self.matrix_pp[0]) + 1,
            len(self.matrix_pp) + 1,
        )

    def match_coords(self, width, height, min_x, min_y, max_x, max_y):
        ans = [[0 for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                ans[y][x] = [
                    min_x + x * (max_x - min_x) / width,
                    min_y + y * (max_y - min_y) / height
                ]
        return ans

    def get_winding_from_polygons(self, polygons: list,
                                  folder: str,
                                  postfix: str):
        for i in range(len(polygons)):
            points = [Point(self.matched_matrix[xy[1]][xy[0]]) for xy in polygons[i]]
            if not points:
                continue
            m_point = MultiPoint(points)
            with open(f'{folder}/points_{i}_{postfix}.geojson', 'w') as f:
                f.write(json.dumps(m_point.__geo_interface__))

    def is_bound(self, matr: list, x: int, y: int) -> bool:
        if not isinstance(matr[y][x], list):
            return False
        try:
            return (matr[y][x][1] != matr[y][x + 1][1]
                    or matr[y][x][1] != matr[y][x - 1][1]
                    or matr[y][x][1] != matr[y + 1][x][1]
                    or matr[y][x][1] != matr[y - 1][x][1]
                    or matr[y][x][1] != matr[y - 1][x - 1][1]
                    or matr[y][x][1] != matr[y + 1][x + 1][1]
                    or matr[y][x][1] != matr[y - 1][x + 1][1]
                    or matr[y][x][1] != matr[y + 1][x - 1][1])
        except (IndexError, TypeError):
            return True

    def get_polygons_for_winding(self, matrix_reception: list, size: int):
        polygons = [[] for _ in range(size)]
        for y in range(len(matrix_reception)):
            for x in range(len(matrix_reception[0])):
                if self.is_bound(matrix_reception, x, y):
                    polygons[matrix_reception[y][x][1]].append((x, y))
        return polygons

    def get_winding_reception_points(self,
                                     matrix_reception: list,
                                     size: int, folder: str):
        polygons = self.get_polygons_for_winding(matrix_reception, size)
        self.get_winding_from_polygons(
            polygons,
            folder,
            'reception_winding',
        )

    def get_unwinding_reception_points(self,
                                       matrix_reception: list,
                                       size: int, folder: str):
        polygons = self.get_polygons_for_winding(matrix_reception, size)
        self.get_winding_from_polygons(
            polygons,
            folder,
            'reception_unwinding',
        )

    def get_excitation_points(self, matrix_point: list,
                              multi_point: MultiPoint,
                              min_x: list, min_y: list,
                              max_x: list, max_y: list,
                              folder: str,
                              postfix: str):
        for i in range(len(min_x)):
            polygon = Polygon([
                matrix_point[min_y[i]][min_x[i]],
                matrix_point[max_y[i]][min_x[i]],
                matrix_point[max_y[i]][max_x[i]],
                matrix_point[min_y[i]][max_x[i]],
            ])
            points = multi_point.intersection(polygon)
            with open(f'{folder}/points_{i}_{postfix}.geojson', 'w') as f:
                f.write(json.dumps(points.__geo_interface__))
        return

    def parse_mesa(self, geom_file, width, height):
        with fiona.open(geom_file) as fiona_rp:
            points = MultiPoint(
                [Point(shape(rp['geometry'])) for rp in fiona_rp])
        min_x_coord = points.bounds[0]
        min_y_coord = points.bounds[3]
        max_x_coord = points.bounds[2]
        max_y_coord = points.bounds[1]
        matrix_p = self.match_coords(width, height,
                                     min_x_coord, min_y_coord,
                                     max_x_coord, max_y_coord)
        return points, matrix_p

    def save_answer(self, answer, folder):
        os.makedirs(folder)
        self.get_excitation_points(
            self.matrix_points,
            self.s_points,
            answer.min_x, answer.min_y,
            answer.max_x, answer.max_y,
            folder,
            'excitation',
        )
        self.get_winding_reception_points(
            answer.matrix,
            len(answer.min_x), folder
        )
        self.get_unwinding_reception_points(
            answer.matrix_unwinding,
            len(answer.min_x), folder,
        )
