import json
import os
from collections import OrderedDict
from itertools import product

import fiona
from shapely.geometry import MultiPoint, Polygon, Point, shape


class SaveResults:
    def __init__(self, mesa_folder, points_per_section=6):
        self.points_per_section = points_per_section

        self.s_points, self.matrix_points = self._get_strict_multipoints(
            os.path.join(mesa_folder, 's'),
        )
        # self.r_points, self.matched_matrix = self.parse_mesa(
        #     os.path.join(mesa_folder, 'r'),
        #     len(self.matrix_pp[0]) + 1,
        #     len(self.matrix_pp) + 1,
        # )

    def match_coords(self, width, height, min_x, min_y, max_x, max_y):
        ans = [[0 for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                ans[y][x] = [
                    min_x + x * (max_x - min_x) / (width - 1),
                    min_y + y * (max_y - min_y) / (height - 1)
                ]
        return ans

    def get_winding_from_polygons(self, polygons: list,
                                  folder: str,
                                  postfix: str):
        for i in range(len(polygons)):
            points = [Point(self.matched_matrix[xy[1]][xy[0]]) for xy in
                      polygons[i]]
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

    def get_excitation_points(self,
                              min_x: list, min_y: list,
                              max_x: list, max_y: list,
                              folder: str,
                              postfix: str):
        excitation_schema = {
            'geometry': 'MultiPoint',
            'properties': OrderedDict([
                ('day', 'int'),
                ('count', 'int')
            ])
        }
        file = os.path.join(folder, postfix + 'geojson')
        with fiona.open(
                file,
                'w',
                driver='GeoJSON',
                schema=excitation_schema,
        ) as fi:
            for i in range(len(min_x)):
                polygon = Polygon([
                    self.matrix_points[min_y[i]][min_x[i]],
                    self.matrix_points[max_y[i]][min_x[i]],
                    self.matrix_points[max_y[i]][max_x[i]],
                    self.matrix_points[min_y[i]][max_x[i]],
                ])
                points = self.s_points.intersection(polygon)
                excitation_points = {
                    'geometry': points.__geo_interface__,
                    'properties': OrderedDict([
                        ('day', i),
                        ('count', len(points))
                    ])
                }
                fi.write(excitation_points)
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
        os.makedirs(folder, exist_ok=True)
        self.get_excitation_points(
            answer.min_x, answer.min_y,
            answer.max_x, answer.max_y,
            folder,
            'excitation',
        )
        # self.get_winding_reception_points(
        #     answer.matrix,
        #     len(answer.min_x), folder
        # )
        # self.get_unwinding_reception_points(
        #     answer.matrix_unwinding,
        #     len(answer.min_x), folder,
        # )

    def _get_strict_multipoints(self, geom_file):
        max_point_numb = 0
        min_point_numb = float('inf')
        max_line_name = 0
        min_line_name = float('inf')
        with fiona.open(geom_file) as fiona_rp:
            points = MultiPoint(
                [Point(shape(rp['geometry'])) for rp in fiona_rp])
            for rp in fiona_rp:
                line_name = rp['properties']['Line name']
                point_numb = rp['properties']['Point numb']
                if line_name > max_line_name:
                    max_line_name = line_name
                if line_name < min_line_name:
                    min_line_name = line_name
                if point_numb > max_point_numb:
                    max_point_numb = point_numb
                if point_numb < min_point_numb:
                    min_point_numb = point_numb

        line_count = int(max_line_name - min_line_name)
        point_numb_count = int(max_point_numb - min_point_numb) + 1
        wight = int(line_count / self.points_per_section + 1)
        height = int(point_numb_count / self.points_per_section)
        print(f'{line_count=}')
        print(f'{point_numb_count=}')

        min_x_coord = points.bounds[0]
        min_y_coord = points.bounds[3]
        max_x_coord = points.bounds[2]
        max_y_coord = points.bounds[1]
        delta_x = (max_x_coord - min_x_coord) / line_count / 2
        delta_y = (max_y_coord - min_y_coord) / point_numb_count / 2
        points = self.match_coords(wight, point_numb_count, min_x_coord,
                                   min_y_coord, max_x_coord, max_y_coord)
        matched_matrix = self.match_coords(
            wight + 1, height + 1, min_x_coord - delta_x,
            min_y_coord - delta_y, max_x_coord, max_y_coord
        )
        multi_points = MultiPoint(
            [points[y][x] for y, x
             in product(range(point_numb_count), range(wight))]
        )
        return multi_points, matched_matrix


from dataclasses import dataclass


@dataclass
class Answer:
    min_x: list
    min_y: list
    max_x: list
    max_y: list
    matrix: list
    matrix_unwinding: list
    max_area: int
    dispersion: int


if __name__ == '__main__':
    ans = {
        "max_x": [42, 20, 20, 31, 42, 42, 31, 20, 10, 14, 14, 25, 40, 40, 25,
                  14, 14, 25, 40, 40, 40, 25, 11, 25, 11, 25],
        "max_y": [4, 5, 10, 11, 11, 19, 19, 19, 19, 25, 31, 28, 25, 31, 36, 37,
                  43, 43, 37, 43, 49, 49, 52, 55, 61, 61],
        "min_x": [20, 0, 0, 20, 31, 31, 20, 10, 0, 0, 0, 14, 25, 25, 14, 0, 0,
                  14, 25, 25, 25, 11, 0, 11, 0, 11],
        "min_y": [0, 0, 5, 4, 4, 11, 11, 10, 10, 19, 25, 19, 19, 25, 28, 31, 37,
                  36, 31, 37, 43, 43, 43, 49, 52, 55]}

    answer = Answer(min_x=ans['min_x'], min_y=ans['min_y'], max_x=ans['max_x'],
                    max_y=ans['max_y'], matrix=[], matrix_unwinding=[],
                    max_area=0, dispersion=0)

    save_results = SaveResults(mesa_folder='mesa/')
    save_results.save_answer(answer, folder='1')
