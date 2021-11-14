import os.path
from typing import List

import fiona
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint, Point, shape


class MesaExporter:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.crs = None

    def get_header_lines(self, content: list) -> int:
        i = 0
        for line in content:
            if line[0] != 'H':
                return i
            i += 1

    def get_zone(self, content: List[str]):
        for line in content:
            if line.lower().find('zone') > 0:
                zone = line.lower().split('zone')[2]
                return zone[:3].strip()
        raise ValueError('Wrong mesa file')

    def get_indexes(self, content: List[str]):
        if content[0].startswith('S'):
            return 1, 2, -4, -3
        if content[0].startswith('R'):
            return 1, 2, -3, -2
        raise ValueError('Wrong mesa file')

    def get_shapefile_from_mesa(self, content, mesa_type='S'):
        nheaderlines = self.get_header_lines(content)
        header = content[:nheaderlines]
        content = content[nheaderlines:]
        data = [str(row).split() for row in content]
        zone = self.get_zone(header)
        self.crs = '284' + zone.strip()
        line_name, point_numb, lon, lat = self.get_indexes(content)
        # Extract x and y coordinates
        l = []
        for e in data:
            l.append([e[line_name], e[point_numb], zone + e[lon], e[lat]])

        df = pd.DataFrame(l, columns=['Line_name', 'Point_numb', 'Lon', 'Lat'])
        df = df.astype(float)
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.Lon, df.Lat))
        file_path = os.path.join(
            self.folder_path, mesa_type.lower(), mesa_type + '.shp')
        gdf.to_file(file_path)
        return file_path

    @classmethod
    def count_all_points(cls, path):
        with fiona.open(path) as fi:
            return len(fi)

    def get_mask_from_shp(self, path) -> List[list]:
        df = gpd.read_file(path)
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.Lon, df.Lat))
        xmin, ymin, xmax, ymax = gdf.total_bounds
        width, height, points = self._get_width_height(path)

        matrix = [[0 for _ in range(width + 1)] for _ in range(height + 1)]
        x_cell_size = (xmax - xmin) / width
        y_cell_size = (ymax - ymin) / height

        x = 0
        for x0 in np.arange(xmin, xmax + x_cell_size, x_cell_size):
            y = -1
            for y0 in np.arange(ymin, ymax + y_cell_size, y_cell_size):
                # bounds
                xb0 = x0 - x_cell_size / 2
                xb1 = x0 + x_cell_size / 2
                yb0 = y0 - y_cell_size / 10
                yb1 = y0 + y_cell_size - y_cell_size / 10
                cell = shapely.geometry.box(xb0, yb0, xb1, yb1)
                if cell.intersects(points):
                    matrix[y][x] = 1
                y -= 1
            x += 1

        return matrix

    def _get_width_height(self, geom_file,
                          y_name='Line name',
                          x_name='Point numb'):
        max_point_numb = 0
        min_point_numb = float('inf')
        max_line_name = 0
        min_line_name = float('inf')
        points_per_section = self.count_points_per_section()
        with fiona.open(geom_file) as fiona_rp:
            points = MultiPoint(
                [Point(shape(rp['geometry'])) for rp in fiona_rp])
            for rp in fiona_rp:
                line_name = rp['properties'][y_name]
                point_numb = rp['properties'][x_name]
                if line_name > max_line_name:
                    max_line_name = line_name
                if line_name < min_line_name:
                    min_line_name = line_name
                if point_numb > max_point_numb:
                    max_point_numb = point_numb
                if point_numb < min_point_numb:
                    min_point_numb = point_numb

        line_count = int(max_line_name - min_line_name)
        point_numb_count = int(max_point_numb - min_point_numb)
        point_numb_count += 1
        width = (line_count / points_per_section)
        height = (point_numb_count / points_per_section)
        width = int(width)
        height = int(height) - 1
        return width, height, points

    def count_points_per_section(self):
        return 6


if __name__ == '__main__':
    diretory_path = './mesa/SPS_FINAL_raw/'
    r_filename = 'R.rps'
    s_filename = 'S.sps'
    with open(os.path.join(diretory_path, s_filename), 'r') as f:
        s_content = f.read().split('\n')
    with open(os.path.join(diretory_path, r_filename), 'r') as f:
        r_content = f.read().split('\n')
    mesa_exporter = MesaExporter('mesa')
    excitation_shp = mesa_exporter.get_shapefile_from_mesa(s_content, 'S')
    reception_shp = mesa_exporter.get_shapefile_from_mesa(r_content, 'R')
    reception_count = MesaExporter.count_all_points(excitation_shp)
    excitation_count = MesaExporter.count_all_points(reception_shp)
    mask = mesa_exporter.get_mask_from_shp(excitation_shp)
