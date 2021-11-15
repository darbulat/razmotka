import os.path

import requests
from shapely.geometry import Point

from algorithm.parse_mesa import MesaExporter
from algorithm.save_results import SaveResults
from algorithm.tsp import Razmotka

from flask import Flask, request, send_file

app = Flask(__name__)

backend_url = os.environ.get('BACKEND_URL')
top_results = int(os.environ.get('TOP_RESULTS', 5))
day_coefficient = float(os.environ.get('DAY_COEFFICIENT', 0.8))
common_timeout = int(os.environ.get('TIMEOUT_COMMON', 1))
small_timeout = int(os.environ.get('TIMEOUT_SMALL', 10))
solutions_for_box = int(os.environ.get('SOLUTIONS_FOR_BOX', 10))


@app.post("/calculate")
def update_item():
    project_id = request.form.get('id')
    project = {}
    if project_id:
        resp = requests.get(
            os.path.join(backend_url, 'projects', str(project_id)))
        if resp.status_code == 200:
            project = resp.json()

    days = int(project.get('num_productive_days', 20))
    active_spread_params = project.get('active_spread_params', {})
    channels = active_spread_params.get('seismic_sensor_limit', 13200)
    active_line_y = int(active_spread_params.get('num_observations', 22) / 2)
    channels_in_active_spread = active_spread_params.get('num_channels', 144)
    start_point_coordinates = project.get('start_point_coordinates')
    start_point_coordinates = start_point_coordinates['geometry']
    start_point_coordinates = Point(12447986.6, 6597120.1)
    mesa_folder = 'algorithm/mesa'
    mesa_sps = active_spread_params.get('mesa_sps', None)
    mesa_rps = active_spread_params.get('mesa_rps', None)
    mesa_xps = active_spread_params.get('mesa_xps', None)
    mesa_sps_file = mesa_rps_file = mesa_xps_file = None
    if mesa_sps:
        mesa_sps_file = requests.get(os.path.join(backend_url, mesa_sps))
        if mesa_sps_file.status_code == 200:
            mesa_sps_file = mesa_sps_file.text
    if mesa_rps:
        mesa_rps_file = requests.get(os.path.join(backend_url, mesa_rps))
        if mesa_rps_file.status_code == 200:
            mesa_rps_file = mesa_rps_file.text
    if mesa_xps:
        mesa_xps_file = requests.get(os.path.join(backend_url, mesa_xps))
        if mesa_xps_file.status_code == 200:
            mesa_xps_file = mesa_xps_file.text

    timeout = common_timeout
    all_excitation_points = 13785  # все точки взрыва на всей территории
    crs = 28412
    points_per_section = 6  # количество точек в одном отрезке взрыва
    mask_excitation = None
    if mesa_sps_file is not None and mesa_rps_file is not None:
        mesa_exporter = MesaExporter(mesa_folder)
        mesa_excitation_path = mesa_exporter.get_shapefile_from_mesa(
            mesa_sps_file.split('\n'), 'S')
        mesa_reception_path = mesa_exporter.get_shapefile_from_mesa(
            mesa_rps_file.split('\n'), 'R')
        all_excitation_points = MesaExporter.count_all_points(
            mesa_excitation_path)
        crs = mesa_exporter.crs
        mask_excitation = mesa_exporter.get_mask_from_shp(
            mesa_excitation_path,
            mesa_reception_path,
        )
        points_per_section = MesaExporter.count_points_per_section(
            s_filepath=mesa_excitation_path,
            r_filepath=mesa_reception_path,
        )

    active_line_x = int(channels_in_active_spread / points_per_section / 2)
    daily_explode_area = int(
        all_excitation_points / points_per_section / days / day_coefficient)
    daily_explode_area = 130
    area_max = int(channels / points_per_section)
    save_result = SaveResults(mesa_folder=mesa_folder,
                              points_per_section=points_per_section,
                              crs=crs, active_line_x=active_line_x,
                              active_line_y=active_line_y)
    start_point_coordinates = save_result.convert_coordinates_to_indexes(
        start_point_coordinates)
    razmotka = Razmotka(active_line_x=active_line_x,
                        active_line_y=active_line_y, area_max=area_max,
                        top=top_results,
                        daily_explode_area=daily_explode_area,
                        matrix=mask_excitation,
                        wait_time=small_timeout,
                        solutions_for_box=solutions_for_box,
                        start_point_coordinates=start_point_coordinates)
    answers_list = razmotka.start_algorithm(timeout=timeout)
    if answers_list is None:
        return {'results': {}}
    i = 1
    answer_folders = {}
    for answer in answers_list:
        answer = answer[1]
        folder_name = 'razmotka/' + project_id + '/' + str(i)
        files = save_result.save_answer(
            answer, folder=folder_name)
        answer_folders[i] = files
        i += 1

    return {'results': answer_folders}


@app.get("/download/<path:file_path>")
def download_file(file_path=None):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_file(file_path)
    return f'{file_path}: No file'
