import os.path

import requests
from algorithm.save_results import SaveResults
from algorithm.tsp import Razmotka

from flask import Flask, request, send_file

app = Flask(__name__)

backend_url = os.environ.get('BACKEND_URL')


@app.post("/calculate")
def update_item():
    project_id = request.form.get('id')
    project = {}
    if project_id:
        resp = requests.get(
            os.path.join(backend_url, 'projects', str(project_id)))
        if resp.status_code == 200:
            project = resp.json()

    days = int(project.get('num_productive_days', 30))
    active_spread_params = project.get('active_spread_params', {})
    channels = active_spread_params.get('seismic_sensor_limit', 13200)
    active_line_y = int(active_spread_params.get('num_observations', 22) / 2)
    channels_in_active_spread = active_spread_params.get('num_channels', 144)
    mesa_sps = active_spread_params.get('mesa_sps', None)
    mesa_rps = active_spread_params.get('mesa_rps', None)
    mesa_xps = active_spread_params.get('mesa_xps', None)
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
    top_answers = 5
    timeout = 1  # TODO timeout = 7
    all_excitation_points = 13785  # все точки взрыва на всей территории
    points_per_section = 6  # количество точек в одном отрезке взрыва
    crs = 28412
    active_line_x = channels_in_active_spread / points_per_section / 2
    daily_explode_area = int(
        all_excitation_points / points_per_section / days / 0.8)
    area_max = int(channels / points_per_section)
    razmotka = Razmotka(active_line_x=active_line_x,
                        active_line_y=active_line_y, area_max=area_max,
                        start_point='up-right', top=top_answers,
                        daily_explode_area=daily_explode_area)
    answers_list = razmotka.start_algorithm(timeout=timeout)
    if answers_list is None:
        return "No answer"
    save_result = SaveResults(mesa_folder='algorithm/mesa/',
                              points_per_section=points_per_section,
                              crs=crs, active_line_x=active_line_x,
                              active_line_y=active_line_y)
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
