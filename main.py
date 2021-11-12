import datetime

from algorithm.save_results import SaveResults
from algorithm.tsp import Razmotka

from flask import Flask

app = Flask(__name__)


@app.route("/start")
def update_item():
    all_excitation_points = 13785
    days = 30
    points_per_section = 6
    channels = 13200
    N = 12
    M = 11
    crs = 28412
    top_answers = 2
    timeout = 1
    daily_explode_area = int(
        all_excitation_points / points_per_section / days / 0.8)
    area_max = channels / points_per_section
    razmotka = Razmotka(n=N, m=M, area_max=area_max,
                        daily_explode_area=daily_explode_area, top=top_answers,
                        start_point='up-right')
    answers_list = razmotka.start_algorithm(timeout=timeout)
    if answers_list is None:
        return "No answer"
    print(len(answers_list))
    save_result = SaveResults(mesa_folder='algorithm/mesa/',
                              points_per_section=points_per_section,
                              crs=crs)
    i = 0
    answer_folders = []
    for answer in answers_list:
        answer = answer[1]
        folder_name = 'razmotka/' + datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%S') + f'_{answer.dispersion}_{i}'
        save_result.save_answer(
            answer, folder=folder_name)
        answer_folders.append(folder_name)
        i += 1

    return {'files': answer_folders}


if __name__ == "__main__":
    update_item()
