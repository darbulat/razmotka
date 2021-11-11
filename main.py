import datetime

from algorithm.save_results import SaveResults
from algorithm.tsp import Razmotka

from flask import Flask

app = Flask(__name__)


@app.route("/start")
def update_item():
    razmotka = Razmotka(n=10, m=10, area_max=2200,
                        daily_explode_area=130, top=2,
                        start_point='up-right')
    answers_list = razmotka.start_algorithm(timeout=7)
    if answers_list is None:
        return "No answer"
    print(len(answers_list))
    save_result = SaveResults(n=10, m=10)
    i = 0
    for answer_ in answers_list:
        answer_ = answer_[1]
        folder_name = 'razmotka/' + datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%S') + f'_{answer_.dispersion}_{i}'
        save_result.save_answer(answer_, folder=folder_name)
        i += 1

    return {"answers_list": answers_list}


if __name__ == "__main__":
    update_item()
