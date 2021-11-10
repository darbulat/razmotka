import datetime

from algorithm.save_results import SaveResults
from algorithm.tsp import Razmotka

from flask import Flask

app = Flask(__name__)


@app.route("/start")
def update_item():
    razmotka = Razmotka(n=5, m=5)
    answers_list = razmotka.start_algorithm(timeout=1)
    if answers_list is None:
        return "No answer"
    print(len(answers_list))
    for answer_ in answers_list:
        answer_ = answer_[1]
        folder_name = 'razmotka/' + datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%S') + f'{answer_.dispersion}'
        save_result = SaveResults()
        save_result.save_answer(answer_, folder=folder_name)

    return {"answers_list": answers_list}


if __name__ == "__main__":
    update_item()
