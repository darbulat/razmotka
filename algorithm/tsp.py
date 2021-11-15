import datetime
import heapq
import math
import multiprocessing as mp
import random

from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from typing import Tuple

import minizinc
import zython as zn
from zython.result import Result


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


@dataclass
class BBox:
    min_x: int = None
    min_y: int = None
    max_x: int = None
    max_y: int = None


class MinizincModel(zn.Model):
    timeout: timedelta = None
    nr_solutions: int = None

    def _solve(self, *how_to_solve, all_solutions, result_as, verbose, solver):
        solver = minizinc.Solver.lookup(solver)
        model = minizinc.Model()
        src = self.compile(how_to_solve)
        if verbose:
            print(src)
        model.add_string(src)
        inst = minizinc.Instance(solver, model)
        for name, param in self._ir.pars.items():
            inst[name] = param.value
        result = inst.solve(all_solutions=all_solutions,
                            nr_solutions=self.nr_solutions,
                            timeout=self.timeout,
                            processes=None)
        if result_as is None:
            return Result(result)
        else:
            return result_as(result)


class FillSmallBox(MinizincModel):
    def __init__(self, bbox: BBox,
                 rect_area: int, count: int = None,
                 nr_solutions: int = None,
                 timeout: timedelta = None):
        self.nr_solutions = nr_solutions
        self.timeout = timeout
        self.rect_area = rect_area
        self.area = (bbox.max_x - bbox.min_x) * (bbox.max_y - bbox.min_y)
        self.count = count if count else math.ceil(self.area / rect_area)
        self.min_x = zn.Array(
            zn.var(zn.var_par.types._range(bbox.min_x, bbox.max_x)),
            shape=self.count)
        self.min_y = zn.Array(
            zn.var(zn.var_par.types._range(bbox.min_y, bbox.max_y)),
            shape=self.count)
        self.max_x = zn.Array(
            zn.var(zn.var_par.types._range(bbox.min_x + 1, bbox.max_x + 1)),
            shape=self.count)
        self.max_y = zn.Array(
            zn.var(zn.var_par.types._range(bbox.min_y + 1, bbox.max_y + 1)),
            shape=self.count)
        self.area_ = (self._area())
        self.constraints = [
            (self.min_x[0] == bbox.min_x),
            (self.min_y[0] == bbox.min_y),
            (self.max_x[self.count - 1] == bbox.max_x),
            (self.max_y[self.count - 1] == bbox.max_y),
            zn.forall(
                zn.var_par.types._range(self.count - 1),
                lambda i: zn.forall(
                    zn.var_par.types._range(i + 1, self.count),
                    lambda j: (
                            (self.min_x[j] >= self.max_x[i])
                            | (self.min_y[j] >= self.max_y[i])
                    )
                )
            ),
            zn.forall(
                zn.var_par.types._range(self.count),
                lambda i: (
                                  (self.max_x[i] - self.min_x[i]) *
                                  (self.max_y[i] - self.min_y[i]) <= int(
                              rect_area * 1.1)
                          ) & (
                                  (self.max_x[i] - self.min_x[i]) *
                                  (self.max_y[i] - self.min_y[i]) >= int(
                              rect_area * 0.8)
                          ) & (
                                  (self.max_x[i] > self.min_x[i]) &
                                  (self.max_y[i] > self.min_y[i])
                          ) & (
                                  (self.max_x[i] - self.min_x[i]) >
                                  (self.max_y[i] - self.min_y[i])
                          )
            ),
            (self.area_ == self.area),
        ]

    def _area(self):
        return zn.sum(
            zn.var_par.types._range(self.count),
            lambda i: (self.max_x[i] - self.min_x[i]) *
                      (self.max_y[i] - self.min_y[i])
        )

    def find_rectangles(self):
        if self.count > 1:
            self.answer = self.solve_satisfy(
                verbose=True, solver='chuffed',
                all_solutions=False,
            )
            self.solution = self.answer.original.solution
            return len(self.answer.original.solution)

    def thin_out(self):
        solutions = []
        if len(self.solution) == 1:
            return self.solution
        for i in range(0, len(self.solution) - 1):
            if i >= len(self.solution):
                return solutions
            solutions.append(self.solution[i])
            for k in range(i + 1, len(self.solution)):
                if k >= len(self.solution):
                    break
                if self._is_near(self.solution[i], self.solution[k], 4):
                    del self.solution[k]
        return solutions

    def _is_near(self, solution1, solution2, delta):
        min_x_1 = solution1.min_x
        min_y_1 = solution1.min_y
        max_x_1 = solution1.max_x
        max_y_1 = solution1.max_y
        min_x_2 = solution2.min_x
        min_y_2 = solution2.min_y
        max_x_2 = solution2.max_x
        max_y_2 = solution2.max_y
        max_delta_max = 0
        for i in range(len(min_x_1)):
            delta_minx = abs(min_x_2[i] - min_x_1[i])
            delta_miny = abs(min_y_2[i] - min_y_1[i])
            delta_maxx = abs(max_x_2[i] - max_x_1[i])
            delta_maxy = abs(max_y_2[i] - max_y_1[i])
            delta_max = max(delta_minx, delta_miny, delta_maxx, delta_maxy)
            if delta_max > max_delta_max:
                max_delta_max = delta_max
        return max_delta_max <= delta


class TSP(MinizincModel):
    def __init__(self, distances, start_position: int = 0):
        self.distances = zn.Array(distances)
        self.size = len(distances)
        self.path = zn.Array(zn.var(zn.var_par.types._range(len(distances))),
                             shape=len(distances))
        self.cost = (self._cost())
        self.constraints = [
            self.cost < self.size,
            self.path[0] == start_position,
            zn.alldifferent(self.path)
        ]

    def _cost(self):
        return (zn.sum(zn.var_par.types._range(1, self.size),
                       lambda i: self.distances[self.path[i - 1],
                                                self.path[i]]))


class FillRectangles(MinizincModel):
    def __init__(self, matrix: list, rect_area: int, count: int = None,
                 nr_solutions: int = None, timeout: timedelta = None,
                 accuracy: float = 1.0):
        self.accuracy = accuracy
        self.nr_solutions = nr_solutions
        self.timeout = timeout
        self.rect_area = rect_area
        self.matrix = zn.Array(matrix)
        self.area = sum(sum(matrix, []))
        self.count = count if count else math.ceil(self.area / rect_area)
        self.min_x = zn.Array(zn.var(zn.var_par.types._range(len(matrix[0]))),
                              shape=self.count)
        self.min_y = zn.Array(zn.var(zn.var_par.types._range(len(matrix))),
                              shape=self.count)
        self.max_x = zn.Array(
            zn.var(zn.var_par.types._range(1, len(matrix[0]) + 1)),
            shape=self.count)
        self.max_y = zn.Array(
            zn.var(zn.var_par.types._range(1, len(matrix) + 1)),
            shape=self.count)
        self.count_ = (self._count())
        self.area_ = (self._area())
        self.constraints = [
            zn.forall(
                zn.var_par.types._range(self.count),
                lambda i:
                (
                        ((self.max_x[i] > self.min_x[i]) &
                         (self.max_y[i] > self.min_y[i]))
                        & ((self.max_x[i] - self.min_x[i]) *
                           (self.max_y[i] - self.min_y[i]) <= rect_area)
                )
            ),
            zn.forall(
                zn.var_par.types._range(self.count - 1),
                lambda i: zn.forall(
                    zn.var_par.types._range(i + 1, self.count),
                    lambda j: (
                            (self.min_x[j] >= self.max_x[i])
                            | (self.min_y[j] >= self.max_y[i])
                    )
                )
            ),
            (self.count_ == self.count),
            (self.area_ >= int(self.accuracy * self.area)),
        ]

    def generate_scheme(self):
        if self.count >= 1:
            answer = self.solve_satisfy(verbose=True,
                                        all_solutions=True,
                                        solver='chuffed')

            for solution in answer:
                yield solution

    def _count(self):
        return zn.sum(
            zn.var_par.types._range(self.count),
            lambda i: self.matrix[self.max_y[i] - 1, self.max_x[i] - 1]
        )

    def _area(self):
        return zn.sum(
            zn.var_par.types._range(self.count),
            lambda i: (self.max_x[i] - self.min_x[i]) *
                      (self.max_y[i] - self.min_y[i])
        )


class ReceptionPointsCounter:

    def __init__(self, matrix: list,
                 min_x: list, min_y: list,
                 max_x: list, max_y: list,
                 max_area: int, dic,
                 active_line_x, active_line_y,
                 start_points: Tuple):
        self.active_line_x = active_line_x
        self.active_line_y = active_line_y
        self.matrix_init = matrix
        self.min_x_init = min_x
        self.min_y_init = min_y
        self.max_x_init = max_x
        self.max_y_init = max_y
        self.max_area = max_area
        self.size = len(min_x)
        self.dic = dic
        self.start_points = start_points

    def create_graph_from_boxes(self):
        graph = [
            [100 for _ in range(self.size)]
            for _ in range(self.size)
        ]
        graph[self.size - 1][self.size - 1] = 0
        start_x = self.start_points[0]
        start_y = self.start_points[1]
        start_position = float('inf')
        for i in range(self.size - 1):
            graph[i][i] = 0
            set_y_i = set(range(self.min_y_init[i], self.max_y_init[i]))
            set_x_i = set(range(self.min_x_init[i], self.max_x_init[i]))
            if start_x in set_x_i and start_y in set_y_i:
                start_position = i
            for j in range(i + 1, self.size):
                set_y_j = set(range(self.min_y_init[j], self.max_y_init[j]))
                set_x_j = set(range(self.min_x_init[j], self.max_x_init[j]))
                if (self.max_x_init[i] == self.min_x_init[j]) and (
                        set_y_i.intersection(set_y_j)):
                    graph[i][j] = graph[j][i] = 1
                if (self.max_y_init[i] == self.min_y_init[j]) and (
                        set_x_i.intersection(set_x_j)):
                    graph[i][j] = graph[j][i] = 1
        return graph, start_position

    def fill_matrix_rp(self):
        graph, start_position = self.create_graph_from_boxes()
        if start_position == float('inf'):
            raise KeyError(f'Wrong start position {self.start_points}')
        model = TSP(graph, start_position=start_position)
        results = model.solve_satisfy(all_solutions=True, solver='chuffed')
        results = results.original.solution
        results = sorted(results, key=lambda i: i.path)
        stop_i = stop_key = None
        self.prev_max_area = float('inf')
        self.dispersion = float('inf')
        answer = None
        for result in results:
            if stop_i and stop_key == result.path[stop_i]:
                continue
            self.prev_max_area_inst = 0
            c = 0
            area = 0
            day = 0
            self.min_x = [self.min_x_init[i] for i in result.path]
            self.min_y = [self.min_y_init[i] for i in result.path]
            self.max_x = [self.max_x_init[i] for i in result.path]
            self.max_y = [self.max_y_init[i] for i in result.path]
            self.matrix = deepcopy(self.matrix_init)
            self.matrix_unwinding = deepcopy(self.matrix_init)
            self.active_matrix = self._get_active_placement(
                deepcopy(self.matrix))
            unwinding_day = 0
            max_common_area = 0
            summa_area = 0
            dispersion = 0
            min_common_area = float('inf')
            next_area = self._get_next_unwinding_area(0, day, day)
            area += next_area
            for i in range(self.size - 1):
                winding_area = 0
                unwinding_day += 1
                self._blow_up(i)
                need_area = self._get_next_unwinding_area(
                    i + 1, unwinding_day, unwinding_day, dry=True)
                if area + need_area > self.max_area:
                    winding_area = self._winding_pp(i, need_area)
                    if winding_area < need_area:
                        stop_i = i
                        stop_key = result.path[i]
                        break
                    area -= winding_area

                next_area = self._get_next_unwinding_area(
                    i + 1, unwinding_day, unwinding_day)
                area += next_area

                c += 1
                common_area = next_area + winding_area
                summa_area += common_area
                if common_area > max_common_area:
                    max_common_area = common_area
                if common_area < min_common_area:
                    min_common_area = common_area
                dispersion = max_common_area - min_common_area
                if dispersion > self.dic.get('dispersion', self.dispersion):
                    stop_i = i
                    stop_key = result.path[i]
                    break
            else:
                self._blow_up(c)
                winding_area = self._winding_pp(c)
                if self.dic.get('dispersion', self.dispersion) > dispersion:
                    self.dic['dispersion'] = dispersion
                    answer = Answer(self.min_x, self.min_y,
                                    self.max_x, self.max_y,
                                    self.matrix, self.matrix_unwinding,
                                    self.prev_max_area_inst,
                                    dispersion=dispersion)
                    self.prev_max_area = self.prev_max_area_inst
                    self.dispersion = dispersion
        return answer

    def _get_active_placement(self, active_matrix):
        for i in range(self.size):
            for y in range(self.min_y[i],
                           self.max_y[i] + 2 * self.active_line_y):
                for x in range(self.min_x[i],
                               self.max_x[i] + 2 * self.active_line_x):
                    active_matrix[y][x] += 1
        for y in range(len(active_matrix)):
            for x in range(len(active_matrix[0])):
                if active_matrix[y][x] == 0:
                    active_matrix[y][x] = self.size
        return active_matrix

    def _get_next_unwinding_area(self, i, c, day, dry=False):
        next_area = 0
        for x in range(self.min_x[i],
                       self.max_x[i] + 2 * self.active_line_x):
            for y in range(self.min_y[i],
                           self.max_y[i] + 2 * self.active_line_y):
                if self.matrix[y][x] == 0:
                    if not dry:
                        self.matrix[y][x] = [c, day]
                        self.matrix_unwinding[y][x] = [c, day]
                    next_area += 1
        return next_area

    def _blow_up(self, i):
        for x in range(self.min_x[i],
                       self.max_x[i] + 2 * self.active_line_x):
            for y in range(self.min_y[i],
                           self.max_y[i] + 2 * self.active_line_y):
                self.active_matrix[y][x] -= 1

    def _winding_pp(self, day: int, need_area: int = None):
        winding_area = 0
        for y in range(len(self.matrix)):
            for x in range(len(self.matrix[0])):
                if self.active_matrix[y][x] == 0:
                    self.matrix[y][x] = [0, day]
                    self.active_matrix[y][x] -= 1
                    winding_area += 1
                    if need_area and need_area == winding_area:
                        return need_area
        return winding_area


class Razmotka:

    def __init__(self, active_line_x, active_line_y,
                 area_max=2200,
                 start_point_coordinates=None,
                 top=5,
                 daily_explode_area=130,
                 solutions_for_box=10,
                 wait_time=10,
                 matrix=None):

        if start_point_coordinates is None:
            start_point_coordinates = (0, 0)
        self.active_line_x = active_line_x
        self.active_line_y = active_line_y
        if matrix is None:
            self.matrix = [[
                0 if ((x > 39 and 18 < y < 49)
                      or (x > 24 and 49 <= y < 61)) else 1
                for x in range(42)
            ] for y in range(61)]
        else:
            self.matrix = matrix
        self.matrix_pp = [
            [0 for _ in range(len(self.matrix[0]) + 2 * self.active_line_x)]
            for _ in range(len(self.matrix) + 2 * self.active_line_y)
        ]
        self.rect_area = 2000
        self.area_max = area_max
        self.start_point_coordinates = start_point_coordinates
        self.top = top
        self.daily_explode_area = daily_explode_area
        self.solutions_for_box = solutions_for_box
        self.wait_time = wait_time

    def find_unwinding_scheme(self, sol1, sol2, sol3, dic):
        x_min = sol1['min_x'] + sol2['min_x'] + sol3['min_x']
        y_min = sol1['min_y'] + sol2['min_y'] + sol3['min_y']
        x_max = sol1['max_x'] + sol2['max_x'] + sol3['max_x']
        y_max = sol1['max_y'] + sol2['max_y'] + sol3['max_y']

        reception_points = ReceptionPointsCounter(
            deepcopy(self.matrix_pp),
            x_min, y_min, x_max, y_max,
            max_area=self.area_max,
            dic=dic,
            active_line_x=self.active_line_x,
            active_line_y=self.active_line_y,
            start_points=self.start_point_coordinates
        )
        answer = reception_points.fill_matrix_rp()
        if answer:
            print(f'{answer.dispersion=}')
        else:
            print('No answer')
        return answer

    def find_unwinding_scheme_parallel(self, coords_0: list, coords_1: list,
                                       coords_2: list, answers: list, dic):
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [
                executor.submit(self.find_unwinding_scheme,
                                solution1.__dict__,
                                solution2.__dict__,
                                solution3.__dict__,
                                dic)
                for solution1, solution2, solution3 in zip(
                    coords_0, coords_1, coords_2
                )
            ]

            ok = wait(futures, timeout=None, return_when=ALL_COMPLETED)
            for future in ok.done:
                answer = future.result()
                if answer:
                    i = random.randint(-10000, 10000)
                    print(i)
                    heapq.heappushpop(answers,
                                      ((-answer.dispersion, i), answer))

    def start_algorithm(self, timeout=1):
        fill_rectangles = FillRectangles(matrix=self.matrix,
                                         rect_area=self.rect_area,
                                         count=3,
                                         accuracy=1)
        answers = [((-float('inf'), i), Answer([], [], [], [], [], [], 0, 0))
                   for i in range(self.top)]
        heapq.heapify(answers)
        dic = dict()
        start_time = datetime.datetime.now()

        for ans in fill_rectangles.generate_scheme():
            if ans is None:
                continue
            coords = self.find_rectangles_for_solution(solution=ans)
            if not coords:
                continue
            i = 0
            coords_0 = []
            coords_1 = []
            coords_2 = []
            for solution0, solution1, solution2 in product(
                    coords[0], coords[1], coords[2]
            ):
                if i < mp.cpu_count():
                    coords_0.append(solution0)
                    coords_1.append(solution1)
                    coords_2.append(solution2)
                    i += 1
                else:
                    self.find_unwinding_scheme_parallel(
                        coords_0, coords_1, coords_2, answers, dic)
                    i = 0
                if (datetime.datetime.now() - start_time
                ).seconds / 60 > timeout:
                    print((datetime.datetime.now() - start_time).seconds / 60)
                    return answers

            return answers

    def find_rectangles_for_solution(self, solution):
        boxes = [BBox(
            solution.min_x[i],
            solution.min_y[i],
            solution.max_x[i],
            solution.max_y[i]
        ) for i in range(solution.count_)]
        answers_for_box = []
        for bbox in boxes:
            fill_small_box = FillSmallBox(
                bbox, self.daily_explode_area,
                nr_solutions=self.solutions_for_box,
                timeout=timedelta(seconds=self.wait_time)
            )
            solution_count = fill_small_box.find_rectangles()
            if not solution_count:
                return None
            random.shuffle(fill_small_box.solution)
            # print(solution_for_box)
            print(solution_count)
            print(len(fill_small_box.solution))
            answers_for_box.append(fill_small_box.solution)
        return answers_for_box
