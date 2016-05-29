import numpy as np


class Sample(object):
    __slots__ = ["score", "fit_time", "score_time", "conf_matrix"]

    def __init__(self, score, fit_time, score_time, conf_matrix):
        self.score = score
        self.fit_time = fit_time
        self.score_time = score_time
        self.conf_matrix = conf_matrix

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "score":
                return self.score
            elif key == "fit_time":
                return self.fit_time
            elif key == "score_time":
                return self.score_time
            else:
                return None


class Samples(object):
    __slots__ = ["scores", "fit_times", "score_times", "conf_matrices"]

    def __init__(self, scores, fit_times, score_times, conf_matrices):
        self.scores = scores
        self.fit_times = fit_times
        self.score_times = score_times
        self.conf_matrices = conf_matrices

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "scores":
                return self.scores
            elif key == "fit_times":
                return self.fit_times
            elif key == "score_times":
                return self.score_times
            elif key == "conf_matrices":
                return self.conf_matrices
            else:
                return None


class ExperimentOutput(object):
    __slots__ = ["bp_samples", "elm_samples", "counter"]

    def __init__(self, bp_samples, elm_samples, counter):
        self.bp_samples = bp_samples
        self.elm_samples = elm_samples
        self.counter = counter

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "bp":
                return self.bp_samples
            elif key == "elm":
                return self.elm_samples
            elif key == "counter":
                return self.counter
            else:
                return None


class ExperimentWrapper(object):
    def __init__(self):
        self.scores = []
        self.fit_times = []
        self.score_times = []
        self.conf_matrices = []

    def add_sample(self, sample):
        self.scores.append(sample.score)
        self.fit_times.append(sample.fit_time)
        self.score_times.append(sample.score_time)
        if len(sample.conf_matrix) > 1:
            self.conf_matrices.append(sample.conf_matrix)

    def samples(self):
        return Samples(self.scores, self.fit_times, self.score_times, self.conf_matrices)


class GraphValues(object):
    __slots__ = ["y_bp", "y_elm", "e_bp", "e_elm"]

    def __init__(self, y_bp, y_elm, e_bp, e_elm):
        self.y_bp = y_bp
        self.y_elm = y_elm
        self.e_bp = e_bp
        self.e_elm = e_elm

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "y_bp":
                return self.y_bp
            elif key == "y_elm":
                return self.y_elm
            elif key == "e_bp":
                return self.e_bp
            elif key == "e_elm":
                return self.e_elm
            else:
                return None


class GraphValuesWrapper(object):
    def __init__(self):
        self.y_bp = []
        self.y_elm = []
        self.e_bp = []
        self.e_elm = []

    def add_value(self, results_bp, results_elm):
        self.y_bp.append(np.mean(results_bp))
        self.y_elm.append(np.mean(results_elm))
        self.e_bp.append(np.std(results_bp))
        self.e_elm.append(np.std(results_elm))

    def convert_s_to_ms(self):
        self.y_bp = [val * 1000 for val in self.y_bp]
        self.y_elm = [val * 1000 for val in self.y_elm]
        self.e_bp = [val * 1000 for val in self.e_bp]
        self.e_elm = [val * 1000 for val in self.e_elm]

    def values(self):
        return GraphValues(self.y_bp, self.y_elm, self.e_bp, self.e_elm)
