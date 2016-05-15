class AlgorithmOutput(object):
    __slots__ = ["score", "fit_time", "score_time"]

    def __init__(self, score, fit_time, score_time):
        self.score = score
        self.fit_time = fit_time
        self.score_time = score_time


class ExperimentOutput(object):
    __slots__ = ["bp_results", "elm_results", "counter"]

    def __init__(self, bp_results, elm_results, counter):
        self.bp_results = bp_results
        self.elm_results = elm_results
        self.counter = counter
