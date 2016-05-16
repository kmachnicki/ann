class Sample(object):
    __slots__ = ["score", "fit_time", "score_time"]

    def __init__(self, score, fit_time, score_time):
        self.score = score
        self.fit_time = fit_time
        self.score_time = score_time


class Samples(object):
    __slots__ = ["scores", "fit_times", "score_times"]

    def __init__(self, scores, fit_times, score_times):
        self.scores = scores
        self.fit_times = fit_times
        self.score_times = score_times


class ExperimentOutput(object):
    __slots__ = ["bp_samples", "elm_samples", "counter"]

    def __init__(self, bp_samples, elm_samples, counter):
        self.bp_samples = bp_samples
        self.elm_samples = elm_samples
        self.counter = counter


class ExperimentWrapper(object):
    def __init__(self):
        self.scores = []
        self.fit_times = []
        self.score_times = []

    def add_sample(self, sample):
        self.scores.append(sample.score)
        self.fit_times.append(sample.fit_time)
        self.score_times.append(sample.score_time)

    def samples(self):
        return Samples(self.scores, self.fit_times, self.score_times)
