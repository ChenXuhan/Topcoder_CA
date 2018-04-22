from math import pow


class Elorating:
    ELO_RESULT_WIN = 1
    ELO_RESULT_LOSS = -1
    ELO_RESULT_TIE = 0

    ELO_RATING_DEFAULT = 1500

    ratingA = 0
    ratingB = 0
    # timesA = 0
    # timesB = 0

    def __init__(self, ratingA=ELO_RATING_DEFAULT, ratingB=ELO_RATING_DEFAULT):
        self.ratingA = ratingA
        self.ratingB = ratingB
        # self.timesA = timesA
        # self.timesB = timesB

    def setResult(self, result):
        scoreAwin = self.computeScore(self.ratingA, self.ratingB)
        scoreBwin = self.computeScore(self.ratingB, self.ratingA)

        score_adjust = 0
        if result == self.ELO_RESULT_WIN:
            score_adjust = 1
        elif result == self.ELO_RESULT_LOSS:
            score_adjust = 0
        else:
            score_adjust = 0.5

        self.ratingA = self.ratingA + self.computeK(self.ratingA) * (score_adjust - scoreAwin)
        self.ratingB = self.ratingB + self.computeK(self.ratingB) * (1 - score_adjust - scoreBwin)

    def computeK(self, rating):
        if rating >= 2400:
            return 16
        elif rating >= 2100:
            return 24
        else:
            return 36

    def computeScore(self, rating1, rating2):
        return 1 / (1 + pow(10, (rating1 - rating2) / 400))


if __name__ == '__main__':
    print "elorating"