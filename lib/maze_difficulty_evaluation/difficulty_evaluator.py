class DifficultyEvaluator:
    """
    Class that evaluate the difficulty of a maze following the approach of the paper "The Quest for the Perfect Perfect-Maze" by Kim and Crawfis.
    Args:
        w (array[float]): weights of the metrics used.
        t (array[float]): target values of the metrics used.
        a (array[float]): values of the metrics used.
    Return:
        float: difficulty of the maze.
    """
    def __init__(self, w: list[float], a: list[float],t: list[float]):
        self.w = w
        self.a = a
        self.t = t

    def evaluate(self,):
        difficulty = 0
        W = sum(self.w)
        for i in range(len(self.a)):
            difficultty += (self.w[i] / W)* (1- (abs(self.a[i]-self.t[i])/max(self.t[i],self.a[i])))



