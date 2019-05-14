# I copied this bit of code from somewhere but I do not remember the source :(


class StatMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.total = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.total = (self.sum / self.count) * 100
