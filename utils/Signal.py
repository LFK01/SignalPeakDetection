from utils import Peak


class Signal:
    def __init__(self,
                 length: int,
                 peak_n: int,
                 peaks: list[Peak],
                 data: list[int]):
        self.length = length
        self.peak_n = peak_n
        self.peaks = peaks
        self.data = data


class SignalLight:
    def __init__(self,
                 length: int,
                 peaks: list[float],
                 data: list[int]):
        self.length = length
        self.peaks = peaks
        self.data = data
