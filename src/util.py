import random as rd


def random_array(low: int, high: int, n: int) -> list[int]:
    """生成一個隨機整數陣列，範圍：[low, high]、長度：n

    Parameters
    ----------
    low : int
        最小值
    high : int
        最大值
    n : int
        長度

    Returns
    -------
    list[int]
        陣列
    """
    array = [rd.randint(low, high) for _ in range(n)]
    rd.shuffle(array)
    return array


def increment_array(low: int, high: int) -> list[int]:
    """生成一個升序陣列，然後隨機打亂順序，範圍：[low, high]

    Parameters
    ----------
    low : int
        起始值
    high : int
        結束值

    Returns
    -------
    list[int]
        陣列
    """
    array = [*range(low, high + 1, 1)]
    rd.shuffle(array)
    return array
