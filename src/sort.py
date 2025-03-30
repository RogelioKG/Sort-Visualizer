from abc import ABC, abstractmethod
from collections.abc import Generator

from src.types import UpdateValue


class SortingAlgorithm(ABC):
    """排序算法抽象類別 (若要自訂新排序算法，請繼承此類別)"""

    def __init__(self, array: list[int]) -> None:
        """將傳入的陣列複製一份，將被用於排序

        Parameters
        ----------
        array : list[int]
            未排序陣列
        """
        self.array = array.copy()
        self.step_generator = self._step()
        self.name = self.__class__.__name__
        self._is_sorted = False

    @abstractmethod
    def _step(self) -> Generator[list[UpdateValue], None, None]:
        """排序過程更新值產生器

        Yields
        ------
        Generator[list[UpdateValue], None, None]
            每次排序算法產生交換元素、回填元素這些 O(1) 變動時，紀錄並產出這些元素的索引與值
        """
        pass

    def get_updated_values(self, *indices: int) -> list[UpdateValue]:
        """獲取變動元素們的索引與值

        Returns
        -------
        list[UpdateValue]
            變動元素們的索引與值
        """
        return [(i, self.array[i]) for i in indices]

    def mark_sorted(self) -> None:
        """將排序算法設定為完成狀態"""
        self._is_sorted = True
        print(f"{self.name}: Done!")

    def is_sorted(self) -> bool:
        """排序算法是否已完成

        Returns
        -------
        bool
            完成狀態
        """
        return self._is_sorted


class BubbleSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        n = len(self.array)

        for i in range(n - 1):
            swapped = False
            for j in range(n - 1 - i):
                if self.array[j] > self.array[j + 1]:
                    swapped = True
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]
                    yield self.get_updated_values(j, j + 1)
            if not swapped:
                break

        self.mark_sorted()


class InsertionSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        n = len(self.array)

        for i in range(1, n):
            key = self.array[i]
            j = i - 1
            while j >= 0 and self.array[j] > key:
                self.array[j + 1] = self.array[j]
                yield self.get_updated_values(j + 1)
                j -= 1
            self.array[j + 1] = key
            yield self.get_updated_values(j + 1)

        self.mark_sorted()


class QuickSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)

    def _quick_sort(self, start: int, end: int) -> Generator[list[UpdateValue], None, None]:
        if start >= end:
            return

        left = start
        right = end
        pivot = self.array[end]

        while left != right:
            while (self.array[left] <= pivot) and (left != right):
                left += 1
            while (self.array[right] >= pivot) and (left != right):
                right -= 1
            if left != right:
                self.array[left], self.array[right] = self.array[right], self.array[left]
                yield self.get_updated_values(left, right)

        self.array[left], self.array[end] = self.array[end], self.array[left]
        yield self.get_updated_values(left, end)

        yield from self._quick_sort(start, left - 1)
        yield from self._quick_sort(left + 1, end)

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        yield from self._quick_sort(0, len(self.array) - 1)
        self.mark_sorted()


class HeapSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)

    def _max_heapify(self, root: int, end: int) -> Generator[list[UpdateValue], None, None]:
        dad = root
        son = dad * 2 + 1

        while son <= end:
            if (son + 1 <= end) and (self.array[son] < self.array[son + 1]):
                son += 1
            if self.array[dad] > self.array[son]:
                return
            else:
                self.array[dad], self.array[son] = self.array[son], self.array[dad]
                yield self.get_updated_values(dad, son)
                dad = son
                son = dad * 2 + 1

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        n = len(self.array)

        for i in range(n // 2 - 1, -1, -1):
            yield from self._max_heapify(i, n - 1)
        for i in range(n - 1, 0, -1):
            self.array[0], self.array[i] = self.array[i], self.array[0]
            yield self.get_updated_values(0, i)
            yield from self._max_heapify(0, i - 1)

        self.mark_sorted()


class MergeSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)

    def _conquer(self, left: int, mid: int, right: int) -> Generator[list[UpdateValue], None, None]:
        i = left
        j = mid + 1
        temp_array: list[int] = []

        while (i <= mid) and (j <= right):
            if self.array[i] < self.array[j]:
                temp_array.append(self.array[i])
                i += 1
            else:
                temp_array.append(self.array[j])
                j += 1

        while i <= mid:
            temp_array.append(self.array[i])
            i += 1
        while j <= right:
            temp_array.append(self.array[j])
            j += 1

        index = 0
        while left <= right:
            self.array[left] = temp_array[index]
            yield self.get_updated_values(left)
            index += 1
            left += 1

    def _divide(self, left: int, right: int) -> Generator[list[UpdateValue], None, None]:
        if left == right:
            return

        mid = (left + right) // 2

        yield from self._divide(left, mid)
        yield from self._divide(mid + 1, right)
        yield from self._conquer(left, mid, right)

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        n = len(self.array)
        yield from self._divide(0, n - 1)
        self.mark_sorted()


class CountingSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        max_value = max(self.array)
        min_value = min(self.array)
        buckets = [0] * (max_value - min_value + 1)

        for num in self.array:
            i = num - min_value
            buckets[i] += 1

        j = 0
        for i, count in enumerate(buckets):
            num = i + min_value
            for _ in range(count):
                self.array[j] = num
                yield self.get_updated_values(j)
                j += 1

        self.mark_sorted()


class RadixSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)

    def _counting_sort(self, exp: int) -> Generator[list[UpdateValue], None, None]:
        n = len(self.array)
        buckets = [0] * 10
        output = [0] * n

        for num in self.array:
            i = (num // exp) % 10
            buckets[i] += 1

        for i in range(1, 10):
            buckets[i] += buckets[i - 1]

        for i in range(n - 1, -1, -1):
            num = self.array[i]
            index = (num // exp) % 10
            output[buckets[index] - 1] = num
            buckets[index] -= 1

        for i in range(n):
            self.array[i] = output[i]
            yield self.get_updated_values(i)

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        max_value = max(self.array)

        exp = 1
        while max_value // exp > 0:
            yield from self._counting_sort(exp)
            exp *= 10

        self.mark_sorted()


class DoubleSelectionSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        left = 0
        right = len(self.array) - 1

        while left < right:
            min_index, max_index = left, left
            for i in range(left, right + 1):
                if self.array[i] < self.array[min_index]:
                    min_index = i
                elif self.array[i] > self.array[max_index]:
                    max_index = i

            self.array[left], self.array[min_index] = self.array[min_index], self.array[left]
            yield self.get_updated_values(left, min_index)

            if max_index == left:
                max_index = min_index

            self.array[right], self.array[max_index] = self.array[max_index], self.array[right]
            yield self.get_updated_values(right, max_index)

            left += 1
            right -= 1

        self.mark_sorted()


class CombSort(SortingAlgorithm):
    def __init__(self, array: list[int]) -> None:
        super().__init__(array)
        self.shrink = 1.3  # 收縮因子

    def _step(self) -> Generator[list[UpdateValue], None, None]:
        gap = len(self.array)
        sorted = False

        while not sorted:
            gap = int(gap / self.shrink)
            if gap <= 1:
                gap = 1
                sorted = True

            for i in range(len(self.array) - gap):
                if self.array[i] > self.array[i + gap]:
                    self.array[i], self.array[i + gap] = self.array[i + gap], self.array[i]
                    yield self.get_updated_values(i, i + gap)
                    sorted = False

        self.mark_sorted()
