from src import plot, sort, util

if __name__ == "__main__":
    plot.init()
    array = util.random_array(1, 100, 50)
    visualizer = plot.SortingVisualizer(plot.VisualizerMode.RUN, nrows=3, ncols=3, fps=50)
    # 若要自定義排序算法，請繼承 sort.SortingAlgorithm
    visualizer.add_sorting_ax(sort.RadixSort(array), index=1)
    visualizer.add_sorting_ax(sort.CountingSort(array), index=2)
    visualizer.add_sorting_ax(sort.QuickSort(array), index=3)
    visualizer.add_sorting_ax(sort.HeapSort(array), index=4)
    visualizer.add_sorting_ax(sort.MergeSort(array), index=5)
    visualizer.add_sorting_ax(sort.DoubleSelectionSort(array), index=6)
    visualizer.add_sorting_ax(sort.CombSort(array), index=7)
    visualizer.add_sorting_ax(sort.InsertionSort(array), index=8)
    visualizer.add_sorting_ax(sort.BubbleSort(array), index=9)
    visualizer.animation_start()
