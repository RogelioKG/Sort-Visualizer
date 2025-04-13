from collections.abc import Generator
from enum import Enum
from typing import Literal

from matplotlib import animation, style, ticker
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from . import sort


def init() -> None:
    """全域初始化一些 matplotlib 設定"""
    style.use("dark_background")  # 風格
    plt.rcParams["font.family"] = "monospace"  # 字體


class AxState(Enum):
    """坐標軸狀態"""

    SORTING = "sorting"  # 排序中階段 (SORTING 階段)
    SORTED = "sorted"  # 排序完成階段 (SORTED 階段)
    PENDING = "pending"  # 結束動畫階段 (PENDING 階段)
    COMPLETE = "complete"  # 完成階段 (COMPLETE 階段)


class BarState(Enum):
    """bar 狀態 (注意：值代表顏色)"""

    INACTIVE = "silver"  # 未變動的元素的顏色
    ACTIVE = "salmon"  # 變動元素的顏色
    COMPLETE = "yellowgreen"  # 結束動畫遍歷時的顏色


class SortingAx:
    def __init__(
        self, algo: sort.SortingAlgorithm, *, fig: plt.Figure, pos: tuple[int, int, int]
    ) -> None:
        """負責呈現排序算法的座標軸

        Parameters
        ----------
        algo : sort.SortingAlgorithm
            排序算法
        fig : plt.Figure
            所依附的圖表
        pos : tuple[int, int, int]
            坐標軸在圖表上的位置 (`nrows`, `ncols`, `index`)
        """
        self.algo = algo
        self.ax = fig.add_subplot(*pos)
        self.bar_artists = self.ax.bar(
            range(len(algo.array)),
            algo.array,
            color=BarState.INACTIVE.value,
            width=1,
            align="edge",
        )
        self.focus_bar_artists: list[Rectangle] = []
        self.state = AxState.SORTING
        self.counter = 0
        self._setup_ax()

    def _setup_ax(self) -> None:
        """座標軸布局設定"""
        self.ax.set_title(self.algo.name)
        self.ax.set_xmargin(0)
        self.ax.set_ymargin(0)
        self.ax.xaxis.set_major_locator(ticker.NullLocator())
        self.ax.xaxis.set_minor_locator(ticker.NullLocator())
        self.ax.yaxis.set_major_locator(ticker.NullLocator())
        self.ax.yaxis.set_minor_locator(ticker.NullLocator())

    def set_focus_bar_color(self, state: BarState) -> None:
        """設定焦點 bar 的顏色"""
        for focus_bar_artist in self.focus_bar_artists:
            focus_bar_artist.set_color(state.value)

    def clear_focus_bars(self) -> None:
        """清除所有焦點 bar"""
        self.focus_bar_artists.clear()

    def add_focus_bar(self, bar: Rectangle) -> None:
        """新增焦點 bar

        Parameters
        ----------
        bar : Rectangle
            bar 藝術家
        """
        self.focus_bar_artists.append(bar)

    def sorting_state_next_step(self) -> None:
        """SORTING 階段時，應執行的下一步"""
        assert self.state == AxState.SORTING

        try:
            updated_values = next(self.algo.step_generator)  # 變動值的產出
        except StopIteration:
            self.state = AxState.SORTED  # 排序完畢階段
        else:
            self.set_focus_bar_color(BarState.INACTIVE)
            self.clear_focus_bars()
            for updated_value in updated_values:
                i, updated_height = updated_value
                updated_bar: Rectangle = self.bar_artists[i]
                updated_bar.set_height(updated_height)  # 只更新變動值 bar 的高度
                self.add_focus_bar(updated_bar)
            self.set_focus_bar_color(BarState.ACTIVE)  # 和變動值 bar 的顏色

    def sorted_state_next_step(self) -> None:
        """SORTED 階段時，應執行的下一步"""
        assert self.state == AxState.SORTED

        self.set_focus_bar_color(BarState.INACTIVE)
        self.clear_focus_bars()
        self.state = AxState.PENDING

    def pending_state_next_step(self) -> None:
        """PENDING 階段時，應執行的下一步"""
        assert self.state == AxState.PENDING

        if self.counter >= len(self.bar_artists):
            self.set_focus_bar_color(BarState.INACTIVE)
            self.clear_focus_bars()
            self.state = AxState.COMPLETE  # 完成階段
            return

        bar: Rectangle = self.bar_artists[self.counter]
        self.set_focus_bar_color(BarState.INACTIVE)
        self.clear_focus_bars()
        self.add_focus_bar(bar)
        self.set_focus_bar_color(BarState.COMPLETE)

        self.counter += 1


class VisualizerMode(Enum):
    """Visualizer 如何呈現動畫 (呈現模式)"""

    RUN = "run"  # 實時運行
    SAVE = "save"  # 保存動畫


class SortingVisualizer:
    def __init__(
        self,
        mode: VisualizerMode,
        *,
        figsize: tuple[int, int] = (10, 10),
        nrows: int,
        ncols: int,
        fps: int,
    ) -> None:
        """排序視覺化管理器

        Parameters
        ----------
        mode : VisualizerMode
            呈現模式，指定是否實時運行或保存動畫
        figsize : tuple[int, int], optional
            圖表大小，預設為 `(10, 10)`
        nrows : int
            行數
        ncols : int
            列數
        fps : int
            每秒畫面更新次數
        """
        self.mode = mode
        self.nrows = nrows
        self.ncols = ncols
        self.fps = fps
        self.fig = plt.figure(figsize=figsize)
        self.sorting_axes: list[SortingAx] = []
        self._end_signal_generator = self._end_signal()
        self._setup_figure()

    def add_sorting_ax(self, algo: sort.SortingAlgorithm, *, index: int) -> None:
        """新增座標軸

        Parameters
        ----------
        algo : sort.SortingAlgorithm
            排序算法
        index : int
            座標軸位置

        Index
        -----
        例如一個 3x3 的圖表中，每個對應的索引位置長這樣
        ```
        +-----+-----+-----+
        |  1  |  2  |  3  |
        +-----+-----+-----+
        |  4  |  5  |  6  |
        +-----+-----+-----+
        |  7  |  8  |  9  |
        +-----+-----+-----+
        ```
        """
        sorting_ax = SortingAx(algo, fig=self.fig, pos=(self.nrows, self.ncols, index))
        self.sorting_axes.append(sorting_ax)

    def animation_start(self) -> None:
        """根據呈現模式，開始運行動畫"""
        common_kwargs = {
            "fig": self.fig,
            "func": self._update,
            "interval": 1000 / self.fps,
            "blit": True,
        }
        if self.mode == VisualizerMode.RUN:
            ani = animation.FuncAnimation(
                frames=1,
                repeat=True,
                **common_kwargs,
            )
            plt.show()
        elif self.mode == VisualizerMode.SAVE:
            try:
                ani = animation.FuncAnimation(
                    frames=self._end_signal_generator,
                    cache_frame_data=False,
                    **common_kwargs,
                )
                ani.save("sorting.gif", fps=self.fps, dpi=200)
            except StopIteration:
                print("Saved!")

    def _update(self, frame: int) -> list[plt.Artist]:
        """更新每一幀的排序視覺化

        Parameters
        ----------
        frame : int
            當前幀數

        Returns
        -------
        list[plt.Artist]
            回傳每個座標軸的所有 bar 藝術家
        """
        complete = 0
        for sorting_ax in self.sorting_axes:
            if sorting_ax.state == AxState.SORTING:  # 排序中階段
                sorting_ax.sorting_state_next_step()
            elif sorting_ax.state == AxState.SORTED:  # 排序完成階段
                sorting_ax.sorted_state_next_step()
            elif sorting_ax.state == AxState.PENDING:  # 結束動畫階段
                sorting_ax.pending_state_next_step()
            elif sorting_ax.state == AxState.COMPLETE:  # 完成階段
                complete += 1

        if complete == len(self.sorting_axes) and complete != 0:
            self._stop()  # 全完成後，關閉動畫

        return [
            bar_artist for sorting_ax in self.sorting_axes for bar_artist in sorting_ax.bar_artists
        ]

    def _setup_figure(self) -> None:
        """圖表布局設定"""
        self.fig.suptitle("Sorting", fontsize=24, fontweight="bold")
        self.fig.tight_layout(pad=2.0)
        self.fig.subplots_adjust(hspace=0.3, wspace=0.1)

    def _end_signal(self) -> Generator[Literal[1], None, None]:
        """發出結束訊號 (`StopIteration`) 的產生器

        Returns
        -------
        Generator[Literal[1], None, None]
            一個簡單的產生器，每次皆返回 1，直到 `send(True)` 為止
        """
        # 被使用於 `VisualizerMode.SAVE` 模式，
        # 由於在保存動畫時，matplotlib 是根據指定的 frames 來決定要產生多少幀。
        # 為了讓其自行判斷停下的位置，此處使用一個神奇小技巧：
        # 利用 frames 也可傳入產生器且 `StopIteration` 會停止動畫的特性，
        # 讓更新函數 _update 握有此產生器，直到所有排序都進入完成階段後，
        # 使用 `send(True)` 讓此產生器發出結束訊號，強迫動畫停止。
        end = False
        while not end:
            end = yield 1

    def _stop(self):
        """強迫動畫停止"""
        if self.mode == VisualizerMode.RUN:
            plt.close(self.fig)
        elif self.mode == VisualizerMode.SAVE:
            self._end_signal_generator.send(True)  # 發出結束訊號
