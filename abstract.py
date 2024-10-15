from abc import ABC, abstractmethod

class VideoProcessor(ABC):
    @abstractmethod
    def run(self, video):
        """
        Processes the given video.

        Args:
            video (Video): The video to process.
        """
        pass

    @abstractmethod
    def get_params_and_range(self):
        """
        Gets the parameters and range associated with the video processing.

        Returns:
            ParamsAndRange: The parameters and range.
        """
        pass
