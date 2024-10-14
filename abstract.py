class VideoProcessor:
    def run(self, video):
        """
        Processes the given video.

        Args:
            video (Video): The video to process.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_params_and_range(self):
        """
        Gets the parameters and range associated with the video processing.

        Returns:
            ParamsAndRange: The parameters and range.
        """
        raise NotImplementedError("Subclasses must implement this method")
