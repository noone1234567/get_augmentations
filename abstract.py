from abc import ABC, abstractmethod

class VideoProcessor(ABC):
    
    @abstractmethod
    def set_params(self, **params):
        '''
        Sets parameters for the filter.

        Args:
            params: dictionary to change the parameters.
        '''
        pass
        
    @abstractmethod
    def get_params(self):
        '''
        Gets parameters for the filter.

        Returns:
            params: dictionary of the parameters.
        '''
        pass
    
    @abstractmethod
    def run_filter(self, video):
        """
        Processes the given video.

        Args:
            video (Video): The video to process.
        """
        pass

    @abstractmethod
    def count_psnr(self, video):
        """
        Counts the given video psnr.

        Args:
            video (Video): The video to process.
        """
        pass
        
    @abstractmethod
    def get_params_info(self):
        """
        Gets the parameters information and their ranges associated with the video processing.

        Returns:
            ParamsAndRange: The parameters and range.
        """
        pass
