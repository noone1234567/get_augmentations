from abc import ABC, abstractmethod

class VideoProcessor(ABC):        
    def __init__(self):
        self.filter = None
        self.params = {}
        
    @abstractmethod
    def set_params(self, filter, params):
        '''
        Sets parameters for the filter.

        Args:
            filter: filter that is going to be applied to the rgb image
            params: dictionary to change the parameters.
        '''
        #self.filter = filter(**params) - sometimes won't work
        #self.params = params
        return
        
    @abstractmethod
    def get_params(self):
        '''
        Gets parameters for the filter.

        Returns:
            params: dictionary of the parameters.
        '''
        return self.params
    
    @abstractmethod
    def run_filter(self, video):
        """
        Processes the given video.

        Args:
            video (Video): The video to process.
        """
        pass

    @abstractmethod
    def run_optuna(self, **idk_yet):
        """
        runs optuna

        returns: best trial
        """
        pass
    
    @abstractmethod
    def count_psnr(self, video): #may be unnecessary
        """
        Counts the given video psnr.

        Args:
            video (Video): The video to process.
        """
        pass
        
    @abstractmethod
    def get_params_info(self, **idk_yet):
        """
        Gets the parameters information and their ranges associated with the video processing.

        Returns:
            ParamsAndRange: The parameters and range.
        """
        pass
