from datetime import timedelta, datetime
from typing import Tuple


class Timer:
    def __init__(self, time_format:str="{hours}:{minutes}:{seconds}"):
        self.start = datetime.now()
        self.time_format = time_format
        
        self.elapsed_time = timedelta(seconds=0)
        self.remain_time = None
    

    @staticmethod
    def get_time_from_timedelta(self, delta:timedelta) -> dict:
        time = {"days": delta.days}
        time["hours"], rem = divmod(delta.seconds, 3600)
        time["minutes"], time["seconds"] = divmod(rem, 60)

        return time
    
    
    def state_dict(self) -> dict:
        state = {
            "start": str(datetime.now()),
            "time_format": self.time_format,
            "elapsed_time": self.elapsed_time.total_seconds(),
            "remain_time": None if self.remain_time is None else self.remain_time.total_seconds(),
        }
        
        return state
    
    def load_state_dict(self, state_dict:dict) -> "Timer":
        self.start = datetime.fromisoformat(state_dict["start"])
        self.time_format = state_dict["time_format"]
        self.elapsed_time = timedelta(seconds=state_dict["elapsed_time"])
        self.remain_time = timedelta(seconds=state_dict["remain_time"])
        
        return self
    

    @staticmethod
    def format_time(self, time:timedelta, time_format:str="{hours}:{minutes}:{seconds}") -> str:
        """
        Formats `timedelta` to user's time format.
        """
        time = Timer.get_time_from_timedelta(time)
        return time_format.format(**time)
    
    @property
    def elapsed(self) -> str:
        """
        Returns elapsed time in user's time format.
        """
        
        return Timer.format_time(self.elapsed_time, time_format=self.time_format)
        
    @property
    def remain(self) -> str:
        """
        Returns remain time in user's time format.
        """
        
        return Timer.format_time(self.remain_time, time_format=self.time_format)
    
    def __call__(self, fraction:float) -> Tuple[str, str]:        
        """
        Inputs:
            fraction: float - fraction of total steps.
        
        Outputs:
            elapsed: str - elapsed time.
            remain: str - remain time.
        """
        
        now = datetime.now()
        self.elapsed_time = now - self.start
        elapsed_seconds = self.elapsed_time.total_seconds()        
        total_seconds = timedelta(seconds=round(elapsed_seconds / fraction))
        self.remain_time = abs(total_seconds - self.elapsed_time)
        
        return self.elapsed, self.remain
        
        
    def __str__(self) -> str:
        return f"Timer(start={self.start}, remain={self.remain}, elapsed={self.elapsed})"
    
    
    __repr__ = __str__