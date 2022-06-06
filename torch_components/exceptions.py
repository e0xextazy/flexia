class LibraryException(Exception):
   def __init__(self,  library):
        self.message = f"Make sure that `{library}` is installed or supported."
        super().__init__(self.message)