class Lambda:
    def __init__(self, 
                 on_init=lambda trainer: None, 
                 on_training_step_start=lambda trainer: None, 
                 on_training_step_end=lambda trainer: None, 
                 on_validation_step_start=lambda trainer: None, 
                 on_validation_step_end=lambda trainer: None, 
                 on_epoch_start=lambda trainer: None, 
                 on_epoch_end=lambda trainer: None, 
                 on_validation_start=lambda trainer: None, 
                 on_validation_end=lambda trainer: None, 
                 on_training_start=lambda trainer: None, 
                 on_training_end=lambda trainer: None,  
                 on_training_stop=lambda trainer: None, 
                 on_checkpoint_save=lambda trainer: None, 
                 on_exception=lambda exception, trainer: None,):

        self.on_init = on_init
        self.on_training_step_start = on_training_step_start
        self.on_training_step_end = on_training_step_end
        self.on_validation_step_start = on_validation_step_start
        self.on_validation_step_end = on_validation_step_end
        self.on_epoch_start = on_epoch_start
        self.on_epoch_end = on_epoch_end
        self.on_validation_start = on_validation_start
        self.on_validation_end = on_validation_end
        self.on_training_start = on_training_start
        self.on_training_end = on_training_end
        self.on_training_stop = on_training_stop
        self.on_checkpoint_save = on_checkpoint_save
        self.on_exception = on_exception