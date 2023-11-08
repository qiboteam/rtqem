def get_training_type(conf_mitigation, conf_noise):
    """
    Defines the training type according to the experiment configuration.
    
    Returns: a string for customizing the data saving.
    """
    if conf_noise:
        if conf_mitigation["method"] is None:
            if conf_mitigation["readout"] is None:
                training_type = "unmitigated_"
            else:
                training_type = "readout_mitigation_"
        else:
            if conf_mitigation["readout"] is None:
                training_type = "realtime_mitigation_"
            else:
                training_type = "full_mitigation_"

        if conf_mitigation["step"] is False:
            if conf_mitigation["final"] is False:
                step_flag = "step_no_final_no"
            else:
                step_flag = "step_no_final_yes"
        else:
            if conf_mitigation["final"] is False:
                step_flag = "step_yes_final_no"
            else:
                step_flag = "step_yes_final_yes"
    else:
        training_type = "noiseless"
        step_flag = ""
    
    return training_type + step_flag
        

