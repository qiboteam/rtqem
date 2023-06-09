def get_training_type(conf_dictionary):
    """
    Defines the training type according to the experiment configuration.
    
    Returns: a string for customizing the data saving.
    """

    conf_mitigation = conf_dictionary["mitigation"]

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
    
    return training_type + step_flag
        

