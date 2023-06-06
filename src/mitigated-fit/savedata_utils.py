def get_training_type(conf_dictionary):
    """
    Defines the training type according to the experiment configuration.
    
    Retuirns: string composed of one number and one letter. The meaning follows:
        (number): an integer between 1 and 4.
            1 -> no real time mitigation, no readout mitigation;
            2 -> no real time mitigation, readout mitigation;
            3 -> real time mitigation, no readout mitigation;
            4 -> real time mitigation, readout mitigation.
        (letter) A, B, C or D:
            A -> step false, final false;
            B -> step true, final false;
            C -> step false, final true;
            D -> step true, final true.
    """

    conf_mitigation = conf_dictionary["mitigation"]

    if conf_mitigation["method"] == None:
        if conf_mitigation["readout"] == "null":
            training_type = "unmitigated_"
        else:
            training_type = "readout_mitigation_"
    else:
        if conf_mitigation["readout"] == "null":
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
        

