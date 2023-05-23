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
            training_type = "1"
        else:
            training_type = "2"
    else:
        if conf_mitigation["readout"] == "null":
            training_type = "3"
        else:
            training_type = "4"

    if conf_mitigation["step"] == "false":
        if conf_mitigation["final"] == "false":
            step_flag = "A"
        else:
            step_flag = "C"
    else:
        if conf_mitigation["final"] == "false":
            step_flag = "B"
        else:
            step_flag = "D"
    
    return training_type + step_flag
        

