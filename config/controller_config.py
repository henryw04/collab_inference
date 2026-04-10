class ControllerConfig:
    
    APISOC_BIND = "tcp://127.0.0.1:6665"  # Change the port if needed, no need to change the IP unless the API is on a different machine
    PUSH_BIND = "tcp://*:6666"  # localhost / 127.0.0.1 / 0.0.0.0 / * + port, make sure the port is open and not used by other applications
    PULL_CONNECT = "tcp://127.0.0.1:6669"  # Terminal node IP + Port
