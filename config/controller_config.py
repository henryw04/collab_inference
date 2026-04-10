class ControllerConfig:
    # ZMQ Topology
    APISOC_BIND = "tcp://127.0.0.1:6665" # Change the port if needed, no need to change the IP
    PUSH_BIND = "tcp://127.0.0.1:6666"  # Change the port if needed, no need to change the IP
    PULL_CONNECT = "tcp://127.0.0.1:6669" # IP of last node + Port
    
