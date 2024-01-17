#!/usr/bin/env python3

import rospy
import yaml

from module.nmpc_controller import NMPC

def load_params(config_file):
    with open(config_file, 'r') as file:
        params = yaml.safe_load(file)
    return params

def main():
    rospy.init_node('nmpc_controller')

    config = rospy.get_param("~config_file", "src/terrasentia_navigation/controllers/nmpc/config/nmpc_params.yaml")
    params = load_params(config)

    control = NMPC(**params)

    rospy.spin()

if __name__ == '__main__':
    main()