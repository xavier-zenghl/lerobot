class Joint_Dim:

    CHASSIS = 3
    TORSO = 4           
    LEFT_ARM = 7           
    LEFT_GRIPPER = 1
    RIGHT_ARM = 7
    RIGHT_GRIPPER = 1
    HEAD = ["yaw", "pitch"]


class Cartesian_Dim:
    TORSO = 9           
    LEFT_ARM = 9           
    LEFT_GRIPPER = 1
    RIGHT_ARM = 9
    RIGHT_GRIPPER = 1
    HEAD = ["yaw", "pitch"]
    CHASSIS = 3

DATASET_FEATURES = {
    "droid": {
        "joint_state_names": [f"chassis_padding_{i}" for i in range(Joint_Dim.CHASSIS)] 
                            + [f"torso_padding_{i}" for i in range(Joint_Dim.TORSO)] 
                            + [f"left_arm_state_{i}" for i in range(Joint_Dim.LEFT_ARM)] 
                            + [f"left_gripper_state_{i}" for i in range(Joint_Dim.LEFT_GRIPPER)] 
                            + [f"right_arm_padding_{i}" for i in range(Joint_Dim.RIGHT_ARM)] 
                            + [f"right_gripper_padding_{i}" for i in range(Joint_Dim.RIGHT_GRIPPER)] 
                            + [f"head_padding_{i}" for i in Joint_Dim.HEAD],
        "cartesian_state_names": [f"torso_padding_{i}" for i in range(Cartesian_Dim.TORSO)] 
                            + [f"left_arm_state_{i}" for i in range(Cartesian_Dim.LEFT_ARM)]
                            + [f"left_gripper_state_{i}" for i in range(Cartesian_Dim.LEFT_GRIPPER)]
                            + [f"right_arm_padding_{i}" for i in range(Cartesian_Dim.RIGHT_ARM)] 
                            + [f"right_gripper_padding_{i}" for i in range(Cartesian_Dim.RIGHT_GRIPPER)]
                            + [f"head_padding_{i}" for i in Cartesian_Dim.HEAD]
                            + [f"chassis_padding_{i}" for i in range(Cartesian_Dim.CHASSIS)],
        "joint_action_names": [f"chassis_padding_{i}" for i in range(Joint_Dim.CHASSIS)] 
                            + [f"torso_padding_{i}" for i in range(Joint_Dim.TORSO)] 
                            + [f"left_arm_action_{i}" for i in range(Joint_Dim.LEFT_ARM)] 
                            + [f"left_gripper_action_{i}" for i in range(Joint_Dim.LEFT_GRIPPER)] 
                            + [f"right_arm_padding_{i}" for i in range(Joint_Dim.RIGHT_ARM)] 
                            + [f"right_gripper_padding_{i}" for i in range(Joint_Dim.RIGHT_GRIPPER)] 
                            + [f"head_padding_{i}" for i in Joint_Dim.HEAD],
        "cartesian_action_names": [f"torso_padding_{i}" for i in range(Cartesian_Dim.TORSO)] 
                            + [f"left_arm_action_{i}" for i in range(Cartesian_Dim.LEFT_ARM)]
                            + [f"left_gripper_action_{i}" for i in range(Cartesian_Dim.LEFT_GRIPPER)]
                            + [f"right_arm_padding_{i}" for i in range(Cartesian_Dim.RIGHT_ARM)] 
                            + [f"right_gripper_padding_{i}" for i in range(Cartesian_Dim.RIGHT_GRIPPER)]
                            + [f"head_padding_{i}" for i in Cartesian_Dim.HEAD]
                            + [f"chassis_padding_{i}" for i in range(Cartesian_Dim.CHASSIS)],
        "image_obs_keys": {
            "exterior_1.rgb": "exterior_image_1_left",
            "exterior_2.rgb": "exterior_image_2_left",
            "wrist.rgb": "wrist_image_left",
        },
        "task_extended_indexs": ["task_1", "task_2", "task_3"],
    },
    "astribot": {
        "joints_dict.joints_position_state": [f"chassis_state_{i}" for i in range(Joint_Dim.CHASSIS)] 
                            + [f"torso_state_{i}" for i in range(Joint_Dim.TORSO)] 
                            + [f"left_arm_state_{i}" for i in range(Joint_Dim.LEFT_ARM)] 
                            + [f"left_gripper_state_{i}" for i in range(Joint_Dim.LEFT_GRIPPER)] 
                            + [f"right_arm_state_{i}" for i in range(Joint_Dim.RIGHT_ARM)] 
                            + [f"right_gripper_state_{i}" for i in range(Joint_Dim.RIGHT_GRIPPER)] 
                            + [f"head_state_{i}" for i in Joint_Dim.HEAD],
        "cartesian_so3_dict.cartesian_pose_state": [f"torso_state_{i}" for i in range(Cartesian_Dim.TORSO)] 
                            + [f"left_arm_state_{i}" for i in range(Cartesian_Dim.LEFT_ARM)]
                            + [f"left_gripper_state_{i}" for i in range(Cartesian_Dim.LEFT_GRIPPER)]
                            + [f"right_arm_state_{i}" for i in range(Cartesian_Dim.RIGHT_ARM)] 
                            + [f"right_gripper_state_{i}" for i in range(Cartesian_Dim.RIGHT_GRIPPER)]
                            + [f"head_state_{i}" for i in Cartesian_Dim.HEAD]
                            + [f"chassis_state_{i}" for i in range(Cartesian_Dim.CHASSIS)],
        "joints_dict.joints_position_command": [f"chassis_action_{i}" for i in range(Joint_Dim.CHASSIS)] 
                            + [f"torso_action_{i}" for i in range(Joint_Dim.TORSO)] 
                            + [f"left_arm_action_{i}" for i in range(Joint_Dim.LEFT_ARM)] 
                            + [f"left_gripper_action_{i}" for i in range(Joint_Dim.LEFT_GRIPPER)] 
                            + [f"right_arm_action_{i}" for i in range(Joint_Dim.RIGHT_ARM)] 
                            + [f"right_gripper_action_{i}" for i in range(Joint_Dim.RIGHT_GRIPPER)] 
                            + [f"head_action_{i}" for i in Joint_Dim.HEAD],
        "cartesian_so3_dict.cartesian_pose_command": [f"torso_action_{i}" for i in range(Cartesian_Dim.TORSO)] 
                            + [f"left_arm_action_{i}" for i in range(Cartesian_Dim.LEFT_ARM)]
                            + [f"left_gripper_action_{i}" for i in range(Cartesian_Dim.LEFT_GRIPPER)]
                            + [f"right_arm_action_{i}" for i in range(Cartesian_Dim.RIGHT_ARM)] 
                            + [f"right_gripper_action_{i}" for i in range(Cartesian_Dim.RIGHT_GRIPPER)]
                            + [f"head_action_{i}" for i in Cartesian_Dim.HEAD]
                            + [f"chassis_action_{i}" for i in range(Cartesian_Dim.CHASSIS)],
    }         
}