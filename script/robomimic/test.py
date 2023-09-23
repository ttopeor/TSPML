joint_values = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.7850000262260437,
    "panda_joint3": 0.0,
    "panda_joint4": -2.3559999465942383,
    "panda_joint5": 0.0,
    "panda_joint6": 1.57079632679,
    "panda_joint7": 0.7850000262260437,
}

joint_scale = {
    "panda_joint1": 2.7,
    "panda_joint2": 1.5,
    "panda_joint3": 0.001,
    "panda_joint4": 0.68,
    "panda_joint5": 2.8,
    "panda_joint6": 1.5,
    "panda_joint7": 2.2,
}

bounds = {}
soft_bounds = {}

for joint, value in joint_values.items():
    scale = joint_scale[joint]
    upper = value + scale
    lower = value - scale
    
    bounds[joint] = {
        "upper": format(upper, '.4f'),
        "lower": format(lower, '.4f')
    }
    
    soft_bounds[joint] = {
        "soft_upper": format(upper - 0.05, '.4f'),
        "soft_lower": format(lower + 0.05, '.4f')
    }

print("Bounds:")
for joint, bound in bounds.items():
    print(f"{joint}: Upper - {bound['upper']}, Lower - {bound['lower']}")

print("\nSoft Bounds:")
for joint, soft_bound in soft_bounds.items():
    print(f"{joint}: Soft Upper - {soft_bound['soft_upper']}, Soft Lower - {soft_bound['soft_lower']}")
