

# 对于刚开始进行深蹲练习的人设置的阈值
def get_thresholds_beginner():

    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  32),
                            'TRANS'  : (35, 65),
                            'PASS'   : (70, 95)
                           }    

    thresholds = {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,

                    'HIP_THRESH'   : [10, 50],
                    'ANKLE_THRESH' : 45,
                    'KNEE_THRESH'  : [50, 70, 95],

                    'OFFSET_THRESH'    : 35.0,
                    'INACTIVE_THRESH'  : 15.0,

                    'CNT_FRAME_THRESH' : 50
                            
                }

    return thresholds



# 对于进阶者设置的阈值
def get_thresholds_pro():

    _ANGLE_HIP_KNEE_VERT = {
                            'NORMAL' : (0,  32),
                            'TRANS'  : (35, 65),
                            'PASS'   : (80, 95)
                           }    

        
    thresholds = {
                    'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,

                    'HIP_THRESH'   : [15, 50],
                    'ANKLE_THRESH' : 30,
                    'KNEE_THRESH'  : [50, 80, 95],

                    'OFFSET_THRESH'    : 35.0,
                    'INACTIVE_THRESH'  : 15.0,

                    'CNT_FRAME_THRESH' : 50
                            
                 }
                 
    return thresholds