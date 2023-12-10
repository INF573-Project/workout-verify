#---------------------------------------------------------------------------
#                            Squat Rules
#---------------------------------------------------------------------------
squat_rules = {
  "knee_right": {
    "cyclic_joint": True,
    "type": "thresholding",
    "dependency": None,
    "args": {
      "min": 90,
      "max": 150
    }
  },
  "knee_left": {
    "cyclic_joint": True,
    "type": "thresholding",
    "dependency": None,
    "args": {
      "min": 90,
      "max": 150
    }
  },
  "spine_middle": {
    "cyclic_joint": False,
    "type": "delta_thresholding",
    "dependency": None,
    "args": {
      "min": -10,
      "max": 10
    }
  },
  "hip_left": {
    "cyclic_joint": False,
    "type": "thresholding",
    "dependency": None,
    "args": {
      "min": 0,
      "max": 40
    }
  },
  "hip_right": {
    "cyclic_joint": False,
    "type": "thresholding",
    "dependency": None,
    "args": {
      "min": 0,
      "max": 40
    }
  },
  "spine_hinge": {
    "cyclic_joint": False,
    "type": "delta_thresholding",
    "dependency": None,
    "args": {
      "min": -10,
      "max": 10
    }
  }
}

#---------------------------------------------------------------------------
#                           Pull-up Rules
#---------------------------------------------------------------------------
pullup_rules = {
  "elbow_right": {
    "cyclic_joint": True,
    "type": "thresholding",
    "dependency": None,
    "args": {
      "min": 55,
      "max": 110
    }
  },
  "elbow_left": {
    "cyclic_joint": True,
    "type": "thresholding",
    "dependency": None,
    "args": {
      "min": 55,
      "max": 110
    }
  },
  "knee_right": {
    "cyclic_joint": False,
    "type": "delta_thresholding",
    "dependency": None,
    "args": {
      "min": -10,
      "max": 10
    }
  },
  "knee_left": {
    "cyclic_joint": False,
    "type": "delta_thresholding",
    "dependency": None,
    "args": {
      "min": -10,
      "max": 10
    }
  }
}
