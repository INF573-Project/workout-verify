#---------------------------------------------------------------------------
#                            Squat Rules
#---------------------------------------------------------------------------
squat_rules = {
  "knee_right": {
    "min": 90,
    "max": 150
  },
  "knee_left": {
    "min": 90,
    "max": 150
  },
  "spine_middle": {
    "max": 20
  },
  "hip_left": {
    "max": 50,
    "min": 0
  },
  "hip_right": {
    "max": 50,
    "min": 0
  },
  "spine_hinge": {
    "max": 45
  }
}

squat_cyclic_pair = ("knee_right", "knee_left")

#---------------------------------------------------------------------------
#                           Pull-up Rules
#---------------------------------------------------------------------------
pullup_rules = {
  "elbow_right": {
    "min": 55,
    "max": 110
  },
  "elbow_left": {
    "min": 55,
    "max": 110
  }
}

pullup_cyclic_pair = ("elbow_right", "elbow_left")
