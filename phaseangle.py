import numpy as np
class PhaseAngleObj:
    def __init__(self, grainsize, incident_angle, emission_angle, data):
        self.grainsize = int(grainsize)
        self.incident_angle = np.deg2rad(float(incident_angle))
        self.emission_angle = np.deg2rad(float(emission_angle))
        self.data = data
