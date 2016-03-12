import sys, os
#import ipdb
import numpy as np
import copy
import numpy as np
import utils.pascal3d as p3d
import scipy.io as sio
import math
import time

def aa2quat(axis, angle):
    q0 = math.cos(angle/2)
    q1 = math.sin(angle/2)*axis[0]
    q2 = math.sin(angle/2)*axis[1]
    q3 = math.sin(angle/2)*axis[2]
    return (q0, q1, q2, q3)

def quatProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q0 = a * e - b * f - c * g - d * h
    q1 = a * f + b * e + c * h - d * g
    q2 = a * g - b * h + c * e + d * f
    q3 = a * h + b * g - c * f + d * e
    return (q0, q1, q2, q3)

def quatConjugate(q):
    (q0, q1, q2, q3) = q
    return (q0, -q1, -q2, -q3)

def eulersToQuat(az,el,cy):
    axisX = np.array([1,0,0],dtype=np.float32)
    axisY = np.array([0,1,0],dtype=np.float32)
    axisZ = np.array([0,0,1],dtype=np.float32)
    return quatProduct(aa2quat(axisZ,float(cy)),(quatProduct(aa2quat(axisX,float(el)), aa2quat(axisY,float(az)))))

class PoseBenchmark():

    def __init__(self,azimuthOnly=False,test=1,useOccluded=0,classes = ['car']):
        self.test = test
        self.azimuthOnly = azimuthOnly
        self.useOccluded = useOccluded
        self.classes = classes
        self._initModelInfo()

    def giveTestInstances(self, objClass):
        nClasses = len(self.classes)
        cInd = [ix for ix in range(nClasses) if self.classes[ix] == objClass]
        assert cInd is not [], "Unkown object class"
        cInd = cInd[0]
        return copy.deepcopy(self.instanceNames[cInd]),copy.deepcopy(self.instanceBoxes[cInd])

    def giveTestPoses(self, objClass):
        nClasses = len(self.classes)
        cInd = [ix for ix in range(nClasses) if self.classes[ix] == objClass]
        assert cInd is not [], "Unkown object class"
        cInd = cInd[0]
        return copy.deepcopy(self.instancePoses[cInd])

    def evaluatePredictions(self, objClass, predictions):
        nClasses = len(self.classes)
        cInd = [ix for ix in range(nClasses) if self.classes[ix] == objClass]
        assert cInd is not [], "Unkown object class"
        cInd = cInd[0]
        gtPoses = copy.deepcopy(self.instancePoses[cInd])
        assert len(predictions) == len(gtPoses), "Different number of instances than expected"
        errors = np.zeros(len(predictions))
        for ix in range(len(predictions)):
            az, el, cy = gtPoses[ix]
            gtQuat = eulersToQuat(az,0,0) if self.azimuthOnly else eulersToQuat(az, el, cy)

            az, el, cy = predictions[ix]
            predQuat = eulersToQuat(az,0,0) if self.azimuthOnly else eulersToQuat(az, el, cy)

            relQuat = quatProduct(gtQuat,quatConjugate(predQuat))
            relCtheta = relQuat[0]

            ## check against numerical errors
            if relCtheta > 1:
                relCtheta = 1
            if relCtheta < -1:
                relCtheta = -1
            #print ix
            #print gtQuat, predQuat, relQuat
            errors[ix] = min(2*math.acos(relCtheta), 2*np.pi - 2*math.acos(relCtheta))
        return errors

    def _initModelInfo(self):
        self.instanceNames, self.instancePoses, self.instanceBoxes = [],[],[]
        for cName in self.classes:
            cInstanceNames, cInstanceBoxes, cInstancePoses = p3d.loadAnnos(cName,self.test,useOccluded=self.useOccluded)
            for ix in range(len(cInstancePoses)):
                [cy,el,az] = cInstancePoses[ix]
                cInstancePoses[ix] = [az,el,cy]
            self.instanceNames.append(cInstanceNames)
            self.instancePoses.append(cInstancePoses)
            self.instanceBoxes.append(cInstanceBoxes)

