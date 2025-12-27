import configparser
import numpy as np
import os
import random

pathForInis="/data/dust/user/sabebert/ConfigFiles/kvsFiles/mdPos/Training3Kvs/"

minPos=[0.05,0.05,0.05]
maxPos=[2.45,0.36,0.36]
deltas =[0.05, 0.05, 0.05]
numberPos=None
splitDomain="full"
kvsFile='kvstest_training3.ini'
def getDomain(minPos=[0.05,0.05,0.05], maxPos=[2.45,0.36,0.36], splitDomain="full"):
    if splitDomain == "half":
        maxPos[1] = (maxPos[1] - minPos[1])/2
    elif splitDomain == "random":
        number = [float(random.randint(minPos[i], maxPos[i])) for i in range(3)]
        cord = random.randint(0, 2)
        maxPos[cord] = number[cord]
    return minPos, maxPos

def createMdPosArray(minPos=[0.05,0.05,0.05], maxPos=[2.45,0.36,0.36], numberPos=None, deltas=[None,None,None], splitDomain="full"):
    mdPosArray = []

    if splitDomain != "full":
        minPos, maxPos = getDomain(minPos, maxPos, splitDomain)
    if numberPos is not None and deltas == [None,None,None]:
        numberPos = numberPos - 2
        deltax, deltay, deltaz = np.subtract(maxPos, minPos) / numberPos
    else:
        deltax, deltay, deltaz = deltas
    
    for xpos in np.arange(minPos[0], maxPos[0], deltax):
        for ypos in np.arange(minPos[1], maxPos[1], deltay):
            for zpos in np.arange(minPos[2], maxPos[2], deltaz):
                newPos = [float(str(xpos)), float(str(ypos)), float(str(zpos))]
                mdPosArray.append(newPos)

    mdPosValues = {i: mdPos for i, mdPos in enumerate(mdPosArray)}
    np.save("mdPosValues.npy", mdPosValues)
    return mdPosArray


#main
def main():
    #Get MD position array
    mdPosArray = createMdPosArray(minPos, maxPos, numberPos, deltas, splitDomain)
    #get config from original/template kvs file
    config = configparser.ConfigParser()
    config.read(pathForInis + kvsFile)

    #Change domain pos/init, coupling times and write adjust config to new config file
    for posIndex, mdPos in enumerate(mdPosArray):
        configI = config
        configI['domain']['md-pos'] = str(mdPos)
        configI['macroscopic-solver']['init-timesteps'] = str(24000)
        configI['microscopic-solver']['equilibration-steps'] = str(10000)
        configI['coupling']['couplingCycles'] = str(1400)

        with open(pathForInis + 'kvstest_' + str(posIndex) + '.ini', 'w') as configfile:
            config.write(configfile)

main()