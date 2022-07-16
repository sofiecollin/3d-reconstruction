import subprocess
import os

FNULL = open(os.devnull, 'w')    #use this if you want to suppress output to stdout from the subprocess
inputPath = "../dataset/point_cloud/horse.npts"
outputPath = "../dataset/point_cloud/horse.ply"
PoissonRecon = "../ext/kazhdan/PoissonRecon.exe"
SSDRecon = "../ext/kazhdan/SSDRecon.exe"


args =  PoissonRecon + " --in " + inputPath + " --out " + outputPath + " --depth 10"


subprocess.call(args, stdout=FNULL, stderr=FNULL, shell=False)