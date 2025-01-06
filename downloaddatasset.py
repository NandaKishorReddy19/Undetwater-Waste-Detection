

from roboflow import Roboflow
rf = Roboflow(api_key="h13Y6T6HB14vupSXkju7")
project = rf.workspace("hi-gcezf").project("under-water-waste-detection")
version = project.version(7)
dataset = version.download("yolov11")
                