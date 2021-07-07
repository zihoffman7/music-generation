from music21 import converter, interval, pitch
import os
    
def mid_to_abc(dir):
    for file in os.listdir(dir):
        path = f"{dir}/{file}"
        s = converter.parse(path).makeNotation()
        s = s.transpose(interval.Interval(s.analyze("key").tonic, pitch.Pitch("C")))
        s.write("xml", fp=f"{path.split('/')[-1].split('.')[0]}.xml")
        os.system(f"python xml2abc.py {path.split('/')[-1].split('.')[0]}.xml -o abc")
        os.remove(f"{path.split('/')[-1].split('.')[0]}.xml")
        s = converter.parse(f"abc/{path.split('/')[-1].split('.')[0]}.abc")
