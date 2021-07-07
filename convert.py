from music21 import converter, interval, pitch
import os

def mid_to_abc(dir):
    for file in os.listdir(dir):
        path = f"{dir}/{file}"
        fname = path.split("/")[-1].split(".")[0]
        s = converter.parse(path).makeNotation()
        s = s.transpose(interval.Interval(s.analyze("key").tonic, pitch.Pitch("C")))
        s.write("xml", fp=f"{fname}.xml")
        os.system(f"python xml2abc.py {fname}.xml -o abc")
        os.remove(f"{fname}.xml")
    print("Done!")
