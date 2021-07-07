from music21 import converter, interval, pitch
import os

def mid_to_abc(dir):
    for file in os.listdir(dir):
        print(file)
        # Ensure the file is a midi file
        if not file.split(".")[-1].lower() == "mid" and not file.split(".")[-1].lower() == "midi":
            continue
        path = f"{dir}/{file}"
        print("loading")
        fname = path.split("/")[-1].split(".")[0]
        # Load midi file
        s = converter.parse(path).makeNotation()
        print("loaded")
        # Attempt to transpose the file to C major
        s = s.transpose(interval.Interval(s.analyze("key").tonic, pitch.Pitch("C")))
        # Convert music21 stream object to MusicXML
        s.write("xml", fp=f"{fname}.xml")
        # Convert MusicXML to abc using xml2abc
        print("converting")
        os.system(f"python xml2abc.py {fname}.xml -o abc")
        # Remove the temporary MusicXML file
        os.remove(f"{fname}.xml")
    print("Done!")
mid_to_abc("midi")
