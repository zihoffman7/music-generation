# music-generation

14 demo songs are located in the <code>output/demo</code> direcory.

Dependencies:
<ul>
  <li>Python 3</li>
  <li>Tensorflow 2</li>
  <li>Numpy</li>
  <li>Music21</li>
</ul>

The weights for different epochs are in <i>weights</i> directory.  In order to change it, change the epoch number in <code>train_model.py</code>.  If the weight path doesn't exist, the model will retrain.

This repository only contains 2 training songs.  To add more, the songs need to be converted to ABC and placed in the <code>data/abc</code> directory.  MIDI files can be converted to ABC by placing them in <code>data/midi</code> and by running <code>mid_to_abc(path)</code> in <code>convert.py</code>.

In order to generate, run <code>predict.py</code>.

In order to change the key of the output (default is C), add an extra parameter to the last line of <code>predict.py</code> (i.e. to transpose to A-flat: <code>generate_melody(GEN_SIZE, "A-")</code>).

To change the length of the output melody, modify <code>GEN_SIZE</code> in <code>predict.py</code>.

The output file will be placed in the <code>output</code> directory.
