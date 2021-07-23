# music-generation

Dependencies:
<ul>
  <li>Python 3</li>
  <li>Tensorflow 2</li>
  <li>Numpy</li>
  <li>Music21</li>
</ul>

The weights for different epochs are in <i>weights</i> directory.  In order to change it, change the epoch number in <code>train_model.py</code>.  If the weight path doesn't exist, the model will retrain.

In order to change the key of the output (default is C), add an extra parameter to the last line of <code>predict.py</code>.

For example, to transpose to A-flat: <code>generate_melody(GEN_SIZE, "A-")</code>

This repository only contains 2 training songs.  To add more, the songs need to be converted to ABC and placed in the <code>data/abc</code> directory.  MIDI files can be converted to ABC by placing them in <code>data/midi</code> and by running <code>mid_to_abc(path)</code> in <code>convert.py</code>.
