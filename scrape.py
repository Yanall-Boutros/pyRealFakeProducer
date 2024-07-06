from pyRealParser import Tune
import pdb
import sys
def ireal_set_add(tunes, trgt_set):
    for tune in tunes:
        ts = str(tune.time_signature[0])+str(tune.time_signature[1])
        chords_string = tune.chord_string[tune.chord_string.find(ts)+2:]
        for chords in chords_string.split("|"):
            for chord in chords.split(" "):
                if chord == '': continue
                if len(chord) == 0: continue
                if any([bad_char in chord for bad_char in "{}[]*<a()>T"]): continue
                if any([bad_word in chord for bad_word in [ "edal","Q", "1st", "use", "alt", "till", "takes", "2nd", "over", "Coda", "minor", "free", "only", "chorus", "feel", "the", "eep", "in", "every", "olos", "is", "out", "on", "AABA", "ow", "time", "ops", "by", "chords", "of", "D.C.", "Miles", "or", "double","key"]]): continue
                if chord[0] == "N": chord = chord[chord.find("l")+1:]
                if chord[0] == "n": continue
                if chord[0] == "s": chord=chord[1:]
                if len(chord) == 0: continue
                if 'p' in chord[0]: chord =chord[1:]
                if len(chord) == 0: continue
                if "pps" in chord: chord=chord[3:]
                if len(chord) == 0: continue
                if "ps" in chord: chord=chord[2:]
                if len(chord) == 0: continue
                if "l" in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if "N" in chord: chord = chord[2:]
                if len(chord) == 0: continue
                if chord == "n": continue
                if 'p' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if 'n' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if 's' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if 'f' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if 'U' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if chord[-1] == "^": continue
                if chord[0] == "f": chord=chord[1:]
                if len(chord) == 0: continue
                if chord[0] == "l": chord=chord[1:]
                if len(chord) == 0: continue
                if 'p' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if chord == "r": continue
                if chord == "S": continue
                trgt_set.add(chord)
def main():
    chord_types = set()
    with open("./ireal_url", "r") as f: my_tune = Tune.parse_ireal_url(f.read())
    ireal_set_add(my_tune, chord_types)
    pdb.set_trace()

if __name__ == "__main__": main()
