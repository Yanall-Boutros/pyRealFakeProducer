# Useful for getting sense of unique chord types (~1000), max length of measures (122)
def ireal_set_add(tunes, trgt_set):
    measure_len_to_tunes = {}
    measures = set()
    for tune in tunes:
        ts = str(tune.time_signature[0])+str(tune.time_signature[1])
        measures_string = tune.measures_as_strings
        measure_len = len(measures_string)
        if measure_len in measure_len_to_tunes: measure_len_to_tunes[measure_len].append(tune)
        else: measure_len_to_tunes[measure_len] = [ tune ]
        for measure in measures_string: measures.add(measure)
        chords_string = tune.chord_string[tune.chord_string.find(ts)+2:]
        for chords in chords_string.split("|"):
            for chord in chords.split(" "):
                if chord == '': continue
                if len(chord) == 0: continue
                if any([bad_char in chord for bad_char in "{}[]*<()>TW"]): continue
                if any([bad_word in chord for bad_word in [ "edal","Q", "1st", "use", "alt", "till", "takes", "2nd", "over", "Coda", "minor", "free", "only", "chorus", "feel", "the", "eep", "in", "every", "olos", "is", "out", "on", "AABA", "ow", "time", "ops", "by", "chords", "of", "D.C.", "Miles", "or", "double","key", "Feel", "until", "CD"]]): continue
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
                #if chord[-1] == "^": continue
                if chord[0] == "f": chord=chord[1:]
                if len(chord) == 0: continue
                if chord[0] == "l": chord=chord[1:]
                if len(chord) == 0: continue
                if 'p' in chord[0]: chord = chord[1:]
                if len(chord) == 0: continue
                if chord == "r": continue
                if chord == "x": continue
                if chord == "S": continue
                if chord[0].isnumeric(): chord = chord[1:]
                if len(chord) == 0: continue
                trgt_set.add(chord)
    return measure_len_to_tunes, measures

