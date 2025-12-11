import sys
import mne

def load_egg()->None:
    raw = mne.io.read_raw_edf("S001R01.edf", preload=True)
    raw.plot(block=True, duration=10.0, title='raw data')
    raw.filter(.1, 30) # rm hz <1 and hz > 30   
    raw.plot(block=True, duration=10.0, title='raw data')
    # print(raw)
    # raw.crop(tmax=60)


def main()->None:
    if (len(sys.argv) > 2):
        print("Usage: python main.py [ <train> | <predict> ]", file=sys.stderr)
        return
    load_egg()



if __name__ == "__main__":
    main()