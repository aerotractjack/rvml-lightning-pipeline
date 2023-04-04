from yaml import load
try:
    from yaml import SafeLoader as Loader
except ImportError:
    from yaml import Loader

def yaml2dict(path):
    with open(path, "r") as fp:
        data = load(fp, Loader=Loader)
    return data

if __name__ == "__main__":
    import sys
    out = yaml2dict(sys.argv[1])
    print(out)