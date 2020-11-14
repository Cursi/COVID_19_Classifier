import sys

if __name__ == "__main__":
    print("Hello from python run with arguments:")
    print(sys.argv)

    f = open(sys.argv[1], "r")
    print(f.read())