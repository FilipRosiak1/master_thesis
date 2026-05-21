from gui.widgets.mainPage import MainPage
import sys, argparse

def main():
    parser = argparse.ArgumentParser(description="Framsticks GUI for library/server")
    parser.add_argument("-l", "--library", help="<path>: load Framsticks library (dll/so/dylib) from the provided path")
    parser.add_argument("-s", "--server", help="<address:port>: connect to the running Framsticks server at the provided address, e.g. localhost:9009")

    args = parser.parse_args()

    if args.library and args.server:
        print("Do not provide both the library path and the network server address.")
        sys.exit(2)

    app = MainPage(None, args.server, args.library)
    app.mainloop()

if __name__ == '__main__':
    main()
