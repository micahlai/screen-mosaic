"""Entry point so `python mosiac` launches the calibration + mapping host."""

try:                       # `python -m mosiac` (package context)
    from .server import main
except ImportError:        # `python mosiac` (directory on sys.path)
    from server import main

if __name__ == "__main__":
    main()
