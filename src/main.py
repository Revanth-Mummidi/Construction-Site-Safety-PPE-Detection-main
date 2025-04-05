import sys
import os
from src.interfaces.webcam import main as webcam_main
from src.interfaces.batch_processor import main as batch_main

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "webcam":
            webcam_main()
        elif sys.argv[1] == "batch":
            batch_main()
        else:
            print("Usage: python -m src.main [webcam|batch]")
    else:
        print("Please specify mode: webcam or batch")
        print("Example: python -m src.main webcam")

if __name__ == "__main__":
    main()