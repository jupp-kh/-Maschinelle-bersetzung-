import sys
import time
import threading
import os
import batches
import encoder
import dictionary
import argparse

# TODO python script for running the differnt method.

commands = {
    "create_batches": batches.create_batches,
    "get_op_sequences": encoder.get_op_sequences,
    "subword_split": encoder.subword_split,
}


def main():
    args = sys.argv
    while len(args) < commands[args[1]].__code__.co_argcount:
        print(commands[args[1]].__code__.co_varnames)
        args = input().split()
    commands[args[1]](
        *[int(args[i]) if args[i].isdigit() else args[i] for i in range(2, len(args))]
    )


if __name__ == "__main__":
    main()
