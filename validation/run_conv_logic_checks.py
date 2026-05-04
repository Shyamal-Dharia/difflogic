import argparse
import subprocess
import sys


def run(cmd):
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--skip-timing', action='store_true')
    args = parser.parse_args()

    run([sys.executable, 'validation/preprocessing_check.py', '--device', args.device])

    reference_cmd = [sys.executable, 'validation/conv_logic_reference_check.py', '--device', args.device]
    if args.skip_timing:
        reference_cmd.append('--skip-timing')
    run(reference_cmd)

    run([sys.executable, 'validation/binary_example_check.py', '--device', args.device])


if __name__ == '__main__':
    main()
