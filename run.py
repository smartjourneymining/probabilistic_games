import io_alergia_greps
import io_alergia_bpic
import argparse

parser = argparse.ArgumentParser(description='Execution script triggering experiments.')
parser.add_argument('--PRISM_PATH', type=str, default="/home/prism-games-3.2.1-src/prism/bin/prism",
                    help='Path to PRISM-games install')
parser.add_argument('--STORE_PATH', type=str, default="/home/generated/",
                    help='Path to where generated models can be stored')
parser.add_argument('--QUERY_PATH', type=str, default="/home/queries/",
                    help='Path to queries')
parser.add_argument('--OUTPUT_PATH', type=str, default="/home/out/",
                    help='Path to PRISM-games generated output files')
parser.add_argument('--short_execution', type=bool, default=False,
                    help='Set true for exhaustive execution (takes 12h). Short execution takes ~2h.')
args = parser.parse_args()

if __name__ == "__main__":
    print("Start execution")
    print("#### Greps Case Study ####")
    io_alergia_greps.main(args.PRISM_PATH, args.STORE_PATH, args.QUERY_PATH, args.OUTPUT_PATH, args.short_execution) 
    print("#### BPIC Case Study ####")
    io_alergia_bpic.main(args.PRISM_PATH, args.STORE_PATH, args.QUERY_PATH, args.OUTPUT_PATH, args.short_execution)