import os
import glob as gb
from argparse import ArgumentParser, BooleanOptionalAction


def run_inference(args):
    data_path = args.path
    # data_path = "../data/DAVIS2016"
    gap = [1, 2]
    reverse = [0, 1]
    rgbpath = data_path + "/JPEGImages"  # path to the dataset
    folder = gb.glob(os.path.join(rgbpath, "*"))

    print(f"folder: {folder}")

    for r in reverse:
        for g in gap:
            for f in folder:
                print("===> Runing {}, gap {}".format(f, g))
                mode = "raft-things.pth"  # model
                if r == 1:
                    raw_outroot = data_path + "/Flows_gap-{}/".format(
                        g
                    )  # where to raw flow
                    outroot = data_path + "/FlowImages_gap-{}/".format(
                        g
                    )  # where to save the image flow
                elif r == 0:
                    raw_outroot = data_path + "/Flows_gap{}/".format(
                        g
                    )  # where to raw flow
                    outroot = data_path + "/FlowImages_gap{}/".format(
                        g
                    )  # where to save the image flow

                clear_flag = "--clear" if args.clear else ""
                os.system(
                    "python predict.py "
                    "--gap {} --model {} --path {} "
                    "--outroot {} --reverse {} --raw_outroot {} {}".format(
                        g, mode, f, outroot, r, raw_outroot, clear_flag
                    )
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    # settings
    parser.add_argument("--path", type=str, default="../data/custom")
    parser.add_argument("--clear", action=BooleanOptionalAction)

    args = parser.parse_args()
    args.inference = True

    run_inference(args)
