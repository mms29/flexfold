
import pandas as pd
import io
import sys
import argparse
import os

def read_star_multi(star_path):
    blocks = {}
    with open(star_path, "r") as f:
        lines = f.readlines()

    current_block = None
    headers = []
    data_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith("data_"):
            # flush previous block if any
            if current_block and headers:
                df = pd.read_csv(
                    io.StringIO("\n".join(data_lines)),
                    delim_whitespace=True,
                    names=headers,
                )
                blocks[current_block] = df.to_dict(orient="list")
            # reset for new block
            current_block = line
            headers, data_lines = [], []
        elif line.startswith("_"):
            headers.append(line.split()[0])
        elif headers and line and not line.startswith("#"):
            data_lines.append(line)

    # flush last block
    if current_block and headers:
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            delim_whitespace=True,
            names=headers,
        )
        blocks[current_block] = df.to_dict(orient="list")

    return blocks

def write_star_from_dicts(star_blocks, out_path, xmipp=False):
    """
    Write a RELION .star file with multiple data blocks.

    Parameters
    ----------
    star_blocks : dict
        Outer keys are block names (e.g. "data_optics", "data_particles").
        Each value is a dict with column names as keys (e.g. "_rlnImageName")
        and lists of entries as values.
    out_path : str
        Path to write the .star file.
    """
    with open(out_path, "w") as f:
        if xmipp:
            f.write("# XMIPP_STAR_1 *\n")
            f.write("#\n")
        for block_name, block_dict in star_blocks.items():
            keys = list(block_dict.keys())
            n_rows = len(next(iter(block_dict.values()))) if block_dict else 0

            # Block header
            f.write(f"{block_name}\n\n")
            f.write("loop_\n")

            # Headers with column numbers
            for i, k in enumerate(keys, start=1):
                f.write(f"{k} #{i}\n")

            # Rows
            for row_idx in range(n_rows):
                row = [str(block_dict[k][row_idx]) for k in keys]
                f.write(" ".join(row) + "\n")

            f.write("\n")  # blank line between blocks


def print_summary(blocks):
    def print_items(items):
        for k2,v2 in items.items():
            print("|--> %s, len=%i, dtype=%s"%(k2, len(v2), type(next(iter(v2)))))
    print("========================== Summary ==========================")
    if isinstance(next(iter(blocks.values())), dict):
        for k,v in blocks.items():
            print(k)
            print("|")
            print_items(v)
            print("")
    else:
        print_items(blocks)

    print("=============================================================")

keywords = [
    ["_anglePsi","_rlnAnglePsi"],
    ["_angleRot","_rlnAngleRot"],
    ["_angleTilt","_rlnAngleTilt"],
    ["_image","_rlnImageName"],
    ["_shiftX","_rlnOriginX"],
    ["_shiftY","_rlnOriginY"],
    ["_shiftX","_rlnOriginXAngst"],
    ["_shiftY","_rlnOriginYAngst"],
    ["_ctfVoltage","_rlnVoltage"],
    ["_ctfSphericalAberration","_rlnSphericalAberration"],
    ["_ctfSamplingRate","_rlnDetectorPixelSize"],
    ["_magnification","_rlnMagnification"],
    ["_ctfDefocusU","_rlnDefocusU"],
    ["_ctfDefocusV","_rlnDefocusV"],
    ["_ctfAstigmatismAngle","_rlnDefocusAngle"],
    ["_ctfPhaseShift","_rlnPhaseShift"],
]

xmp2rln_kw = {k:v for k,v in keywords}
rln2xmp_kw = {v:k for k,v in keywords}

def rln2xmp(star_dict):
    if not any([i in rln2xmp_kw for i in star_dict.keys()]):
        raise RuntimeError("Could not find any Relion key in %s"%str(star_dict.keys()))
    return {(rln2xmp_kw[k] if k in rln2xmp_kw else k):v for k,v in star_dict.items() }
    
def xmp2rln(star_dict):
    if not any([i in xmp2rln_kw for i in star_dict.keys()]):
        raise RuntimeError("Could not find any Xmipp key in %s"%str(star_dict.keys()))
    return {(xmp2rln_kw[k] if k in xmp2rln_kw else k):v for k,v in star_dict.items()}
    

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        type=os.path.abspath,
        help="Input metadata (.star file or .xmd if --inverse)",
    )
    parser.add_argument(
        "output",
        type=os.path.abspath,
        help="output metadata (.xmd file or .star if --inverse)",
    )
    parser.add_argument(
        "--inverse", action="store_true", help="If true, convert Xmipp to Relion, otherwise convert Relion to Xmipp"
    )  
    parser.add_argument(
        "--optics_group_from",
        type=os.path.abspath,
        help="star file to read optic group from",
        default=None,
    )
    parser.add_argument(
        "--add_missing_cols_from",
        type=os.path.abspath,
        help="star file to read missing cols",
        default=None,
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        help="Override pixel size in STAR optics",
        default=None,
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="Override dimensions in STAR optics",
        default=None,
    )
    return parser


def main(args):
    if args.inverse:
        print("Reading XMD file %s ..."%args.input)
        xmd = read_star_multi(args.input)
        print("Done")

        print_summary(xmd)

        print("Reading STAR file with optics group %s ..."%args.optics_group_from)
        optics = read_star_multi(args.optics_group_from)
        print("Done")
        print_summary(optics["data_optics"])

        print("Converting Xmipp to Relion metadata ...")
        converted = xmp2rln(next(iter(xmd.values())))
        print("Done")

        if args.add_missing_cols_from is not None:
            print("Reading STAR file with missing cols %s ..."%args.add_missing_cols_from)
            missing_cols = read_star_multi(args.add_missing_cols_from)
            print("Done")
            missing_cols = missing_cols["data_particles"]
            missing_cols.update(converted)
            converted = missing_cols

        if args.pixel_size is not None:
            optics["data_optics"]["_rlnImagePixelSize"] = [args.pixel_size for i in optics["data_optics"]["_rlnImagePixelSize"]]
        if args.dimension is not None:
            optics["data_optics"]["_rlnImageSize"] = [args.dimension for i in optics["data_optics"]["_rlnImageSize"]]

        print_summary(converted)

        merged = {
            "data_optics":optics["data_optics"],
            "data_particles":converted
        }

        print("Writing STAR file %s ..."%args.output)
        write_star_from_dicts(merged,args.output)
        print("Done")
    else:
        print("Reading STAR file %s ..."%args.input)
        star = read_star_multi(args.input)
        print("Done")

        print_summary(star)

        print("Converting Relion to Xmipp metadata ...")
        converted = rln2xmp(star["data_particles"])
        print("Done")

        converted = {"data_particles" : converted}

        print_summary(converted)

        print("Writing XMD file %s ..."%args.output)
        write_star_from_dicts(converted,args.output, xmipp=True)
        print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser= add_args(parser)

    args = parser.parse_args()
    main(args)


