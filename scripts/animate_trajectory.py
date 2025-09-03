
import argparse



def main(args):
    s = f"""
    lighting soft

    open pc{args.pc_ind}/vol*pdb coordset t
    open pc{args.pc_ind}/*backproject/backproject.mrc 

    volume #11 originIndex {args.grid_size/2}
    vol gaussian #11 sDev  {args.gaussian_sigma}
    vol #12-21 color lightgrey
    vol #12-21 transparency 0.5
    vol #12-21 encloseVolume {args.nres*100}

    fitmap #1 inMap #12
    fitmap #2 inMap #13
    fitmap #3 inMap #14
    fitmap #4 inMap #15
    fitmap #5 inMap #16
    fitmap #6 inMap #17
    fitmap #7 inMap #18
    fitmap #8 inMap #19
    fitmap #9 inMap #20
    fitmap #10 inMap #21


    view all orient

    {"movie record supersample 3" if args.make_movie else ""}

    morph #1-10 same t frames 10 play false
    coordset #22 loop 5 bounce t
    vol morph #12-21 playStep 0.011 frames 910

    {"wait 910" if args.make_movie else ""}
    {"movie encode %i/mov.mp4 quality high"%args.pc_ind if args.make_movie else ""}

    """

    with open(args.outfile, "w") as f:
        f.write(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "outfile", type=str,help="")
    parser.add_argument( "--pc_ind", type=int,help="", default=1)
    parser.add_argument( "--nres", type=int,help="", default=100)
    parser.add_argument( "--gaussian_sigma", type=float,help="", default=1.0)
    parser.add_argument( "--grid_size", type=int,help="", default=100)
    parser.add_argument( "--make_movie",action="store_true",)
    args = parser.parse_args()
    main(args)
    