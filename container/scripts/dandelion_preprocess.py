#!/opt/conda/envs/sc-dandelion-container/bin/python
import argparse
import os
import shutil

import dandelion as ddl
import numpy as np
import pandas as pd
import scanpy as sc

from pathlib import Path
from scanpy import logging as logg

sc.settings.verbosity = 3


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        help=(
            "Optional metadata CSV file, header required, first column for "
            + "sample ID matching folder names in the directory this is being "
            + 'ran in. Can have a "prefix"/"suffix" column for barcode '
            + 'alteration, and "individual" to provide tigger groupings that '
            + "isn't analysing all of the samples jointly."
        ),
    )
    parser.add_argument(
        "--chain",
        type=str,
        default="IG",
        help=(
            "Whether the data is TR or IG, as the preprocessing pipelines "
            + 'differ. Defaults to "IG".'
        ),
    )
    parser.add_argument(
        "--org",
        type=str,
        default="human",
        help=("organism for running the reannotation. human or mouse."),
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="all",
        help=(
            "Which set of contig files to take for the folder. For a given "
            + "PREFIX, will use PREFIX_contig_annotations.csv and "
            + 'PREFIX_contig.fasta. Defaults to "all".'
        ),
    )
    parser.add_argument(
        "--db",
        type=str,
        default="imgt",
        help=("Which database to use for reannotation. imgt or ogrdb."),
    )
    parser.add_argument(
        "--strain",
        type=str,
        default=None,
        help=(
            "Which mouse strain to use for running the reannotation. Only for ogrdb. Defaults to all (None) mouse strains."
        ),
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="_",
        help=(
            "The separator to place between the barcode and prefix/suffix. "
            + "Uses sample names as a prefix for BCR data if metadata CSV "
            + "file absent and more than one sample to process. "
            + 'Defaults to "_".'
        ),
    )
    parser.add_argument(
        "--flavour",
        type=str,
        default="strict",
        help=(
            'The "flavour" for running igblastn reannotation. Accepts either '
            + '"strict" or "original". strict will enforce evalue and penalty cutoffs.'
        ),
    )
    parser.add_argument(
        "--filter_to_high_confidence",
        action="store_true",
        help=(
            "If passed, limits the contig space to ones that are set to "
            + '"True" in the high_confidence column of the contig annotation.'
        ),
    )
    parser.add_argument(
        "--keep_trailing_hyphen_number",
        action="store_false",
        help=(
            "If passed, do not strip out the trailing hyphen number, "
            + 'e.g. "-1", from the end of barcodes.'
        ),
    )
    parser.add_argument(
        "--skip_format_header",
        action="store_true",
        help=("If passed, skips formatting of contig headers."),
    )
    parser.add_argument(
        "--skip_tigger",
        action="store_true",
        help=("If passed, skips TIgGER reassign alleles step."),
    )
    parser.add_argument(
        "--skip_reassign_dj",
        action="store_false",
        help=(
            "If passed, skips reassigning d/j calls with blastn when flavour=strict."
        ),
    )
    parser.add_argument(
        "--skip_correct_c",
        action="store_false",
        help=(
            "If passed, skips correcting c calls at assign_isotypes stage. Only if Chain == IG."
        ),
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help=(
            "If passed, remove intermediate files that aren't the primary "
            + "output from the run reults. The intermediate files may be "
            + "occasionally useful for inspection."
        ),
    )
    args = parser.parse_args()
    # convert loci to lower case for compatibility, and ensure it's in TR/IG
    args.chain = args.chain.lower()
    if args.chain not in ["tr", "ig"]:
        raise ValueError("Chain must be TR or IG")
    return args


def main():
    """Main dandelion-preprocess."""
    logg.info("Software versions:\n")
    ddl.logging.print_header()
    # sponge up command line arguments to begin with
    args = parse_args()
    start = logg.info("\nBegin preprocessing\n")

    if args.keep_trailing_hyphen_number:
        keep_trailing_hyphen_number_log = False
    else:
        keep_trailing_hyphen_number_log = True

    if args.skip_reassign_dj:
        skip_reassign_dj_log = False
    else:
        skip_reassign_dj_log = True

    if args.skip_correct_c:
        skip_correct_c_log = False
    else:
        skip_correct_c_log = True

    logg.info(
        "command line parameters:\n",
        deep=(
            f"--------------------------------------------------------------\n"
            f"    --meta = {args.meta}\n"
            f"    --chain = {args.chain}\n"
            f"    --org = {args.org}\n"
            f"    --file_prefix = {args.file_prefix}\n"
            f"    --db = {args.db}\n"
            f"    --strain = {str(args.strain)}\n"
            f"    --sep = {args.sep}\n"
            f"    --flavour = {args.flavour}\n"
            f"    --filter_to_high_confidence = {args.filter_to_high_confidence}\n"
            f"    --keep_trailing_hyphen_number = {keep_trailing_hyphen_number_log}\n"
            f"    --skip_format_header = {args.skip_format_header}\n"
            f"    --skip_tigger = {args.skip_tigger}\n"
            f"    --skip_reassign_dj = {skip_reassign_dj_log}\n"
            f"    --skip_correct_c = {skip_correct_c_log}\n"
            f"    --clean_output = {args.clean_output}\n"
            f": --------------------------------------------------------------\n"
        ),
    )

    # set up a sample list
    # do we have metadata?
    if args.meta is not None:
        # if so, read it and use the index as the sample list
        meta = pd.read_csv(args.meta, index_col=0)
        samples = [str(s) for s in meta.index]
        if "individual" in meta.columns:
            individuals = list(meta["individual"])
            if not args.skip_tigger:
                if any(ind in samples for ind in individuals):
                    if args.clean_output:
                        raise ValueError(
                            "Individuals in metadata file must not be the same as sample names when `--clean_output` flag is used."
                            "Otherwise, your sample folders will be deleted. "
                            "Please rename the individual or sample folders, or run without `--clean_output`."
                        )
    else:
        # no metadata file. create empty data frame so we can easily check for
        # column presence
        meta = pd.DataFrame()
        # get list of all subfolders in current folder and run with that
        samples = []
        for item in os.listdir("."):
            if os.path.isdir(item):
                if not item.startswith(
                    "."
                ):  # exclude hidden folders like .ipynb_checkpoints
                    samples.append(item)

    # STEP ONE - ddl.pp.format_fastas()
    # do we have a prefix/suffix?
    if not args.skip_format_header:
        if "prefix" in meta.columns:
            # process with prefix
            vals = list(meta["prefix"].values)
            ddl.pp.format_fastas(
                samples,
                prefix=vals,
                sep=args.sep,
                high_confidence_filtering=args.filter_to_high_confidence,
                remove_trailing_hyphen_number=args.keep_trailing_hyphen_number,
                filename_prefix=args.file_prefix,
            )
        elif "suffix" in meta.columns:
            # process with suffix
            vals = list(meta["suffix"].values)
            ddl.pp.format_fastas(
                samples,
                suffix=vals,
                sep=args.sep,
                high_confidence_filtering=args.filter_to_high_confidence,
                remove_trailing_hyphen_number=args.keep_trailing_hyphen_number,
                filename_prefix=args.file_prefix,
            )
        else:
            # neither. tag with the sample names as default, if more than one
            # sample and the data is IG
            if (len(samples) > 1) and (args.chain == "ig"):
                ddl.pp.format_fastas(
                    samples,
                    prefix=samples,
                    sep=args.sep,
                    high_confidence_filtering=args.filter_to_high_confidence,
                    remove_trailing_hyphen_number=args.keep_trailing_hyphen_number,
                    filename_prefix=args.file_prefix,
                )
            else:
                # no need to tag as it's a single sample.
                ddl.pp.format_fastas(
                    samples,
                    high_confidence_filtering=args.filter_to_high_confidence,
                    remove_trailing_hyphen_number=args.keep_trailing_hyphen_number,
                    filename_prefix=args.file_prefix,
                )
    else:
        ddl.pp.format_fastas(
            samples,
            high_confidence_filtering=args.filter_to_high_confidence,
            remove_trailing_hyphen_number=False,
            filename_prefix=args.file_prefix,
        )

    # STEP TWO - ddl.pp.reannotate_genes()
    # no tricks here
    ddl.pp.reannotate_genes(
        samples,
        loci=args.chain,
        org=args.org,
        filename_prefix=args.file_prefix,
        flavour=args.flavour,
        reassign_dj=args.skip_reassign_dj,
        db=args.db,
        strain=args.strain,
    )

    # IG requires further preprocessing, TR is done now
    if args.chain == "ig":
        if not args.skip_tigger:
            # STEP THREE - ddl.pp.reassign_alleles()
            # do we have individual information
            if "individual" in meta.columns:
                # run the function for each individual separately
                for ind in np.unique(meta["individual"]):
                    # yes, this screwy thing is needed so the function ingests it
                    # correctly, sorry
                    ddl.pp.reassign_alleles(
                        [
                            str(i)
                            for i in meta[
                                meta["individual"] == ind
                            ].index.values
                        ],
                        combined_folder=ind,
                        org=args.org,
                        save_plot=True,
                        show_plot=False,
                        filename_prefix=args.file_prefix,
                        db=args.db,
                        strain=args.strain,
                    )
                    # remove if cleaning output - the important information is
                    # ported to sample folders already
                    if args.clean_output:
                        os.system("rm -r " + str(ind))
            else:
                # run on the whole thing at once
                ddl.pp.reassign_alleles(
                    samples,
                    combined_folder="tigger",
                    org=args.org,
                    save_plot=True,
                    show_plot=False,
                    filename_prefix=args.file_prefix,
                    db=args.db,
                    strain=args.strain,
                )
                # remove if cleaning output - the important information is ported
                # to sample folders already
                if args.clean_output:
                    os.system("rm -r tigger")

        # STEP FOUR - ddl.pp.assign_isotypes()
        # also no tricks here
        # only imgt here, there's no ogrdb c references afaik.
        ddl.pp.assign_isotypes(
            samples,
            org=args.org,
            save_plot=True,
            show_plot=False,
            filename_prefix=args.file_prefix,
            correct_c_call=args.skip_correct_c,
            # correction_dict=correction_dict, # TODO: next time, maybe provide path to fasta file so that this can be used?
        )
        # STEP FIVE - ddl.pp.quantify_mutations()
        # this adds the mu_count and mu_freq columns into the table
        for s in samples:
            samp_path = (
                Path(s)
                / "dandelion"
                / (str(args.file_prefix) + "_contig_dandelion.tsv")
            )
            if args.skip_tigger:
                ddl.pp.create_germlines(
                    vdj=samp_path,
                    org=args.org,
                    db=args.db,
                    strain=args.strain,
                    save=samp_path,
                )
            ddl.pp.quantify_mutations(samp_path)
            ddl.pp.quantify_mutations(
                samp_path,
                frequency=True,
            )

    # at this stage it's safe to remove the per-sample dandelion/tmp folder if
    # need be
    if args.clean_output:
        for sample in samples:
            tmp_path = Path(sample) / "dandelion" / "tmp"
            shutil.rmtree(tmp_path)
    logg.info("Pre-processing finished.\n", time=start)


if __name__ == "__main__":
    main()
