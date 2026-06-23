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
            "Optional metadata file. Legacy mode: CSV with first column as "
            + "sample ID matching folder names in the current directory. "
            + "Demultiplex mode: CSV/TSV (or .h5ad, using .obs) with per-cell "
            + "mapping columns for cell ID and individual/sample assignment."
        ),
    )
    parser.add_argument(
        "--meta_cell_id_col",
        type=str,
        default="cell_id",
        help=(
            "Column in --meta containing cell IDs when using per-cell demultiplex metadata. "
            + 'Defaults to "cell_id".'
        ),
    )
    parser.add_argument(
        "--meta_individual_col",
        type=str,
        default="individual",
        help=(
            "Column in --meta containing individual IDs for TIgGER grouping when using per-cell demultiplex metadata. "
            + 'Defaults to "individual".'
        ),
    )
    parser.add_argument(
        "--meta_sample_col",
        type=str,
        default=None,
        help=(
            "Optional source sample folder column in per-cell --meta. "
            + "Required when multiple source sample folders are present."
        ),
    )
    parser.add_argument(
        "--meta_output_col",
        type=str,
        default=None,
        help=(
            "Optional output sample/folder name column in per-cell --meta. "
            + "Defaults to --meta_individual_col."
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

    def discover_sample_folders() -> list[str]:
        folders = []
        for item in os.listdir("."):
            if os.path.isdir(item) and (not item.startswith(".")):
                folders.append(item)
        return sorted(folders)

    def read_meta_table(path: str) -> pd.DataFrame:
        if str(path).lower().endswith(".h5ad"):
            obs = sc.read_h5ad(path).obs.copy()
            if args.meta_cell_id_col not in obs.columns:
                obs[args.meta_cell_id_col] = obs.index.astype(str)
            return obs.reset_index(drop=True)
        # sep=None + python engine infers comma/tab separators.
        return pd.read_csv(path, sep=None, engine="python")

    def split_multiplexed_samples(
        assignment: pd.DataFrame, source_samples: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        cell_col = args.meta_cell_id_col
        individual_col = args.meta_individual_col
        source_col = args.meta_sample_col
        out_col = (
            args.meta_output_col
            if args.meta_output_col is not None
            else individual_col
        )

        if cell_col not in assignment.columns:
            raise ValueError(
                f"Per-cell metadata is missing '{cell_col}' column."
            )
        if individual_col not in assignment.columns:
            raise ValueError(
                f"Per-cell metadata is missing '{individual_col}' column."
            )
        if out_col not in assignment.columns:
            raise ValueError(
                f"Per-cell metadata is missing output column '{out_col}'."
            )

        assign = assignment.copy()
        assign[cell_col] = assign[cell_col].astype(str)
        assign[individual_col] = assign[individual_col].astype(str)
        assign[out_col] = assign[out_col].astype(str)

        if source_col is not None:
            if source_col not in assign.columns:
                raise ValueError(
                    f"Per-cell metadata is missing source sample column '{source_col}'."
                )
            assign[source_col] = assign[source_col].astype(str)
            requested_sources = sorted(assign[source_col].unique().tolist())
        else:
            if len(source_samples) != 1:
                raise ValueError(
                    "Per-cell metadata across multiple folders requires --meta_sample_col."
                )
            source_col = "_source_sample"
            assign[source_col] = source_samples[0]
            requested_sources = [source_samples[0]]

        missing = sorted(set(requested_sources) - set(source_samples))
        if missing:
            raise ValueError(
                "Source sample folders referenced in per-cell metadata were not found: "
                + ", ".join(missing)
            )

        logg.info(
            "Detected per-cell metadata. Running pre-format demultiplex step."
        )

        generated_rows = []
        generated_samples = []
        for src_sample in requested_sources:
            src_assign = assign[assign[source_col] == src_sample]
            if src_assign.empty:
                continue

            vdj = ddl.read_10x_vdj(
                src_sample,
                filename_prefix=args.file_prefix,
                remove_trailing_hyphen_number=args.keep_trailing_hyphen_number,
            )

            for out_name, out_df in src_assign.groupby(out_col, sort=False):
                cell_ids = list(pd.unique(out_df[cell_col]))
                if len(cell_ids) == 0:
                    continue

                sub_vdj = vdj[cell_ids]
                out_folder = Path(str(out_name))
                out_folder.mkdir(parents=True, exist_ok=True)
                sub_vdj.write_10x(
                    folder=out_folder,
                    filename_prefix=args.file_prefix,
                )

                generated_samples.append(str(out_name))
                row = {
                    "sample": str(out_name),
                    "individual": str(out_df[individual_col].iloc[0]),
                }
                if "prefix" in out_df.columns:
                    row["prefix"] = str(out_df["prefix"].iloc[0])
                if "suffix" in out_df.columns:
                    row["suffix"] = str(out_df["suffix"].iloc[0])
                generated_rows.append(row)

        if len(generated_rows) == 0:
            raise ValueError(
                "No demultiplexed outputs were generated from per-cell metadata."
            )

        generated_meta = pd.DataFrame(generated_rows).drop_duplicates(
            subset=["sample"], keep="first"
        )
        generated_meta = generated_meta.set_index("sample")
        generated_samples = sorted(list(pd.unique(generated_samples)))
        return generated_meta, generated_samples

    # set up sample list + metadata
    meta = pd.DataFrame()
    samples = discover_sample_folders()

    if args.meta is not None:
        raw_meta = read_meta_table(args.meta)
        # Flexible --meta mode:
        # 1) per-cell demultiplex mapping if required columns are present
        # 2) legacy sample-level metadata otherwise
        if (
            args.meta_cell_id_col in raw_meta.columns
            and args.meta_individual_col in raw_meta.columns
        ):
            meta, samples = split_multiplexed_samples(raw_meta, samples)
        else:
            # legacy sample-level CSV metadata (first column = sample id)
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
