#!/usr/bin/env python3
"""
Create tab-delimited database file to store sequence alignment information

Modified to make it work for mouse TRA
"""

# Info
__author__ = "Namita Gupta, Jason Anthony Vander Heiden"
from changeo import __version__, __date__

# Imports
import os
import re
import csv
from argparse import ArgumentParser
from collections import OrderedDict
from textwrap import dedent
from time import time
from Bio import SeqIO

# Presto and changeo imports
from presto.Annotation import parseAnnotation
from presto.IO import (
    countSeqFile,
    printLog,
    printMessage,
    printProgress,
    printError,
    printWarning,
    readSeqFile,
)
from changeo.Defaults import (
    default_format,
    default_out_args,
    default_imgt_id_len,
)
from changeo.Commandline import (
    CommonHelpFormatter,
    checkArgs,
    getCommonArgParser,
    parseCommonArgs,
)
from changeo.Alignment import RegionDefinition, gapV
from changeo.Gene import buildGermline
from changeo.IO import (
    countDbFile,
    extractIMGT,
    readGermlines,
    getFormatOperators,
    getOutputHandle,
    AIRRWriter,
    ChangeoWriter,
    IgBLASTReader,
    IgBLASTReaderAA,
    IMGTReader,
    IHMMuneReader,
    checkFields,
)
from changeo.Receptor import ChangeoSchema, AIRRSchema

# 10X Receptor attributes
cellranger_base = ["cell", "c_call", "conscount", "umicount"]
cellranger_extended = [
    "cell",
    "c_call",
    "conscount",
    "umicount",
    "v_call_10x",
    "d_call_10x",
    "j_call_10x",
    "junction_10x",
    "junction_10x_aa",
]


def readCellRanger(cellranger_file, fields=cellranger_base):
    """
    Load a Cell Ranger annotation table

    Arguments:
      cellranger_file (str): path to the annotation file.
      fields (list): list of fields to keep.

    Returns:
      dict: dict of dicts with contig_id as the primary key.
    """
    # Mapping of 10X annotations to Receptor attributes
    cellranger_map = {
        "cell": "barcode",
        "c_call": "c_gene",
        "locus": "chain",
        "conscount": "reads",
        "umicount": "umis",
        "v_call_10x": "v_gene",
        "d_call_10x": "d_gene",
        "j_call_10x": "j_gene",
        "junction_10x": "cdr3_nt",
        "junction_10x_aa": "cdr3",
    }

    # Function to parse individual fields
    def _parse(x):
        return "" if x == "None" else x

    # Generate annotation dictionary
    ann_dict = {}
    with open(cellranger_file) as csv_file:
        # Detect delimiters
        dialect = csv.Sniffer().sniff(csv_file.readline())
        csv_file.seek(0)
        # Read in annotation file
        csv_reader = csv.DictReader(csv_file, dialect=dialect)

        # Generate annotation dictionary
        for row in csv_reader:
            ann_dict[row["contig_id"]] = {
                f: _parse(row[cellranger_map[f]]) for f in fields
            }

    return ann_dict


def addGermline(receptor, references, amino_acid=False):
    """
    Add full length germline to Receptor object

    Arguments:
      receptor (changeo.Receptor.Receptor): Receptor object to modify.
      references (dict): dictionary of IMGT-gapped references sequences.
      amino_acid (bool): if True build amino acid germline, otherwise build nucleotide germline

    Returns:
      changeo.Receptor.Receptor: modified Receptor with the germline sequence added.
    """
    if amino_acid:
        __, germlines, __ = buildGermline(
            receptor, references, seq_field="sequence_aa_imgt", amino_acid=True
        )
        germline_seq = None if germlines is None else germlines["full"]
        receptor.setField("germline_aa_imgt", germline_seq)
    else:
        __, germlines, __ = buildGermline(
            receptor, references, amino_acid=False
        )
        germline_seq = None if germlines is None else germlines["full"]
        receptor.setField("germline_imgt", germline_seq)

    return receptor


def getIDforIMGT(seq_file, imgt_id_len=default_imgt_id_len):
    """
    Create a sequence ID translation using IMGT truncation.

    Arguments:
      seq_file (str): a fasta file of sequences input to IMGT.

    Returns:
      dict: a dictionary of with the IMGT truncated ID as the key and the full sequence description as the value.
    """

    # Create a sequence ID translation using IDs truncate up to space or 49 chars
    ids = {}
    for rec in readSeqFile(seq_file):
        if len(rec.description) <= imgt_id_len:
            id_key = rec.description
        else:  # truncate and replace characters
            if imgt_id_len == 49:  # 28 September 2021 (version 1.8.4)
                id_key = re.sub("\\s|\t", "_", rec.description[:imgt_id_len])
            else:  # older versions
                id_key = re.sub(
                    r"\||\s|!|&|\*|<|>|\?", "_", rec.description[:imgt_id_len]
                )
        ids.update({id_key: rec.description})

    return ids


def correctIMGTFields(receptor, references):
    """
    Add IMGT-numbering to IMGT fields in a Receptor object

    Arguments:
      receptor (changeo.Receptor.Receptor): Receptor object to modify.
      references (dict): dictionary of IMGT-gapped references sequences.

    Returns:
      changeo.Receptor.Receptor: modified Receptor with IMGT-gapped fields.
    """
    # Initialize update object
    imgt_dict = {
        "sequence_imgt": None,
        "v_germ_start_imgt": None,
        "v_germ_length_imgt": None,
        "germline_imgt": None,
    }

    # Check for necessary fields
    try:
        if not all(
            [
                receptor.sequence_imgt,
                receptor.v_germ_start_imgt,
                receptor.v_germ_length_imgt,
                receptor.v_call,
            ]
        ):
            raise AttributeError
    except AttributeError:
        return None

    # Gap V region
    try:
        gapped = gapV(
            receptor.sequence_imgt,
            receptor.v_germ_start_imgt,
            receptor.v_germ_length_imgt,
            receptor.v_call,
            references,
        )
    except KeyError as e:
        printWarning(e)
        return None

    # Verify IMGT-gapped sequence and junction concur
    # try:
    #     check = (receptor.junction == gapped['sequence_imgt'][309:(309 + receptor.junction_length)])
    # except TypeError:
    #     check = False
    # if not check:
    #     return None

    # Rebuild germline sequence
    receptor.setDict(gapped, parse=False)
    __, germlines, __ = buildGermline(receptor, references)
    # log, germlines, genes = buildGermline(receptor, references)
    # print(log)
    if germlines is not None:
        gapped["germline_imgt"] = germlines["full"]
    else:
        return None

    # Update return object
    imgt_dict.update(gapped)

    return imgt_dict


def getSeqDict(seq_file):
    """
    Create a dictionary from a sequence file.

    Arguments:
      seq_file (str): sequence file.

    Returns:
      dict: sequence description as keys with Bio.SeqRecords as values.
    """
    seq_dict = SeqIO.to_dict(
        readSeqFile(seq_file), key_function=lambda x: x.description
    )

    return seq_dict


def writeDb(
    records,
    fields,
    aligner_file,
    total_count,
    id_dict=None,
    annotations=None,
    amino_acid=False,
    validate="gentle",
    asis_id=True,
    regions="default",
    writer=AIRRWriter,
    out_file=None,
    out_args=default_out_args,
):
    """
    Writes parsed records to an output file

    Arguments:
      records (iter): a iterator of Receptor objects containing alignment data.
      fields (list): a list of ordered field names to write.
      aligner_file (str): input file name.
      total_count (int): number of records (for progress bar).
      id_dict (dict): a dictionary of the truncated sequence ID mapped to the full sequence ID.
      annotations (dict): additional annotation dictionary.
      amino_acid (bool): if True do verification on amino acid fields.
      validate (str): validation criteria for passing records; one of 'strict', 'gentle', or 'partial'.
      asis_id (bool): if ID is to be parsed for pRESTO output with default delimiters.
      regions (str): name of the IMGT FWR/CDR region definitions to use.
      writer (changeo.IO.TSVWriter): writer class.
      out_file (str): output file name. Automatically generated from the input file if None.
      out_args (dict): common output argument dictionary from parseCommonArgs.

    Returns:
      None
    """

    # Wrapper for opening handles and writers
    def _open(x, f, writer=writer, out_file=out_file):
        if out_file is not None and x == "pass":
            handle = open(out_file, "w")
        else:
            handle = getOutputHandle(
                aligner_file,
                out_label="db-%s" % x,
                out_dir=out_args["out_dir"],
                out_name=out_args["out_name"],
                out_type=out_args["out_type"],
            )
        return handle, writer(handle, fields=f)

    # Function to convert fasta header annotations to changeo columns
    def _changeo(f, header):
        h = [
            ChangeoSchema.fromReceptor(x) for x in header if x.upper() not in f
        ]
        f.extend(h)
        return f

    def _airr(f, header):
        h = [AIRRSchema.fromReceptor(x) for x in header if x.lower() not in f]
        f.extend(h)
        return f

    # Function to verify IMGT-gapped sequence and junction concur
    def _imgt_check(rec):
        try:
            if amino_acid:
                rd = RegionDefinition(
                    rec.junction_aa_length,
                    amino_acid=amino_acid,
                    definition=regions,
                )
                x, y = rd.positions["junction"]
                check = rec.junction_aa == rec.sequence_aa_imgt[x:y]
            else:
                rd = RegionDefinition(
                    rec.junction_length,
                    amino_acid=amino_acid,
                    definition=regions,
                )
                x, y = rd.positions["junction"]
                check = rec.junction == rec.sequence_imgt[x:y]
        except (TypeError, AttributeError):
            check = False
        return check

    # Default function to check for valid records
    def _gentle(rec):
        if amino_acid:
            valid = [
                rec.v_call and rec.v_call != "None",
                rec.j_call and rec.j_call != "None",
                rec.functional is not None,
                rec.sequence_aa_imgt,
                rec.junction_aa,
            ]
        else:
            valid = [
                rec.v_call and rec.v_call != "None",
                rec.j_call and rec.j_call != "None",
                rec.functional is not None,
                rec.sequence_imgt,
                rec.junction,
            ]
        return all(valid)

    # Function to check for valid records strictly
    def _strict(rec):
        if amino_acid:
            valid = [
                rec.v_call and rec.v_call != "None",
                rec.j_call and rec.j_call != "None",
                rec.functional is not None,
                rec.sequence_aa_imgt,
                rec.junction_aa,
                _imgt_check(rec),
            ]
        else:
            valid = [
                rec.v_call and rec.v_call != "None",
                rec.j_call and rec.j_call != "None",
                rec.functional is not None,
                rec.sequence_imgt,
                rec.junction,
                _imgt_check(rec),
            ]
        return all(valid)

    # Function to check for valid records loosely
    def _partial(rec):
        valid = [
            rec.v_call and rec.v_call != "None",
            rec.d_call and rec.d_call != "None",
            rec.j_call and rec.j_call != "None",
        ]
        return any(valid)

    # Set writer class and annotation conversion function
    if writer == ChangeoWriter:
        _annotate = _changeo
    elif writer == AIRRWriter:
        _annotate = _airr
    else:
        printError("Invalid output writer.")

    # Set pass criteria
    validate_map = {"strict": _strict, "gentle": _gentle, "partial": _partial}
    _pass = validate_map.get(validate, _gentle)

    # Define log handle
    if out_args["log_file"] is None:
        log_handle = None
    else:
        log_handle = open(out_args["log_file"], "w")

    # Initialize handles, writers and counters
    pass_handle, pass_writer = None, None
    fail_handle, fail_writer = None, None
    pass_count, fail_count = 0, 0
    start_time = time()

    # Validate and write output
    printProgress(0, total_count, 0.05, start_time=start_time)
    for i, record in enumerate(records, start=1):
        # Replace sequence description with full string, if required
        if id_dict is not None and record.sequence_id in id_dict:
            record.sequence_id = id_dict[record.sequence_id]

        # Parse sequence description into new columns
        if not asis_id:
            try:
                ann_raw = parseAnnotation(record.sequence_id)
                record.sequence_id = ann_raw.pop("ID")

                # Convert to Receptor fields
                ann_parsed = OrderedDict()
                for k, v in ann_raw.items():
                    ann_parsed[ChangeoSchema.toReceptor(k)] = v

                # Add annotations to Receptor and update field list
                record.setDict(ann_parsed, parse=True)
                if i == 1:
                    fields = _annotate(fields, ann_parsed.keys())
            except IndexError:
                # Could not parse pRESTO-style annotations so fall back to no parse
                asis_id = True
                printWarning(
                    "Sequence annotation format not recognized. Sequence headers will not be parsed."
                )

        # Add supplemental annotation fields
        # if _append_table is not None:
        #     record.setDict(_append_table(record.sequence_id), parse=True)
        if annotations is not None:
            record.setDict(annotations[record.sequence_id], parse=True)
            if i == 1:
                fields = _annotate(
                    fields, annotations[record.sequence_id].keys()
                )

        # Count pass or fail and write to appropriate file
        if _pass(record):
            pass_count += 1
            # Write row to pass file
            try:
                pass_writer.writeReceptor(record)
            except AttributeError:
                # Open pass file and writer
                pass_handle, pass_writer = _open("pass", fields)
                pass_writer.writeReceptor(record)
        else:
            fail_count += 1
            # Write row to fail file if specified
            if out_args["failed"]:
                try:
                    fail_writer.writeReceptor(record)
                except AttributeError:
                    # Open fail file and writer
                    fail_handle, fail_writer = _open("fail", fields)
                    fail_writer.writeReceptor(record)

        # Write log
        if log_handle is not None:
            log = OrderedDict(
                [
                    ("ID", record.sequence_id),
                    ("V_CALL", record.v_call),
                    ("D_CALL", record.d_call),
                    ("J_CALL", record.j_call),
                    ("PRODUCTIVE", record.functional),
                ]
            )
            if not _imgt_check(record) and not amino_acid:
                log["ERROR"] = (
                    "Junction does not match the sequence starting at position 310 in the IMGT numbered V(D)J sequence."
                )
            printLog(log, log_handle)

        # Print progress
        printProgress(i, total_count, 0.05, start_time=start_time)

    # Print console log
    log = OrderedDict()
    log["OUTPUT"] = (
        os.path.basename(pass_handle.name) if pass_handle is not None else None
    )
    log["PASS"] = pass_count
    log["FAIL"] = fail_count
    log["END"] = "MakeDb_gentle"
    printLog(log)

    # Close file handles
    output = {"pass": None, "fail": None}
    if pass_handle is not None:
        output["pass"] = pass_handle.name
        pass_handle.close()
    if fail_handle is not None:
        output["fail"] = fail_handle.name
        fail_handle.close()

    return output


def parseIMGT(
    aligner_file,
    seq_file=None,
    repo=None,
    cellranger_file=None,
    validate="gentle",
    asis_id=True,
    extended=False,
    format=default_format,
    out_file=None,
    out_args=default_out_args,
    imgt_id_len=default_imgt_id_len,
):
    """
    Main for IMGT aligned sample sequences.

    Arguments:
      aligner_file (str): zipped file or unzipped folder output by IMGT.
      seq_file (str): FASTA file input to IMGT (from which to get seqID).
      repo (str): folder with germline repertoire files.
      validate (str): validation criteria for passing records; one of 'strict', 'gentle', or 'partial'.
      asis_id (bool): if ID is to be parsed for pRESTO output with default delimiters.
      extended (bool): if True add alignment score, FWR, CDR and junction fields to output file.
      format (str): output format. one of 'changeo' or 'airr'.
      out_file (str): output file name. Automatically generated from the input file if None.
      out_args (dict): common output argument dictionary from parseCommonArgs.
      imgt_id_len (int): maximum character length of sequence identifiers reported by IMGT/HighV-QUEST.

    Returns:
      dict: names of the 'pass' and 'fail' output files.
    """
    # Print parameter info
    log = OrderedDict()
    log["START"] = "MakeDb_gentle"
    log["COMMAND"] = "imgt"
    log["ALIGNER_FILE"] = aligner_file
    log["SEQ_FILE"] = os.path.basename(seq_file) if seq_file else ""
    log["ASIS_ID"] = asis_id
    log["VALIDATE"] = validate
    log["EXTENDED"] = extended
    printLog(log)

    start_time = time()
    printMessage("Loading files", start_time=start_time, width=20)

    # Extract IMGT files
    temp_dir, imgt_files = extractIMGT(aligner_file)

    # Count records in IMGT files
    total_count = countDbFile(imgt_files["summary"])

    # Get (parsed) IDs from fasta file submitted to IMGT
    id_dict = getIDforIMGT(seq_file, imgt_id_len) if seq_file else {}

    # Load supplementary annotation table
    if cellranger_file is not None:
        f = cellranger_extended if extended else cellranger_base
        annotations = readCellRanger(cellranger_file, fields=f)
    else:
        annotations = None

    printMessage("Done", start_time=start_time, end=True, width=20)

    # Define format operators
    try:
        __, writer, schema = getFormatOperators(format)
    except ValueError:
        printError("Invalid format %s." % format)
    out_args["out_type"] = schema.out_type

    # Define output fields
    fields = list(schema.required)
    if extended:
        custom = IMGTReader.customFields(
            scores=True, regions=True, junction=True, schema=schema
        )
        fields.extend(custom)

    # Parse IMGT output and write db
    with (
        open(imgt_files["summary"]) as summary_handle,
        open(imgt_files["gapped"]) as gapped_handle,
        open(imgt_files["ntseq"]) as ntseq_handle,
        open(imgt_files["junction"]) as junction_handle,
    ):

        # Open parser
        parse_iter = IMGTReader(
            summary_handle, gapped_handle, ntseq_handle, junction_handle
        )

        # Add germline sequence
        if repo is None:
            germ_iter = parse_iter
        else:
            references = readGermlines(repo)
            # Check for IMGT-gaps in germlines
            if all("..." not in x for x in references.values()):
                printWarning(
                    "Germline reference sequences do not appear to contain IMGT-numbering spacers. Results may be incorrect."
                )
            germ_iter = (addGermline(x, references) for x in parse_iter)
        # Write db
        output = writeDb(
            germ_iter,
            fields=fields,
            aligner_file=aligner_file,
            total_count=total_count,
            annotations=annotations,
            id_dict=id_dict,
            asis_id=asis_id,
            validate=validate,
            writer=writer,
            out_file=out_file,
            out_args=out_args,
        )

    # Cleanup temp directory
    temp_dir.cleanup()

    return output


def parseIgBLAST(
    aligner_file,
    seq_file,
    repo,
    amino_acid=False,
    cellranger_file=None,
    validate="gentle",
    asis_id=True,
    asis_calls=False,
    extended=False,
    regions="default",
    infer_junction=False,
    format="changeo",
    out_file=None,
    out_args=default_out_args,
):
    """
    Main for IgBLAST aligned sample sequences.

    Arguments:
      aligner_file (str): IgBLAST output file to process.
      seq_file (str): fasta file input to IgBlast (from which to get sequence).
      repo (str): folder with germline repertoire files.
      amino_acid (bool): if True then the IgBLAST output files are results from igblastp. igblastn is assumed if False.
      validate (str): validation criteria for passing records; one of 'strict', 'gentle', or 'partial'.
      asis_id (bool): if ID is to be parsed for pRESTO output with default delimiters.
      asis_calls (bool): if True do not parse gene calls for allele names.
      extended (bool): if True add alignment scores, FWR regions, and CDR regions to the output.
      regions (str): name of the IMGT FWR/CDR region definitions to use.
      infer_junction (bool): if True, infer the junction sequence, if not reported by IgBLAST.
      format (str): output format. one of 'changeo' or 'airr'.
      out_file (str): output file name. Automatically generated from the input file if None.
      out_args (dict): common output argument dictionary from parseCommonArgs.

    Returns:
      dict: names of the 'pass' and 'fail' output files.
    """
    # Print parameter info
    log = OrderedDict()
    log["START"] = "MakeDB"
    log["COMMAND"] = "igblast-aa" if amino_acid else "igblast"
    log["ALIGNER_FILE"] = os.path.basename(aligner_file)
    log["SEQ_FILE"] = os.path.basename(seq_file)
    log["ASIS_ID"] = asis_id
    log["ASIS_CALLS"] = asis_calls
    log["VALIDATE"] = validate
    log["EXTENDED"] = extended
    log["INFER_JUNCTION"] = infer_junction
    printLog(log)

    # Set amino acid conditions
    if amino_acid:
        format = "%s-aa" % format
        parser = IgBLASTReaderAA
    else:
        parser = IgBLASTReader

    # Start
    start_time = time()
    printMessage("Loading files", start_time=start_time, width=20)

    # Count records in sequence file
    total_count = countSeqFile(seq_file)

    # Get input sequence dictionary
    seq_dict = getSeqDict(seq_file)

    # Create germline repo dictionary
    references = readGermlines(repo, asis=asis_calls)

    # Load supplementary annotation table
    if cellranger_file is not None:
        f = cellranger_extended if extended else cellranger_base
        annotations = readCellRanger(cellranger_file, fields=f)
    else:
        annotations = None

    printMessage("Done", start_time=start_time, end=True, width=20)

    # Check for IMGT-gaps in germlines
    if all("..." not in x for x in references.values()):
        printWarning(
            "Germline reference sequences do not appear to contain IMGT-numbering spacers. Results may be incorrect."
        )

    # Define format operators
    try:
        __, writer, schema = getFormatOperators(format)
    except ValueError:
        printError("Invalid format %s." % format)
    out_args["out_type"] = schema.out_type

    # Define output fields
    fields = list(schema.required)
    if extended:
        custom = parser.customFields(schema=schema)
        fields.extend(custom)

    # Parse and write output
    with open(aligner_file) as f:
        parse_iter = parser(
            f,
            seq_dict,
            references,
            regions=regions,
            asis_calls=asis_calls,
            infer_junction=infer_junction,
        )
        germ_iter = (
            addGermline(x, references, amino_acid=amino_acid)
            for x in parse_iter
        )
        output = writeDb(
            germ_iter,
            fields=fields,
            aligner_file=aligner_file,
            total_count=total_count,
            annotations=annotations,
            amino_acid=amino_acid,
            validate=validate,
            asis_id=asis_id,
            regions=regions,
            writer=writer,
            out_file=out_file,
            out_args=out_args,
        )

    return output


def parseIHMM(
    aligner_file,
    seq_file,
    repo,
    cellranger_file=None,
    validate="gentle",
    asis_id=True,
    extended=False,
    format=default_format,
    out_file=None,
    out_args=default_out_args,
):
    """
    Main for iHMMuneAlign aligned sample sequences.

    Arguments:
      aligner_file (str): iHMMune-Align output file to process.
      seq_file (str): fasta file input to iHMMuneAlign (from which to get sequence).
      repo (str): folder with germline repertoire files.
      validate (str): validation criteria for passing records; one of 'strict', 'gentle', or 'partial'.
      asis_id (bool): if ID is to be parsed for pRESTO output with default delimiters.
      extended (bool): if True parse alignment scores, FWR and CDR region fields.
      format (str): output format. One of 'changeo' or 'airr'.
      out_file (str): output file name. Automatically generated from the input file if None.
      out_args (dict): common output argument dictionary from parseCommonArgs.

    Returns:
      dict: names of the 'pass' and 'fail' output files.
    """
    # Print parameter info
    log = OrderedDict()
    log["START"] = "MakeDB"
    log["COMMAND"] = "ihmm"
    log["ALIGNER_FILE"] = os.path.basename(aligner_file)
    log["SEQ_FILE"] = os.path.basename(seq_file)
    log["ASIS_ID"] = asis_id
    log["VALIDATE"] = validate
    log["EXTENDED"] = extended
    printLog(log)

    start_time = time()
    printMessage("Loading files", start_time=start_time, width=20)

    # Count records in sequence file
    total_count = countSeqFile(seq_file)

    # Get input sequence dictionary
    seq_dict = getSeqDict(seq_file)

    # Create germline repo dictionary
    references = readGermlines(repo)

    # Load supplementary annotation table
    if cellranger_file is not None:
        f = cellranger_extended if extended else cellranger_base
        annotations = readCellRanger(cellranger_file, fields=f)
    else:
        annotations = None

    printMessage("Done", start_time=start_time, end=True, width=20)

    # Check for IMGT-gaps in germlines
    if all("..." not in x for x in references.values()):
        printWarning(
            "Germline reference sequences do not appear to contain IMGT-numbering spacers. Results may be incorrect."
        )

    # Define format operators
    try:
        __, writer, schema = getFormatOperators(format)
    except ValueError:
        printError("Invalid format %s." % format)
    out_args["out_type"] = schema.out_type

    # Define output fields
    fields = list(schema.required)
    if extended:
        custom = IHMMuneReader.customFields(
            scores=True, regions=True, schema=schema
        )
        fields.extend(custom)

    # Parse and write output
    with open(aligner_file) as f:
        parse_iter = IHMMuneReader(f, seq_dict, references)
        germ_iter = (addGermline(x, references) for x in parse_iter)
        output = writeDb(
            germ_iter,
            fields=fields,
            aligner_file=aligner_file,
            total_count=total_count,
            annotations=annotations,
            asis_id=asis_id,
            validate=validate,
            writer=writer,
            out_file=out_file,
            out_args=out_args,
        )

    return output


def numberAIRR(
    aligner_file,
    repo=None,
    format=default_format,
    out_file=None,
    out_args=default_out_args,
):
    """
    Inserts IMGT numbering into V fields

    Arguments:
      aligner_file (str): AIRR Rearrangement file from the alignment tool.
      repo (str): folder with germline repertoire files. If None, do not updated alignment columns with IMGT gaps.
      format (str): output format.
      out_file (str): output file name. Automatically generated from the input file if None.
      out_args (dict): common output argument dictionary from parseCommonArgs.

    Returns:
      str: output file name.
    """
    log = OrderedDict()
    log["START"] = "MakeDb_gentle"
    log["COMMAND"] = "number"
    log["ALIGNER_FILE"] = os.path.basename(aligner_file)
    printLog(log)

    # Define format operators
    try:
        reader, writer, schema = getFormatOperators(format)
    except ValueError:
        printError("Invalid format %s." % format)

    # Open input
    db_handle = open(aligner_file)
    db_iter = reader(db_handle)

    # Define log handle
    if out_args["log_file"] is None:
        log_handle = None
    else:
        log_handle = open(out_args["log_file"], "w")

    # Check for required columns
    try:
        required = ["sequence_imgt", "v_germ_start_imgt"]
        checkFields(required, db_iter.fields, schema=schema)
    except LookupError as e:
        printError(e)

    # Load references
    reference_dict = readGermlines(repo)

    # Check for IMGT-gaps in germlines
    if all("..." not in x for x in reference_dict.values()):
        printWarning(
            "Germline reference sequences do not appear to contain IMGT-numbering spacers. Results may be incorrect."
        )

    # Open output writer
    if out_file is not None:
        pass_handle = open(out_file, "w")
    else:
        pass_handle = getOutputHandle(
            aligner_file,
            out_label="db-pass",
            out_dir=out_args["out_dir"],
            out_name=out_args["out_name"],
            out_type=schema.out_type,
        )
    pass_writer = writer(pass_handle, fields=db_iter.fields)

    if out_args["failed"]:
        fail_handle = getOutputHandle(
            aligner_file,
            out_label="db-fail",
            out_dir=out_args["out_dir"],
            out_name=out_args["out_name"],
            out_type=schema.out_type,
        )
        fail_writer = writer(fail_handle, fields=db_iter.fields)

    # Count records
    result_count = countDbFile(aligner_file)

    # Iterate over records
    start_time = time()
    rec_count = pass_count = fail_count = 0
    for rec in db_iter:
        # Print progress for previous iteration
        printProgress(rec_count, result_count, 0.05, start_time=start_time)
        rec_count += 1
        # Update IMGT fields
        imgt_dict = correctIMGTFields(rec, reference_dict)
        # Write records
        if imgt_dict is not None:
            pass_count += 1
            rec.setDict(imgt_dict, parse=False)
            pass_writer.writeReceptor(rec)
        else:
            fail_count += 1
            # Write row to fail file if specified
            if out_args["failed"]:
                fail_writer.writeReceptor(rec)
        # Write log
        if log_handle is not None:
            log = OrderedDict(
                [
                    ("ID", rec.sequence_id),
                    ("V_CALL", rec.v_call),
                    ("D_CALL", rec.d_call),
                    ("J_CALL", rec.j_call),
                    ("PRODUCTIVE", rec.functional),
                ]
            )
            printLog(log, log_handle)

    # Print counts
    printProgress(rec_count, result_count, 0.05, start_time=start_time)
    log = OrderedDict()
    log["OUTPUT"] = os.path.basename(pass_handle.name)
    log["RECORDS"] = rec_count
    log["PASS"] = pass_count
    log["FAIL"] = rec_count - pass_count
    log["END"] = "MakeDb_gentle"
    printLog(log)

    # Close file handles
    pass_handle.close()
    db_handle.close()

    return pass_handle.name


def getArgParser():
    """
    Defines the ArgumentParser.

    Returns:
      argparse.ArgumentParser
    """
    fields = dedent(
        """
              output files:
                  db-pass
                      database of alignment records with functionality information,
                      V and J calls, and a junction region.
                  db-fail
                      database with records that fail due to no productivity information,
                      no gene V assignment, no J assignment, or no junction region.

              universal output fields:
                 sequence_id, sequence, sequence_alignment, germline_alignment,
                 rev_comp, productive, stop_codon, vj_in_frame, locus,
                 v_call, d_call, j_call, c_call, junction, junction_length, junction_aa,
                 v_sequence_start, v_sequence_end, v_germline_start, v_germline_end,
                 d_sequence_start, d_sequence_end, d_germline_start, d_germline_end,
                 j_sequence_start, j_sequence_end, j_germline_start, j_germline_end,
                 np1_length, np2_length, fwr1, fwr2, fwr3, fwr4, cdr1, cdr2, cdr3

              imgt specific output fields:
                  n1_length, n2_length, p3v_length, p5d_length, p3d_length, p5j_length,
                  d_frame, v_score, v_identity, d_score, d_identity, j_score, j_identity

              igblast specific output fields:
                  v_score, v_identity, v_support, v_cigar,
                  d_score, d_identity, d_support, d_cigar,
                  j_score, j_identity, j_support, j_cigar

              ihmm specific output fields:
                  vdj_score

              10x specific output fields:
                  cell_id, consensus_count, umi_count,
                  v_call_10x, d_call_10x, j_call_10x,
                  junction_10x, junction_10x_aa
              """
    )

    # Define ArgumentParser
    parser = ArgumentParser(
        description=__doc__,
        epilog=fields,
        formatter_class=CommonHelpFormatter,
        add_help=False,
    )
    group_help = parser.add_argument_group("help")
    group_help.add_argument(
        "--version",
        action="version",
        version="%(prog)s:" + " {} {}".format(__version__, __date__),
    )
    group_help.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
    subparsers = parser.add_subparsers(
        title="subcommands", dest="command", help="Aligner used", metavar=""
    )
    # TODO:  This is a temporary fix for Python issue 9253
    subparsers.required = True

    # Parent parser
    parser_parent = getCommonArgParser(db_in=False)

    # igblastn output parser
    parser_igblast = subparsers.add_parser(
        "igblast",
        parents=[parser_parent],
        formatter_class=CommonHelpFormatter,
        add_help=False,
        help="Process igblastn output.",
        description="Process igblastn output.",
    )
    group_igblast = parser_igblast.add_argument_group(
        "aligner parsing arguments"
    )
    group_igblast.add_argument(
        "-i",
        nargs="+",
        action="store",
        dest="aligner_files",
        required=True,
        help="""IgBLAST output files in format 7 with query sequence
                                     (igblastn argument \'-outfmt "7 std qseq sseq btop"\').""",
    )
    group_igblast.add_argument(
        "-r",
        nargs="+",
        action="store",
        dest="repo",
        required=True,
        help="""List of folders and/or fasta files containing
                                     the same germline set used in the IgBLAST alignment. These
                                     reference sequences must contain IMGT-numbering spacers (gaps)
                                     in the V segment.""",
    )
    group_igblast.add_argument(
        "-s",
        action="store",
        nargs="+",
        dest="seq_files",
        required=True,
        help="""List of input FASTA files (with .fasta, .fna or .fa
                                     extension), containing sequences.""",
    )
    group_igblast.add_argument(
        "--10x",
        action="store",
        nargs="+",
        dest="cellranger_files",
        help="""Table file containing 10X annotations (with .csv or .tsv
                                     extension).""",
    )
    group_igblast.add_argument(
        "--asis-id",
        action="store_true",
        dest="asis_id",
        help="""Specify to prevent input sequence headers from being parsed
                                     to add new columns to database. Parsing of sequence headers requires
                                     headers to be in the pRESTO annotation format, so this should be specified
                                     when sequence headers are incompatible with the pRESTO annotation scheme.
                                     Note, unrecognized header formats will default to this behavior.""",
    )
    group_igblast.add_argument(
        "--asis-calls",
        action="store_true",
        dest="asis_calls",
        help="""Specify to prevent gene calls from being parsed into standard allele names
                                     in both the IgBLAST output and reference database. Note, this requires
                                     the sequence identifiers in the reference sequence set and the IgBLAST
                                     database to be exact string matches.""",
    )
    group_igblast.add_argument(
        "--extended",
        action="store_true",
        dest="extended",
        help="""Specify to include additional aligner specific fields in the output.
                                    Adds <vdj>_score, <vdj>_identity, <vdj>_support, <vdj>_cigar,
                                    fwr1, fwr2, fwr3, fwr4, cdr1, cdr2 and cdr3.""",
    )
    group_igblast.add_argument(
        "--regions",
        action="store",
        dest="regions",
        choices=("default", "rhesus-igl"),
        default="default",
        help="""IMGT CDR and FWR boundary definition to use.""",
    )
    group_igblast.add_argument(
        "--infer-junction",
        action="store_true",
        dest="infer_junction",
        help="""Infer the junction sequence. For use with IgBLAST v1.6.0 or older,
                                    prior to the addition of IMGT-CDR3 inference.""",
    )
    group_igblast_validate = group_igblast.add_mutually_exclusive_group(
        required=False
    )
    # group_igblast_validate.add_argument('--strict', action='store_const', const='strict', dest='validate',
    #                                     help='''By default, passing records must contain valid values for the
    #                                          V gene, J gene, junction region, and productivity call. If specified,
    #                                          this argument adds the additional requirement that the junction region must
    #                                          start at position 310 in the IMGT-numbered sequence.''')
    group_igblast_validate.add_argument(
        "--partial",
        action="store_const",
        const="partial",
        dest="validate",
        help="""If specified, include incomplete V(D)J alignments in
                                             the pass file instead of the fail file. An incomplete alignment
                                             is defined as a record that is missing a V gene assignment,
                                             J gene assignment, junction region, or productivity call.""",
    )
    parser_igblast.set_defaults(
        func=parseIgBLAST, amino_acid=False, validate="gentle"
    )

    # igblastp output parser
    parser_igblast_aa = subparsers.add_parser(
        "igblast-aa",
        parents=[parser_parent],
        formatter_class=CommonHelpFormatter,
        add_help=False,
        help="Process igblastp output.",
        description="Process igblastp output.",
    )
    group_igblast_aa = parser_igblast_aa.add_argument_group(
        "aligner parsing arguments"
    )
    group_igblast_aa.add_argument(
        "-i",
        nargs="+",
        action="store",
        dest="aligner_files",
        required=True,
        help="""IgBLAST output files in format 7 with query sequence
                                       (igblastp argument \'-outfmt "7 std qseq sseq btop"\').""",
    )
    group_igblast_aa.add_argument(
        "-r",
        nargs="+",
        action="store",
        dest="repo",
        required=True,
        help="""List of folders and/or fasta files containing
                                       the same germline set used in the IgBLAST alignment. These
                                       reference sequences must contain IMGT-numbering spacers (gaps)
                                       in the V segment.""",
    )
    group_igblast_aa.add_argument(
        "-s",
        action="store",
        nargs="+",
        dest="seq_files",
        required=True,
        help="""List of input FASTA files (with .fasta, .fna or .fa
                                       extension), containing sequences.""",
    )
    group_igblast_aa.add_argument(
        "--10x",
        action="store",
        nargs="+",
        dest="cellranger_files",
        help="""Table file containing 10X annotations (with .csv or .tsv extension).""",
    )
    group_igblast_aa.add_argument(
        "--asis-id",
        action="store_true",
        dest="asis_id",
        help="""Specify to prevent input sequence headers from being parsed
                                       to add new columns to database. Parsing of sequence headers requires
                                       headers to be in the pRESTO annotation format, so this should be specified
                                       when sequence headers are incompatible with the pRESTO annotation scheme.
                                       Note, unrecognized header formats will default to this behavior.""",
    )
    group_igblast_aa.add_argument(
        "--asis-calls",
        action="store_true",
        dest="asis_calls",
        help="""Specify to prevent gene calls from being parsed into standard allele names
                                       in both the IgBLAST output and reference database. Note, this requires
                                       the sequence identifiers in the reference sequence set and the IgBLAST
                                       database to be exact string matches.""",
    )
    group_igblast_aa.add_argument(
        "--extended",
        action="store_true",
        dest="extended",
        help="""Specify to include additional aligner specific fields in the output.
                                       Adds v_score, v_identity, v_support, v_cigar, fwr1, fwr2, fwr3, cdr1 and cdr2.""",
    )
    group_igblast_aa.add_argument(
        "--regions",
        action="store",
        dest="regions",
        choices=("default", "rhesus-igl"),
        default="default",
        help="""IMGT CDR and FWR boundary definition to use.""",
    )
    parser_igblast_aa.set_defaults(
        func=parseIgBLAST, amino_acid=True, validate="gentle"
    )

    # IMGT aligner
    parser_imgt = subparsers.add_parser(
        "imgt",
        parents=[parser_parent],
        formatter_class=CommonHelpFormatter,
        add_help=False,
        help="""Process IMGT/HighV-Quest output
                                             (does not work with V-QUEST).""",
        description="""Process IMGT/HighV-Quest output
                                             (does not work with V-QUEST).""",
    )
    group_imgt = parser_imgt.add_argument_group("aligner parsing arguments")
    group_imgt.add_argument(
        "-i",
        nargs="+",
        action="store",
        dest="aligner_files",
        required=True,
        help="""Either zipped IMGT output files (.zip or .txz) or a
                                 folder containing unzipped IMGT output files (which must
                                 include 1_Summary, 2_IMGT-gapped, 3_Nt-sequences,
                                 and 6_Junction).""",
    )
    group_imgt.add_argument(
        "-s",
        nargs="*",
        action="store",
        dest="seq_files",
        required=False,
        help="""List of FASTA files (with .fasta, .fna or .fa
                                  extension) that were submitted to IMGT/HighV-QUEST.
                                  If unspecified, sequence identifiers truncated by IMGT/HighV-QUEST
                                  will not be corrected.""",
    )
    group_imgt.add_argument(
        "-r",
        nargs="+",
        action="store",
        dest="repo",
        required=False,
        help="""List of folders and/or fasta files containing
                                 the germline sequence set used by IMGT/HighV-QUEST.
                                 These reference sequences must contain IMGT-numbering spacers (gaps)
                                 in the V segment. If unspecified, the germline sequence reconstruction
                                 will not be included in the output.""",
    )
    group_imgt.add_argument(
        "--10x",
        action="store",
        nargs="+",
        dest="cellranger_files",
        help="""Table file containing 10X annotations (with .csv or .tsv
                                 extension).""",
    )
    group_imgt.add_argument(
        "--extended",
        action="store_true",
        dest="extended",
        help="""Specify to include additional aligner specific fields in the output.
                                 Adds <vdj>_score, <vdj>_identity>, fwr1, fwr2, fwr3, fwr4,
                                 cdr1, cdr2, cdr3, n1_length, n2_length, p3v_length, p5d_length,
                                 p3d_length, p5j_length and d_frame.""",
    )
    group_imgt.add_argument(
        "--asis-id",
        action="store_true",
        dest="asis_id",
        help="""Specify to prevent input sequence headers from being parsed
                                 to add new columns to database. Parsing of sequence headers requires
                                 headers to be in the pRESTO annotation format, so this should be specified
                                 when sequence headers are incompatible with the pRESTO annotation scheme.
                                 Note, unrecognized header formats will default to this behavior.""",
    )
    group_imgt.add_argument(
        "--imgt-id-len",
        action="store",
        dest="imgt_id_len",
        type=int,
        default=default_imgt_id_len,
        help="""The maximum character length of sequence identifiers reported by IMGT/HighV-QUEST.
                                 Specify 50 if the IMGT files (-i) were generated with an IMGT/HighV-QUEST version older
                                 than 1.8.3 (May 7, 2021).""",
    )
    group_imgt_validate = group_imgt.add_mutually_exclusive_group(
        required=False
    )
    # group_imgt_validate.add_argument('--strict', action='store_const', const='strict', dest='validate',
    #                                  help='''By default, passing records must contain valid values for the
    #                                       V gene, J gene, junction region, and productivity call. If specified,
    #                                       this argument adds the additional requirement that the junction region must
    #                                       start at position 310 in the IMGT-numbered sequence.''')
    group_imgt_validate.add_argument(
        "--partial",
        action="store_const",
        const="partial",
        dest="validate",
        help="""If specified, include incomplete V(D)J alignments in
                                          the pass file instead of the fail file. An incomplete alignment
                                          is defined as a record that is missing a V gene assignment,
                                          J gene assignment, junction region, or productivity call.""",
    )
    parser_imgt.set_defaults(func=parseIMGT, validate="gentle")

    # iHMMuneAlign Aligner
    parser_ihmm = subparsers.add_parser(
        "ihmm",
        parents=[parser_parent],
        formatter_class=CommonHelpFormatter,
        add_help=False,
        help="Process iHMMune-Align output.",
        description="Process iHMMune-Align output.",
    )
    group_ihmm = parser_ihmm.add_argument_group("aligner parsing arguments")
    group_ihmm.add_argument(
        "-i",
        nargs="+",
        action="store",
        dest="aligner_files",
        required=True,
        help="""iHMMune-Align output file.""",
    )
    group_ihmm.add_argument(
        "-r",
        nargs="+",
        action="store",
        dest="repo",
        required=True,
        help="""List of folders and/or FASTA files containing
                                   the set of germline sequences used by iHMMune-Align. These
                                   reference sequences must contain IMGT-numbering spacers (gaps)
                                   in the V segment.""",
    )
    group_ihmm.add_argument(
        "-s",
        action="store",
        nargs="+",
        dest="seq_files",
        required=True,
        help="""List of input FASTA files (with .fasta, .fna or .fa
                                  extension) containing sequences.""",
    )
    group_ihmm.add_argument(
        "--10x",
        action="store",
        nargs="+",
        dest="cellranger_files",
        help="""Table file containing 10X annotations (with .csv or .tsv
                                     extension).""",
    )
    group_ihmm.add_argument(
        "--asis-id",
        action="store_true",
        dest="asis_id",
        help="""Specify to prevent input sequence headers from being parsed
                                  to add new columns to database. Parsing of sequence headers requires
                                  headers to be in the pRESTO annotation format, so this should be specified
                                  when sequence headers are incompatible with the pRESTO annotation scheme.
                                  Note, unrecognized header formats will default to this behavior.""",
    )
    group_ihmm.add_argument(
        "--extended",
        action="store_true",
        dest="extended",
        help="""Specify to include additional aligner specific fields in the output.
                                  Adds the path score of the iHMMune-Align hidden Markov model as vdj_score;
                                  adds fwr1, fwr2, fwr3, fwr4, cdr1, cdr2 and cdr3.""",
    )
    group_ihmm_validate = group_ihmm.add_mutually_exclusive_group(
        required=False
    )
    # group_ihmm_validate.add_argument('--strict', action='store_const', const='strict', dest='validate',
    #                                  help='''By default, passing records must contain valid values for the
    #                                       V gene, J gene, junction region, and productivity call. If specified,
    #                                       this argument adds the additional requirement that the junction region must
    #                                       start at position 310 in the IMGT-numbered sequence.''')
    group_ihmm_validate.add_argument(
        "--partial",
        action="store_const",
        const="partial",
        dest="validate",
        help="""If specified, include incomplete V(D)J alignments in
                                          the pass file instead of the fail file. An incomplete alignment
                                          is defined as a record that is missing a V gene assignment,
                                          J gene assignment, junction region, or productivity call.""",
    )
    parser_ihmm.set_defaults(func=parseIHMM, validate="gentle")

    # Subparser to normalize AIRR file with IMGT-numbering
    # desc_number = dedent('''
    #                      Inserts IMGT numbering spacers into sequence_alignment, rebuilds the germline sequence
    #                      in germline_alignment, and adjusts the values in the coordinate fields v_germline_start
    #                      and v_germline_end accordingly.
    #                      ''')
    # parser_number = subparsers.add_parser('number', parents=[parser_parent],
    #                                       formatter_class=CommonHelpFormatter, add_help=False,
    #                                       help='Add IMGT-numbering to an AIRR Rearrangement TSV.',
    #                                       description=desc_number)
    # group_number = parser_number.add_argument_group('aligner parsing arguments')
    # group_number.add_argument('-i', nargs='+', action='store', dest='aligner_files', required=True,
    #                         help='''AIRR Rearrangement TSV files.''')
    # group_number.add_argument('-r', nargs='+', action='store', dest='repo', required=False,
    #                         help='''List of folders and/or fasta files containing
    #                              IMGT-numbered germline sequences corresponding to the
    #                              set of germlines used for the alignment.''')
    # parser_number.set_defaults(func=numberAIRR)

    return parser


def main():
    """
    Parses command line arguments and calls main
    """
    parser = getArgParser()
    checkArgs(parser)
    args = parser.parse_args()
    args_dict = parseCommonArgs(args, in_arg="aligner_files")

    # Set no ID parsing if sequence files are not provided
    if "seq_files" in args_dict and not args_dict["seq_files"]:
        args_dict["asis_id"] = True

    # Delete
    if "aligner_files" in args_dict:
        del args_dict["aligner_files"]
    if "seq_files" in args_dict:
        del args_dict["seq_files"]
    if "cellranger_files" in args_dict:
        del args_dict["cellranger_files"]
    if "out_files" in args_dict:
        del args_dict["out_files"]
    if "command" in args_dict:
        del args_dict["command"]
    if "func" in args_dict:
        del args_dict["func"]

    # Call main
    for i, f in enumerate(args.__dict__["aligner_files"]):
        args_dict["aligner_file"] = f
        args_dict["out_file"] = (
            args.__dict__["out_files"][i]
            if args.__dict__["out_files"]
            else None
        )
        if "seq_files" in args.__dict__:
            args_dict["seq_file"] = (
                args.__dict__["seq_files"][i]
                if args.__dict__["seq_files"]
                else None
            )
        if "cellranger_files" in args.__dict__:
            args_dict["cellranger_file"] = (
                args.__dict__["cellranger_files"][i]
                if args.__dict__["cellranger_files"]
                else None
            )
        args.func(**args_dict)


if __name__ == "__main__":
    main()
