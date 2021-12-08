import json
import sys

"""run_gift.py
This module contains default parameters and functions for running
gift via NiPype.
Functions:
    gift_gica - run group independent component analysis
    gift_dfnc - run dynamic functional network connectivity
    gift_mancova - run mancova
Todo:
    * Finishg Adding Mancova parameters
    * Merge any more shared parameters
    * Mancova docstring
    * Tests
"""
import os

# import utils as ut
import nipype.interfaces.gift as gift
from nipype import config, logging
import nibabel as nib
from nibabel.processing import resample_from_to
from nibabel.funcs import four_to_three
import argparse

# from django.conf import settings

# CONSTANTS
ICA_TYPES = ["spatial", "temporal"]
PCA_TYPES = ["subject specific", "group grand mean"]
BACK_RECON = ["regular", "spatial-temporal regression", "gica3", "gica", "gig-ica"]
PREPROC_TYPES = [
    "remove mean per timepoint",
    "remove mean per voxel",
    "intensity normalization",
    "variance normalization",
]
SCALE_TYPE = ["No scaling", "percent signal change", "Z-scores"]
WHICH_ANALYSIS = ["standard", "ICASSO", "MST"]
ICA_ALGORITHMS = [
    "InfoMax",
    "Fast ICA",
    "Erica",
    "Simbec",
    "Evd",
    "Jade Opac",
    "Amuse",
    "SDD ICA",
    "Semi-blind Infomax",
    "Constrained ICA (Spatial)",
    "Radical ICA",
    "Combi",
    "ICA-EBM",
    "ERBM",
    "IVA-GL",
    "GIG-ICA",
    "IVA-L",
]

# SHARED DEFAULTS
# matlab_cmd = os.getenv('MATLAB_COMMAND')
matlab_cmd = "/app/groupicatv4.0b/GroupICATv4.0b_standalone/run_groupica.sh /usr/local/MATLAB/MATLAB_Runtime/v91/"
# DEFAULT_OUT_DIR = os.path.join(str(settings.ROOT_DIR), 'media', 'figures')
DEFAULT_OUT_DIR = "/out"
DEFAULT_DISPLAY_RESULTS = 1
DEFAULT_NUM_COMPS = 53
DEFAULT_COMP_NETWORK_NAMES = {}
DEFAULT_TR = 2

# GICA DEFAULTS
DEFAULT_DIM = 53
DEFAULT_ALG = 1
DEFAULT_ICA_PARAM_FILE = ""
DEFAULT_REFS = os.path.join("/app", "template", "NeuroMark.nii")
DEFAULT_RUN_NAME = "dfnc"
DEFAULT_GROUP_PCA_TYPE = 0
DEFAULT_BACK_RECON_TYPE = 1
DEFAULT_PREPROC_TYPE = 1
DEFAULT_NUM_REDUCTION_STEPS = 1
DEFAULT_SCALE_TYPE = 1
DEFAULT_GROUP_ICA_TYPE = "spatial"
DEFAULT_WHICH_ANALYSIS = 1
DEFAULT_MASK = None

#Connectogram plot options
DEFAULT_Network_summary_comp_network_names = {"SC": [1, 2, 3, 4, 5], "AUD": [6, 7], "SM": [8, 9, 10, 11, 12, 13, 14, 15, 16], "VIS": [17, 18, 19, 20, 21, 22, 23, 24, 25], "CC": [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42], "DMN": [43, 44, 45, 46, 47, 48, 49], "CR": [50, 51, 52, 53]}
DEFAULT_Network_summary_conn_threshold = 0.0
DEFAULT_Network_summary_threshold = 1

#Temporal sort options
DEFAULT_SPM_MAT_FILE_EXISTS= None
DEFAULT_REGRESSORS_OF_INTEREST = None
DEFAULT_SPM_MAT_FILE_NAME="SPM.mat"

# dFNC DEFAULTS
# populated into the dfnc_parameters dict
DEFAULT_TC_DETREND = (3,)
DEFAULT_TC_DESPIKE = ("yes",)
DEFAULT_TC_FILTER = (0.15,)
DEFAULT_TC_COVARIATES_FILES_LIST = []
DEFAULT_TC_COVARIATES_FILE_NUMBERS = []
DEFAULT_METHOD = "none"
DEFAULT_WSIZE = 30
DEFAULT_WINDOW_ALPHA = 3
DEFAULT_NUM_REPETITIONS = 10
# populated into the postprocessing dict
DEFAULT_NUM_CLUSTERS = 5
DEFAULT_ICA_ALGORITHM = "infomax"
DEFAULT_ICA_NUM_ICA_RUNS = 5
DEFAULT_REGRESS_COV_FILE = ""
DEFAULT_KMEANS_MAX_ITER = 150
DEFAULT_DMETHOD = "city"

# ManCova
DEFAULT_FEATURES = []
DEFAULT_COVARIATES = {}
DEFAULT_INTERACTIONS = []
DEFAULT_FEATURE_PARAMS = {}
DEFAULT_P_THRESHOLD = 1.0


def gift_gica(
    in_files=[],
    dim=DEFAULT_DIM,
    algoType=DEFAULT_ALG,
    refFiles=DEFAULT_REFS,
    run_name=DEFAULT_RUN_NAME,
    out_dir=DEFAULT_OUT_DIR,
    group_pca_type=DEFAULT_GROUP_PCA_TYPE,
    backReconType=DEFAULT_BACK_RECON_TYPE,
    preproc_type=DEFAULT_PREPROC_TYPE,
    numReductionSteps=DEFAULT_NUM_REDUCTION_STEPS,
    scaleType=DEFAULT_SCALE_TYPE,
    group_ica_type=DEFAULT_GROUP_ICA_TYPE,
    display_results=DEFAULT_DISPLAY_RESULTS,
    which_analysis=DEFAULT_WHICH_ANALYSIS,
    mask=DEFAULT_MASK,
    NS_comp_network_names=DEFAULT_Network_summary_comp_network_names,
    NS_conn_threshold=DEFAULT_Network_summary_conn_threshold,
    NS_threshold=DEFAULT_Network_summary_threshold,
    TS_SPM_mat_file_path='',
    TS_SPM_mat_file_exists=DEFAULT_SPM_MAT_FILE_EXISTS,
    TS_regressors_of_interest=DEFAULT_REGRESSORS_OF_INTEREST,
    **kwargs
):
    """
    Wrapper for initializing GIFT nipype interface to run Group ICA.
    Args:
        in_files            (List [Str])    :   Input file names (either single file name or a list)
        dim                 (Int)           :   Dimensionality reduction into #num dimensions
        algoType            (Int)           :   options are 1 - Infomax, 2 - Fast ica , ...
        refFiles            (List [Str])    :   file names for reference templates (either single file name or a list)
        run_name            (Str)           :   Name of the analysis run
        out_dir             (Str)           :   Full file path of the results directory
        group_pca_type      (Str)           :   options are 'subject specific' and 'grand mean'
        backReconType       (Int)           :   options are 1 - regular, 2 - spatial-temporal regression, 3 - gica3, 4 - gica, 5 - gig-ica
        preproc_type        (Int)           :   options are 1 - remove mean per timepoint, 2 - remove mean per voxel, 3 - intensity norm, 4 - variance norm
        numReductionSteps   (Int)           :   Number of reduction steps used in the first pca step
        scaleType           (Int)           :   options are 0 - No scaling, 1 - percent signal change, 2 - Z-scores
        group_ica_type      (Str)           :   1 - Spatial ica, 2 - Temporal ica.
        display_results     (Int)           :   0 - No display, 1 - HTML report, 2 - PDF
        which_analysis      (Int)           :   Options are 1, 2, and 3. 1 - standard group ica, 2 - ICASSO and 3 - MST.
        mask                (Str)           :   Enter file names using full path of the mask. If you wish to use default mask leave it empty
        algoType full options:
        1           2           3       4           5       6
        'Infomax'   'Fast ICA'  'Erica' 'Simbec'    'Evd'   'Jade Opac',
        7           8           9                   10
        'Amuse'     'SDD ICA'   'Semi-blind'        'Constrained ICA (Spatial)'
        11              12      13          14      15          16          17
        'Radical ICA'   'Combi' 'ICA-EBM'   'ERBM'  'IVA-GL'    'GIG-ICA'   'IVA-L'
    Args (not supported here, but available for nipype):
        perfType            (Int)           :   Options are 1, 2, and 3. 1 - maximize performance, 2 - less memory usage  and 3 - user specified settings.
        prefix              (Str)           :   Enter prefix to be appended with the output files
        dummy_scans         (Int)           :   enter dummy scans
        numWorkers          (Int)           :   Number of parallel workers
        doEstimation        (Int)           :   options are 0 and 1
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gift.GICACommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

    gc = gift.GICACommand()
    gc.inputs.in_files = in_files
    gc.inputs.algoType = algoType
    gc.inputs.group_pca_type = group_pca_type
    gc.inputs.backReconType = backReconType
    gc.inputs.preproc_type = preproc_type
    gc.inputs.numReductionSteps = numReductionSteps
    gc.inputs.scaleType = scaleType
    gc.inputs.group_ica_type = group_ica_type
    gc.inputs.which_analysis = which_analysis
    gc.inputs.refFiles = get_interpolated_nifti(in_files[0], refFiles, out_dir)
    gc.inputs.display_results = display_results
    if mask is not None:
        gc.inputs.mask = mask

    # if comp_network_names is not None:
    #     gc.inputs.network_summary_opts = {"comp_network_names": comp_network_names}
    gc.inputs.network_summary_opts={"comp_network_names": NS_comp_network_names,'conn_threshold':NS_conn_threshold,'threshold':NS_threshold}

    # Get temportal sorting options from params.json
    stats_folder = '/'.join(str(in_files[0]).split('/')[:-2])+'/stats/'

    if TS_SPM_mat_file_exists and os.path.exists(os.path.join(stats_folder,DEFAULT_SPM_MAT_FILE_NAME)):
        designMatrix_list=[]
        designMatrix_list.append(os.path.join(stats_folder, DEFAULT_SPM_MAT_FILE_NAME))
        gc.inputs.designMatrix=designMatrix_list
    if os.path.exists(TS_SPM_mat_file_path):
        designMatrix_list=[]
        designMatrix_list.append(TS_SPM_mat_file_path)
        gc.inputs.designMatrix=designMatrix_list
    if  TS_regressors_of_interest is not None:
        regressors_list=(str(TS_regressors_of_interest)).split(",")
        gc.inputs.regressors = regressors_list

    if dim > 0:
        gc.inputs.dim = dim

    gc.inputs.out_dir = out_dir

    return gc.run()


def resolve_comp_network_names(num_comps, network_names):
    """
        Resolve the DEFAULT network names, so that the number of components in that dict does
        not exceed the desired num_comps. This is required for dFNC input.
        Currently just use the network numbers.
    """
    return {"%d" % v: v + 1 for v in range(num_comps)}


def gift_dfnc(
    ica_param_file=DEFAULT_ICA_PARAM_FILE,
    out_dir=DEFAULT_OUT_DIR,
    run_name=DEFAULT_RUN_NAME,
    comp_network_names=DEFAULT_COMP_NETWORK_NAMES,
    TR=DEFAULT_TR,
    tc_detrend=DEFAULT_TC_DETREND,
    tc_despike=DEFAULT_TC_DESPIKE,
    tc_filter=DEFAULT_TC_FILTER,
    tc_covariates_filesList=DEFAULT_TC_COVARIATES_FILES_LIST,
    tc_covariates_file_numbers=DEFAULT_TC_COVARIATES_FILE_NUMBERS,
    method=DEFAULT_METHOD,
    wsize=DEFAULT_WSIZE,
    window_alpha=DEFAULT_WINDOW_ALPHA,
    num_repetitions=DEFAULT_NUM_REPETITIONS,
    num_clusters=DEFAULT_NUM_CLUSTERS,
    ica_num_comps=DEFAULT_NUM_COMPS,
    ica_algorithm=DEFAULT_ICA_ALGORITHM,
    ica_num_ica_runs=DEFAULT_ICA_NUM_ICA_RUNS,
    regressCovFile=DEFAULT_REGRESS_COV_FILE,
    kmeans_max_iter=DEFAULT_KMEANS_MAX_ITER,
    dmethod=DEFAULT_DMETHOD,
    display_results=DEFAULT_DISPLAY_RESULTS,
    **kwargs
):
    """
    Wrapper for initializing GIFT nipype interface to run dynamic FNC.
    Args:
        ica_param_file      (Str)   :   Enter fullfile path of the ICA parameter file
        out_dir             (Str)   :   Enter fullfile path of the results directory
        run_name            (Str)   :   Name of the analysis run
        comp_network_names  (Dict)  :   dictionary containing network names and network values
        TR                  (Float) :   Enter experimental TR in seconds
        tc_detrend          (Int)   :   Detrend number used to remove the trends in timecourses. Options are 0, 1, 2 and 3.
        tc_despike          (Str)   :   Remove spikes from the timecourses. Options are 'yes' and 'no'.
        tc_filter           (Float) :   High frequency cutoff
        tc_covariates_filesList (List)  :   List of covariate files to include. Leave empty if you want to select all.
        tc_covariates_file_numbers (List)  :   Enter scan numbers to include. Leave empty if you want to select all.
        wsize               (Int)   :   Window size (number of scans)
        window_alpha        (Float) :   Gaussian Window alpha value.
        num_repetitions     (Int)   :   No. of repetitions (L1 regularisation).
        num_clusters        (Int)   :   Number of KMeans clusters to use for dFNC
        ica_num_comps       (Int)   :   Number of ICA components
        ica_algorithm       (Int)   :   ICA Algorithm used in ICA estimation
        ica_num_ica_runs    (Int)   :   Number of ICA runs to perform
        regressCovFile      (Str)   :   'yes' or 'no' to regress out covariates
        kmeans_max_iter     (Int)   :   Maximum number of KMeans iterations
        dmethod             (Str)   :   Distance Method ('city', 'sqEuclid',...)
        display_results     (Int)   :   0 - No display, 1 - HTML report, 2 - PDF
    Args (not supported here, but available for nipype):
        Regularisation      (Str)   :   Options are 'none' and 'L1'.
    """
    out_dir = os.path.join(out_dir, run_name)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    gift.DFNCCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

    gc = gift.DFNCCommand()
    gc.inputs.ica_param_file = ica_param_file
    gc.inputs.out_dir = out_dir
    gc.inputs.comp_network_names = resolve_comp_network_names(
        ica_num_comps, comp_network_names
    )
    gc.inputs.TR = TR
    dfnc_params = dict(
        tc_detrend=tc_detrend,
        tc_covariates=dict(
            file_numbers=tc_covariates_file_numbers, filesList=tc_covariates_filesList
        ),
        tc_despike=tc_despike,
        tc_filter=tc_filter,
        method=method,
        wsize=wsize,
        window_alpha=window_alpha,
        num_repetitions=num_repetitions,
    )
    postprocess = dict(
        num_clusters=num_clusters,
        ica=dict(
            algorithm=ica_algorithm,
            num_comps=ica_num_comps,
            num_ica_runs=ica_num_ica_runs,
        ),
        regressCovFile=regressCovFile,
        kmeans_max_iter=kmeans_max_iter,
        dmethod=dmethod,
        display_results=display_results,
    )
    gc.inputs.dfnc_params = dfnc_params
    gc.inputs.postprocess = postprocess

    return gc.run()


def gift_mancova(
    ica_param_file=DEFAULT_ICA_PARAM_FILE,
    out_dir=DEFAULT_OUT_DIR,
    run_name=DEFAULT_RUN_NAME,
    comp_network_names=DEFAULT_COMP_NETWORK_NAMES,
    TR=DEFAULT_TR,
    features=DEFAULT_FEATURES,
    covariates=DEFAULT_COVARIATES,
    interactions=DEFAULT_INTERACTIONS,
    numOfPCs=DEFAULT_NUM_COMPS,
    feature_params=DEFAULT_FEATURE_PARAMS,
    p_threshold=DEFAULT_P_THRESHOLD,
    univariate_tests=None,
):
    gift.MancovanCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

    gc = gift.MancovanCommand()
    gc_ica_param_file=list()
    gc_ica_param_file.append(ica_param_file)
    gc.inputs.ica_param_file = gc_ica_param_file
    gc.inputs.out_dir = out_dir
    gc.inputs.comp_network_names = comp_network_names
    gc.inputs.TR = TR
    gc.inputs.features = features
    gc.inputs.covariates = covariates
    gc.inputs.interactions = interactions
    gc.inputs.numOfPCs = numOfPCs
    gc.inputs.feature_params = feature_params
    gc.inputs.p_threshold = p_threshold
    gc.inputs.display = {
        "freq_limits": [0.1, 0.15],
        "structFile": "/app/groupicatv4.0b/icatb/src/icatb_templates/ch2bet.nii",
        "t_threshold": 1.0,
        "image_values": "positive",
        "threshdesc": "fdr",
    }
    if univariate_tests is not None:
        gc.inputs.univariate_tests = univariate_tests
    return gc.run()


def gift_patch(**kwargs):
    gica_result = gift_gica(**kwargs)
    out_dir = gica_result.inputs["out_dir"]
    nc = gica_result.inputs["dim"]
    alg = ICA_ALGORITHMS[gica_result.inputs["algoType"]]
    param_file = os.path.join(out_dir, "gica_cmd_ica_parameter_info.mat")
    kwargs["ica_param_file"] = param_file
    kwargs["ica_algorithm"] = alg
    kwargs["out_dir"] = out_dir
    dfnc_result = gift_dfnc(**kwargs)


def get_interpolated_nifti(template_filename, input_filename, destination_dir="/out"):
    """
        Get an interpolated version of an file which is interpolated to match a reference.
        First, check if interpolated dimensions of nifti files match, if so, just return the input_filename.
        Else, if an interpolated version of the file has been created and saved in the root directory before, return its filename,
            else, create the interpolated version, and return its filename.
        Args:
            template_filename - the filename which has the desired spatial dimension
            input_filename - the filename to be interpolated
        Template for interpolated filenames example:
            input_filename = ' example.nii ' has dimension 53 x 63 x 52
            template_filename = 'template.nii' has dimension 53 x 63 x 46
            output_filename = 'example_INTERP_53_63_46.nii' has dimension 53 x 63 x 46
    """

    base_dir = os.path.dirname(input_filename)
    input_prefix, input_ext = os.path.splitext(input_filename)
    template_img = nib.load(template_filename)
    input_img = nib.load(input_filename)
    template_img = template_img.slicer[:, :, :, : input_img.shape[3]]
    template_dim = template_img.shape

    if input_img.shape == template_dim:
        return input_filename

    output_filename = os.path.join(
        base_dir,
        "%s_INTERP_%d_%d_%d.nii"
        % (
            input_prefix,
            template_img.shape[0],
            template_img.shape[1],
            template_img.shape[2],
        ),
    )

    if os.path.exists(output_filename):
        return output_filename

    output_img = resample_from_to(input_img, template_img)
    if destination_dir is not None:
        output_filename = os.path.join(
            destination_dir, os.path.basename(output_filename)
        )
    nib.save(output_img, output_filename)

    return output_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algorithm",
        help="the algorithm to use: gica, dfnc, or both in sequence",
    )
    parser.add_argument("-i", "--infiles", help="input files, comma separated")
    parser.add_argument("-o", "--outfile", help="output directory")
    parser.add_argument("-j", "--json", help="additional json arguments")
    args = parser.parse_args()
    config.update_config(
        {"logging": {"log_directory": args.outfile, "log_to_file": True}}
    )
    logging.update_logging(config)
    algorithm = args.algorithm
    json_args = {"out_dir": args.outfile}
    if args.infiles is not None:
        json_args["in_files"] = args.infiles.split(",")
    params = json.load(open(args.json, "r"))
    for key, val in params.items():
        json_args[key] = val
    if algorithm == "gica":
        # Do Group ICA
        gift_gica(**json_args)
    elif algorithm == "dfnc":
        # Do only dFNC
        gift_dfnc(**json_args)
    elif algorithm == "mancova":
        gift_mancova(**json_args)
    else:
        # Do both
        gift_patch(**json_args)
