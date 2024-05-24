# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on November 01, 2023 07:43:28 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowPrtoc(mfpackage.MFPackage):
    """
    ModflowPrtoc defines a oc package within a prt6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    budget_filerecord : [budgetfile]
        * budgetfile (string) name of the output file to write budget
          information.
    budgetcsv_filerecord : [budgetcsvfile]
        * budgetcsvfile (string) name of the comma-separated value (CSV) output
          file to write budget summary information. A budget summary record
          will be written to this file for each time step of the simulation.
    concentration_filerecord : [concentrationfile]
        * concentrationfile (string) name of the output file to write conc
          information.
    concentrationprintrecord : [columns, width, digits, format]
        * columns (integer) number of columns for writing data.
        * width (integer) width for writing each number.
        * digits (integer) number of digits to use for writing a number.
        * format (string) write format can be EXPONENTIAL, FIXED, GENERAL, or
          SCIENTIFIC.
    trackevent : string
        * trackevent (string) particle tracking event(s) to include in track
          output files. Can be ALL, RELEASE, TRANSIT, TIMESTEP, TERMINATE, or
          WEAKSINK. RELEASE selects particle releases. TRANSIT selects cell-to-
          cell transitions. TIMESTEP selects transitions between timesteps.
          TERMINATE selects particle terminations. WEAKSINK selects particle
          exits from weak sink cells. Events may coincide with other events.
    track_filerecord : [trackfile]
        * trackfile (string) name of the output file to write tracking
          information.
    trackcsv_filerecord : [trackcsvfile]
        * trackcsvfile (string) name of the comma-separated value (CSV) file to
          write tracking information.
    saverecord : [rtype, ocsetting]
        * rtype (string) type of information to save or print. Can be BUDGET or
          CONCENTRATION.
        * ocsetting (keystring) specifies the steps for which the data will be
          saved.
            all : [keyword]
                * all (keyword) keyword to indicate save for all time steps in
                  period.
            first : [keyword]
                * first (keyword) keyword to indicate save for first step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            last : [keyword]
                * last (keyword) keyword to indicate save for last step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            frequency : [integer]
                * frequency (integer) save at the specified time step
                  frequency. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            steps : [integer]
                * steps (integer) save for each step specified in STEPS. This
                  keyword may be used in conjunction with other keywords to
                  print or save results for multiple time steps.
    printrecord : [rtype, ocsetting]
        * rtype (string) type of information to save or print. Can be BUDGET or
          CONCENTRATION.
        * ocsetting (keystring) specifies the steps for which the data will be
          saved.
            all : [keyword]
                * all (keyword) keyword to indicate save for all time steps in
                  period.
            first : [keyword]
                * first (keyword) keyword to indicate save for first step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            last : [keyword]
                * last (keyword) keyword to indicate save for last step in
                  period. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            frequency : [integer]
                * frequency (integer) save at the specified time step
                  frequency. This keyword may be used in conjunction with other
                  keywords to print or save results for multiple time steps.
            steps : [integer]
                * steps (integer) save for each step specified in STEPS. This
                  keyword may be used in conjunction with other keywords to
                  print or save results for multiple time steps.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have 
        a mfgwflak package parent_file.

    """
    budget_filerecord = ListTemplateGenerator(('prt6', 'oc', 'options',
                                               'budget_filerecord'))
    budgetcsv_filerecord = ListTemplateGenerator(('prt6', 'oc',
                                                  'options',
                                                  'budgetcsv_filerecord'))
    concentration_filerecord = ListTemplateGenerator((
        'prt6', 'oc', 'options', 'concentration_filerecord'))
    concentrationprintrecord = ListTemplateGenerator((
        'prt6', 'oc', 'options', 'concentrationprintrecord'))
    track_filerecord = ListTemplateGenerator(('prt6', 'oc', 'options',
                                              'track_filerecord'))
    trackcsv_filerecord = ListTemplateGenerator(('prt6', 'oc', 'options',
                                                 'trackcsv_filerecord'))
    saverecord = ListTemplateGenerator(('prt6', 'oc', 'period',
                                        'saverecord'))
    printrecord = ListTemplateGenerator(('prt6', 'oc', 'period',
                                         'printrecord'))
    package_abbr = "prtoc"
    _package_type = "oc"
    dfn_file_name = "prt-oc.dfn"

    dfn = [
           ["header", ],
           ["block options", "name budget_filerecord",
            "type record budget fileout budgetfile", "shape", "reader urword",
            "tagged true", "optional true"],
           ["block options", "name budget", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name fileout", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name budgetfile", "type string",
            "preserve_case true", "shape", "in_record true", "reader urword",
            "tagged false", "optional false"],
           ["block options", "name budgetcsv_filerecord",
            "type record budgetcsv fileout budgetcsvfile", "shape",
            "reader urword", "tagged true", "optional true"],
           ["block options", "name budgetcsv", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name budgetcsvfile", "type string",
            "preserve_case true", "shape", "in_record true", "reader urword",
            "tagged false", "optional false"],
           ["block options", "name concentration_filerecord",
            "type record concentration fileout concentrationfile", "shape",
            "reader urword", "tagged true", "optional true"],
           ["block options", "name concentration", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name concentrationfile", "type string",
            "preserve_case true", "shape", "in_record true", "reader urword",
            "tagged false", "optional false"],
           ["block options", "name concentrationprintrecord",
            "type record concentration print_format formatrecord", "shape",
            "reader urword", "optional true"],
           ["block options", "name print_format", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name formatrecord",
            "type record columns width digits format", "shape",
            "in_record true", "reader urword", "tagged", "optional false"],
           ["block options", "name trackevent", "type string", "shape",
            "reader urword", "tagged true", "optional true"],
           ["block options", "name track_filerecord",
            "type record track fileout trackfile", "shape", "reader urword",
            "tagged true", "optional true"],
           ["block options", "name track", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name trackfile", "type string",
            "preserve_case true", "shape", "in_record true", "reader urword",
            "tagged false", "optional false"],
           ["block options", "name trackcsv_filerecord",
            "type record trackcsv fileout trackcsvfile", "shape",
            "reader urword", "tagged true", "optional true"],
           ["block options", "name trackcsv", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block options", "name trackcsvfile", "type string",
            "preserve_case true", "shape", "in_record true", "reader urword",
            "tagged false", "optional false"],
           ["block options", "name columns", "type integer", "shape",
            "in_record true", "reader urword", "tagged true", "optional"],
           ["block options", "name width", "type integer", "shape",
            "in_record true", "reader urword", "tagged true", "optional"],
           ["block options", "name digits", "type integer", "shape",
            "in_record true", "reader urword", "tagged true", "optional"],
           ["block options", "name format", "type string", "shape",
            "in_record true", "reader urword", "tagged false",
            "optional false"],
           ["block period", "name iper", "type integer",
            "block_variable True", "in_record true", "tagged false", "shape",
            "valid", "reader urword", "optional false"],
           ["block period", "name saverecord",
            "type record save rtype ocsetting", "shape", "reader urword",
            "tagged false", "optional true"],
           ["block period", "name save", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block period", "name printrecord",
            "type record print rtype ocsetting", "shape", "reader urword",
            "tagged false", "optional true"],
           ["block period", "name print", "type keyword", "shape",
            "in_record true", "reader urword", "tagged true",
            "optional false"],
           ["block period", "name rtype", "type string", "shape",
            "in_record true", "reader urword", "tagged false",
            "optional false"],
           ["block period", "name ocsetting",
            "type keystring all first last frequency steps", "shape",
            "tagged false", "in_record true", "reader urword"],
           ["block period", "name all", "type keyword", "shape",
            "in_record true", "reader urword"],
           ["block period", "name first", "type keyword", "shape",
            "in_record true", "reader urword"],
           ["block period", "name last", "type keyword", "shape",
            "in_record true", "reader urword"],
           ["block period", "name frequency", "type integer", "shape",
            "tagged true", "in_record true", "reader urword"],
           ["block period", "name steps", "type integer", "shape (<nstp)",
            "tagged true", "in_record true", "reader urword"]]

    def __init__(self, model, loading_package=False, budget_filerecord=None,
                 budgetcsv_filerecord=None, concentration_filerecord=None,
                 concentrationprintrecord=None, trackevent=None,
                 track_filerecord=None, trackcsv_filerecord=None,
                 saverecord=None, printrecord=None, filename=None, pname=None,
                 **kwargs):
        super().__init__(model, "oc", filename, pname,
                         loading_package, **kwargs)

        # set up variables
        self.budget_filerecord = self.build_mfdata("budget_filerecord",
                                                   budget_filerecord)
        self.budgetcsv_filerecord = self.build_mfdata("budgetcsv_filerecord",
                                                      budgetcsv_filerecord)
        self.concentration_filerecord = self.build_mfdata(
            "concentration_filerecord", concentration_filerecord)
        self.concentrationprintrecord = self.build_mfdata(
            "concentrationprintrecord", concentrationprintrecord)
        self.trackevent = self.build_mfdata("trackevent", trackevent)
        self.track_filerecord = self.build_mfdata("track_filerecord",
                                                  track_filerecord)
        self.trackcsv_filerecord = self.build_mfdata("trackcsv_filerecord",
                                                     trackcsv_filerecord)
        self.saverecord = self.build_mfdata("saverecord", saverecord)
        self.printrecord = self.build_mfdata("printrecord", printrecord)
        self._init_complete = True
