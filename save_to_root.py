#!/usr/bin/env python3

import hist
from coffea import util
from coffea.processor import accumulate
import numpy as np
import uproot
import os

from ttgamma.utils.plotting import RebinHist, SetRangeHist, GroupBy, DictToHist

# NOTE: your timestamps will differ!
outputMC = accumulate(
    [
        util.load("Outputs/output_MCOther_run20260312_003844.coffea"),
        util.load("Outputs/output_MCSingleTop_run20260311_235355.coffea"),
        util.load("Outputs/output_MCTTbar1l_run20260312_040117.coffea"),
        util.load("Outputs/output_MCTTbar2l_run20260312_040113.coffea"),
        util.load("Outputs/output_MCTTGamma_run20260311_233804.coffea"),
        util.load("Outputs/output_MCWJets_run20260312_034301.coffea"),
        util.load("Outputs/output_MCZJets_run20260312_003608.coffea"),
    ]
)

outputData = util.load("Outputs/output_Data_run20260312_035655.coffea")

# following the definition in the processor categorizeGenPhoton() function
# 1 * isGenPho + 2 * isMisIDele + 3 * isHadPho + 4 * isHadFake
groupingCategory = {
    "NonPrompt": [3j, 4j],
    "MisID": [2j],
    "Prompt": [1j],
}

groupingMCDatasets = {
    "ZG": [
        "ZGamma_01J_5f_lowMass",
    ],
    "WG": [
        "WGamma",
    ],
    "other": [
        "TTbarPowheg_Dilepton",
        "TTbarPowheg_Semilept",
        "TTbarPowheg_Hadronic",
        "W2jets",
        "W3jets",
        "W4jets",
        "DYjetsM50",
        "ST_s_channel",
        "ST_tW_channel",
        "ST_tbarW_channel",
        "ST_tbar_channel",
        "ST_t_channel",
        "TTWtoLNu",
        "TTWtoQQ",
        "TTZtoLL",
        "GJets_HT200To400",
        "GJets_HT400To600",
        "GJets_HT600ToInf",
        "ZZ",
        "WZ",
        "WW",
        "TGJets",
    ],
    "ttgamma": [
        "TTGamma_Dilepton",
        "TTGamma_SingleLept",
        "TTGamma_Hadronic",
    ],
}

renorm_fact_scale_total = np.array(
    [1.28623179, 1.2127664, 1.0606851, 0.94312969, 0.83927955, 0.79149344]
)

s = hist.tag.Slicer()

if __name__ == "__main__":
    # Group MC histograms
    outputMC = {
        process: accumulate(outputMC[ds] for ds in datasets)
        for process, datasets in groupingMCDatasets.items()
    }
    def getHisto(name):
        return DictToHist(
            {process: histos[name] for process, histos in outputMC.items()},
            name="dataset"
        )

    # Group data histograms
    outputDataHist = accumulate([histo for key, histo in outputData.items()])

    h = getHisto("M3")
    h = h[{"lepFlavor": sum}]
    h = GroupBy(h, "category", "category", groupingCategory)
    h = h[{"M3": s[50j:550j+1:hist.rebin(4)]}]
    # h = SetRangeHist(h, "M3", 50, 550)

    hData = outputDataHist["M3"][
        {"lepFlavor": sum, "category": sum, "systematic": sum}
    ]
    hData = hData[{"M3": s[:: hist.rebin(4)]}]
    hData = SetRangeHist(hData, "M3", 50, 550)

    outdir = "RootFiles"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outputFile = uproot.recreate(os.path.join(outdir, "M3_Output.root"))

    outputFile["data_obs"] = hData

    systematics = [s for s in h.axes["systematic"] if not s.startswith("Q2Scale")]

    for _category in ["MisID", "NonPrompt"]:
        for _systematic in systematics:
            histname = (
                f"{_category}_{_systematic}"
                if (not f"{_systematic}" == "nominal")
                else f"{_category}"
            )
            outputFile[histname] = h[
                {"dataset": sum, "category": _category, "systematic": _systematic}
            ]

    for _dataset in ["ttgamma", "WG", "ZG", "other"]:
        for _systematic in systematics:
            histname = (
                f"{_dataset}_{_systematic}"
                if (not f"{_systematic}" == "nominal")
                else f"{_dataset}"
            )
            outputFile[histname] = h[
                {"dataset": _dataset, "category": "Prompt", "systematic": _systematic}
            ]

    # isolate acceptance effect and compute the max of the variations
    # TODO: this should be done also for other channels
    hnom = h[{"dataset": "ttgamma", "category": "Prompt", "systematic": "nominal"}]
    hscale = h[
        {
            "dataset": "ttgamma",
            "category": "Prompt",
            "systematic": [f"Q2Scale{i}Up" for i in range(6)],
        }
    ]
    hscale.values()[:] /= renorm_fact_scale_total
    hnom.values()[:] = hscale.values().max(axis=1)
    outputFile["ttgamma_Q2ScaleUp"] = hnom
    hnom.values()[:] = hscale.values().min(axis=1)
    outputFile["ttgamma_Q2ScaleDown"] = hnom

    outputFile.close()

    """
    nonprompt control region
    """

    # regroup for the different photon categories, summing over all data sets.

    h = getHisto("photon_chIso")
    h = h[{"lepFlavor": sum}]
    h = GroupBy(h, "category", "category", groupingCategory)

    new_bins = np.array(
        [1.1500000000000001, 2.5, 5, 9, 10]
    )  # 1.14 is in the cutbased medium ID
    chIso_axis = hist.axis.Variable(
        new_bins, name="chIso", label=r"Charged Hadron Isolation"
    )

    hData = outputDataHist["photon_chIso"][{"lepFlavor": sum}]
    hData = hData[{"category": sum, "systematic": sum}]

    outputFile = uproot.recreate(os.path.join(outdir, "Isolation_Output.root"))
    outputFile["data_obs"] = RebinHist(hData, chIso=chIso_axis)

    for _category in ["MisID", "NonPrompt"]:
        for _systematic in systematics:
            histname = (
                f"{_category}_{_systematic}"
                if (not f"{_systematic}" == "nominal")
                else f"{_category}"
            )
            outputFile[histname] = RebinHist(
                h[{"dataset": sum, "category": _category, "systematic": _systematic}],
                chIso=chIso_axis,
            )

    for _dataset in ["ttgamma", "WG", "ZG", "other"]:
        for _systematic in systematics:
            histname = (
                f"{_dataset}_{_systematic}"
                if (not f"{_systematic}" == "nominal")
                else f"{_dataset}"
            )
            outputFile[histname] = RebinHist(
                h[
                    {
                        "dataset": _dataset,
                        "category": "Prompt",
                        "systematic": _systematic,
                    }
                ],
                chIso=chIso_axis,
            )

    outputFile.close()

    """
    Mis-ID and Vgamma control region
    (electron and muon channels, respectively)
    """

    h = getHisto("photon_lepton_mass_3j0t")
    h = GroupBy(h, "category", "category", groupingCategory)
    h = h[{"mass": s[40j:200j+1:hist.rebin(10)]}]
    h = SetRangeHist(h, "mass", 40, 200)

    hData = outputDataHist["photon_lepton_mass_3j0t"]
    hData = hData[{"category": sum, "systematic": sum}]
    hData = hData[{"mass": s[:: hist.rebin(10)]}]
    hData = SetRangeHist(hData, "mass", 40, 200)

    for _lepton in ["electron", "muon"]:
        outputFile = uproot.recreate(
            os.path.join(outdir, f"MisID_Output_{_lepton}.root")
        )

        outputFile["data_obs"] = hData[{"lepFlavor": _lepton}]

        systematics = h.axes["systematic"]
        for _category in ["MisID", "NonPrompt"]:
            for _systematic in systematics:
                histname = (
                    f"{_category}_{_systematic}"
                    if (not f"{_systematic}" == "nominal")
                    else f"{_category}"
                )
                outputFile[histname] = h[
                    {
                        "category": _category,
                        "systematic": _systematic,
                        "lepFlavor": _lepton,
                        "dataset": sum,
                    }
                ]

        for _dataset in ["ttgamma", "WG", "ZG", "other"]:
            for _systematic in systematics:
                histname = (
                    f"{_dataset}_{_systematic}"
                    if (not f"{_systematic}" == "nominal")
                    else f"{_dataset}"
                )
                outputFile[histname] = h[
                    {
                        "category": "Prompt",
                        "systematic": _systematic,
                        "lepFlavor": _lepton,
                        "dataset": _dataset,
                    }
                ]

        outputFile.close()
