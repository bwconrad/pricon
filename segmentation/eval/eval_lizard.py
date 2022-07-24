import argparse
import os

import numpy as np
import pandas as pd

from .utils import binarize, get_fast_pq, remap_label


def calculate_pq(true_path, pred_path, save_path, **kwargs):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # load the data
    true = np.load(true_path)
    pred = np.load(pred_path)

    mPQ_all = []
    bPQ_all = []

    # loop over the images
    for i in range(true.shape[0]):
        pq = []
        pred_bin = binarize(pred[i, :, :, 1:])
        true_bin = binarize(true[i, :, :, 1:])

        if len(np.unique(true_bin)) == 1:
            pq_bin = (
                np.nan
            )  # if ground truth is empty for that class, skip from calculation
        else:
            [_, _, pq_bin], _ = get_fast_pq(true_bin, pred_bin)  # compute PQ

        # loop over the classes
        for j in range(6):
            pred_tmp = pred[i, :, :, j + 1]
            pred_tmp = pred_tmp.astype("int32")
            true_tmp = true[i, :, :, j + 1]
            true_tmp = true_tmp.astype("int32")
            pred_tmp = remap_label(pred_tmp)
            true_tmp = remap_label(true_tmp)

            if len(np.unique(true_tmp)) == 1:
                pq_tmp = (
                    np.nan
                )  # if ground truth is empty for that class, skip from calculation
            else:
                [_, _, pq_tmp], _ = get_fast_pq(true_tmp, pred_tmp)  # compute PQ

            pq.append(pq_tmp)

        mPQ_all.append(pq)
        bPQ_all.append([pq_bin])

    # using np.nanmean skips values with nan from the mean calculation
    mPQ_each_image = [np.nanmean(pq) for pq in mPQ_all]
    bPQ_each_image = [np.nanmean(pq_bin) for pq_bin in bPQ_all]

    # class metric
    epi_PQ = np.nanmean([pq[0] for pq in mPQ_all])
    lymph_PQ = np.nanmean([pq[1] for pq in mPQ_all])
    pla_PQ = np.nanmean([pq[2] for pq in mPQ_all])
    neut_PQ = np.nanmean([pq[3] for pq in mPQ_all])
    eos_PQ = np.nanmean([pq[4] for pq in mPQ_all])
    conn_PQ = np.nanmean([pq[5] for pq in mPQ_all])

    # Print for each class
    print("Printing calculated metrics on a single split")
    print("-" * 40)
    print("epithelial PQ: {}".format(epi_PQ))
    print("lymphocyte PQ: {}".format(lymph_PQ))
    print("plasma PQ: {}".format(pla_PQ))
    print("neutrophil PQ: {}".format(neut_PQ))
    print("eosinophil PQ: {}".format(eos_PQ))
    print("connective PQ: {}".format(conn_PQ))
    print("-" * 40)

    class_names = [
        "epithelial",
        "lymphocyte",
        "plasma",
        "neutrophil",
        "eosinophil",
        "connective",
    ]
    # Save per-class metrics as a csv file
    for_dataframe = {
        "Class Name": class_names,
        "PQ": [epi_PQ, lymph_PQ, pla_PQ, neut_PQ, eos_PQ, conn_PQ],
    }
    df = pd.DataFrame(for_dataframe, columns=["Tissue name", "PQ"])
    df.to_csv(save_path + "/class_stats.csv")

    # Print for each tissue
    all_tissue_mPQ = []
    all_tissue_bPQ = []
    for tissue_name in ["colon"]:
        indices = [i for i, x in enumerate(true)]
        tissue_PQ = [mPQ_each_image[i] for i in indices]
        print("{} PQ: {} ".format(tissue_name, np.nanmean(tissue_PQ)))
        tissue_PQ_bin = [bPQ_each_image[i] for i in indices]
        print("{} PQ binary: {} ".format(tissue_name, np.nanmean(tissue_PQ_bin)))
        all_tissue_mPQ.append(np.nanmean(tissue_PQ))
        all_tissue_bPQ.append(np.nanmean(tissue_PQ_bin))

    for_dataframe = {
        "Tissue name": ["colon"],
        "PQ": all_tissue_mPQ,
        "PQ bin": all_tissue_bPQ,
    }
    print(for_dataframe)
    df = pd.DataFrame(for_dataframe, columns=["Tissue name", "PQ", "PQ bin"])
    df.to_csv(save_path + "/tissue_stats.csv")

    # Show overall metrics - mPQ is average PQ over the classes and the tissues, bPQ is average binary PQ over the tissues
    print("-" * 40)
    print("Average mPQ:{}".format(np.nanmean(all_tissue_mPQ)))
    print("Average bPQ:{}".format(np.nanmean(all_tissue_bPQ)))

    return np.nanmean(all_tissue_mPQ), np.nanmean(all_tissue_bPQ)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="results/")
    args = parser.parse_args()
    calculate_pq(
        args.true_path,
        args.pred_path,
        args.save_path,
    )
