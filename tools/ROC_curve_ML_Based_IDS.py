import argparse
import yaml
from yaml import Loader
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import pyfiglet
from halo import Halo
from colorama import Fore, init, Style

from src.model import Generator, Scaler
from src.utils import find_sid, create_prediction_file_1, create_prediction_file_5, data_processing_after, data_processing_before, extract_features_from_raw_data

np.random.seed(1510)
torch.manual_seed(1510)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate ROC Curve before and after using synthetic data")
    parser.add_argument('dataset',
                        help="Data type",
                        type=str)
    parser.add_argument('strategy',
                        help="Sampling strategy",
                        type=str)
    parser.add_argument("-s", "--save", action='store_true',
                        help="Save figure")
    parser.add_argument("-d", "--display", action='store_true',
                        help="Display data distribution")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    stream = open('configs//configs.yaml')
    dictionary = yaml.load(stream, Loader=Loader)

    init(autoreset=True)
    ascii_art = pyfiglet.figlet_format("*IDS LSTM GAN*")
    banner = Fore.BLUE + ascii_art
    print(banner)

    dataset = args.dataset
    strategy = args.strategy
    print(Style.BRIGHT + Fore.BLUE + f"GENERATEING ML ROC CURVE IN {dataset.upper()} DATASETS USING {strategy.upper()} SAMPLING STRATEGY".center(120, "-"))

    data_path = dictionary['LSTMGAN']['data'][dataset]
    df1 = pd.read_csv(data_path[0])
    df2 = pd.read_csv(data_path[1])
    df = pd.concat([df1, df2], ignore_index=True)
    print()
    print(Style.BRIGHT + Fore.BLUE + "DATASETS".center(120, "-"))
    print()
    print(Style.BRIGHT + Fore.CYAN + f" - {data_path[0]}")
    print(Style.BRIGHT + Fore.CYAN + f" - {data_path[1]}")

    print()
    print(Style.BRIGHT + Fore.BLUE + "CREATE SCALER".center(120, "-"))
    scaler = Scaler(df)

    sids = find_sid(df)
    LEN_SESSIONS = dictionary['LSTMGAN']['parammeters']['len_session']
    LATENT_DIM = dictionary['LSTMGAN']['parammeters']['latent_dim']

    print()
    print(Style.BRIGHT + Fore.BLUE + "PREPROCESS DATASETS".center(120, "-"))
    data, labels = data_processing_before(df, LEN_SESSIONS, scaler)

    data_normal = data[labels == 1]
    label_normal = labels[labels == 1]
    data_attack = data[labels == 0]
    label_attack = labels[labels == 0]

    print()
    print(Style.BRIGHT + Fore.BLUE + "LOADING ML-BASED IDS AND GENERATOR".center(120, "-"))
    ml_paths = dictionary['ML_Based_IDS']['models'][dataset]
    lof = joblib.load(ml_paths[0])
    mlp = joblib.load(ml_paths[3])
    rnd = joblib.load(ml_paths[6])

    generator_mlp = Generator(latent_dim=LATENT_DIM, len_session=LEN_SESSIONS).to(device)
    generator_rnd = Generator(latent_dim=LATENT_DIM, len_session=LEN_SESSIONS).to(device)
    generator_lof = Generator(latent_dim=LATENT_DIM, len_session=LEN_SESSIONS).to(device)

    generator_path = dictionary['LSTMGAN']['generator'][dataset][strategy]
    generator_lof.load_state_dict(torch.load(generator_path[0]))
    generator_mlp.load_state_dict(torch.load(generator_path[1]))
    generator_rnd.load_state_dict(torch.load(generator_path[2]))

    X_d_attack = data_attack.to(device)
    X_d_normal = data_normal[:len(data_attack)].to(device)
    y_d_attack = label_attack
    y_d_normal = label_normal[:len(label_attack)]

    print()
    print(Style.BRIGHT + Fore.BLUE + "GENERATING ADVERSARIAL CHARGING SESSIONS".center(120, "-"))
    X_ff = X_d_attack if strategy == 'attack' else X_d_normal
    charge_speed = X_ff[:, :, 1:2].to(device)
    noise = torch.randn((X_d_attack.shape[0], LATENT_DIM)).to(device)
    gen_adv_lof = generator_lof(noise)
    gen_adv_mlp = generator_mlp(noise)
    gen_adv_rnd = generator_rnd(noise)

    gen_adv_lof = torch.cat((gen_adv_lof, charge_speed), dim=-1)[:, :, [0, 2, 1]]
    gen_adv_mlp = torch.cat((gen_adv_mlp, charge_speed), dim=-1)[:, :, [0, 2, 1]]
    gen_adv_rnd = torch.cat((gen_adv_rnd, charge_speed), dim=-1)[:, :, [0, 2, 1]]

    synthetic_data_lof = torch.cat((X_d_normal, gen_adv_lof), dim=0)
    synthetic_data_mlp = torch.cat((X_d_normal, gen_adv_mlp), dim=0)
    synthetic_data_rnd = torch.cat((X_d_normal, gen_adv_rnd), dim=0)
    X_d = torch.cat((X_d_normal, X_d_attack), dim=0)
    y_d = torch.cat((y_d_normal, y_d_attack), dim=0).view(-1, 1)

    X_d_lof = data_processing_after(synthetic_data_lof, LEN_SESSIONS, sids, scaler, df)
    X_d_mlp = data_processing_after(synthetic_data_mlp, LEN_SESSIONS, sids, scaler, df)
    X_d_rnd = data_processing_after(synthetic_data_rnd, LEN_SESSIONS, sids, scaler, df)
    X_d = data_processing_after(X_d, LEN_SESSIONS, sids, scaler, df)

    X_d_1 = create_prediction_file_1(X_d, dataset)
    X_d_5 = create_prediction_file_5(X_d, dataset)
    X_d_lof_1 = create_prediction_file_1(X_d_lof, dataset)
    X_d_lof_5 = create_prediction_file_5(X_d_lof, dataset)
    X_d_mlp_1 = create_prediction_file_1(X_d_mlp, dataset)
    X_d_mlp_5 = create_prediction_file_5(X_d_mlp, dataset)
    X_d_rnd_1 = create_prediction_file_1(X_d_rnd, dataset)
    X_d_rnd_5 = create_prediction_file_5(X_d_rnd, dataset)

    data_clf = extract_features_from_raw_data(df_base_file=X_d, df_pred_single=X_d_1, df_pred_part_5=X_d_5, CONFIG=dataset, do_clf='MLPClassifier')
    data_nov = extract_features_from_raw_data(df_base_file=X_d, df_pred_single=X_d_1, df_pred_part_5=X_d_5, CONFIG=dataset, do_clf='LocalOutlierFactor')
    data_nov_lof = extract_features_from_raw_data(df_base_file=X_d_lof, df_pred_single=X_d_lof_1, df_pred_part_5=X_d_lof_5, CONFIG=dataset, do_clf='LocalOutlierFactor')
    data_clf_mlp = extract_features_from_raw_data(df_base_file=X_d_mlp, df_pred_single=X_d_mlp_1, df_pred_part_5=X_d_mlp_5, CONFIG=dataset, do_clf='MLPClassifier')
    data_clf_rnd = extract_features_from_raw_data(df_base_file=X_d_rnd, df_pred_single=X_d_rnd_1, df_pred_part_5=X_d_rnd_5, CONFIG=dataset, do_clf='MLPClassifier')

    spinner = Halo(text=Style.BRIGHT + Fore.RED + "PROCESSING ROC CURVE", spinner='dots2')
    spinner.start()

    mlp_score = mlp.predict_proba(data_clf[list(c for c in data_clf.columns if c != "_y" and c != "Unnamed: 0")])[:, 1]
    rnd_score = rnd.predict_proba(data_clf[list(c for c in data_clf.columns if c != "_y" and c != "Unnamed: 0")])[:, 1]
    lof_score = lof.decision_function(data_nov[list(c for c in data_nov.columns if c != "_y" and c != "Unnamed: 0")])

    mlp_adv_score = mlp.predict_proba(data_clf_mlp[list(c for c in data_clf_mlp.columns if c != "_y" and c != "Unnamed: 0")])[:, 1]
    rnd_adv_score = rnd.predict_proba(data_clf_rnd[list(c for c in data_clf_rnd.columns if c != "_y" and c != "Unnamed: 0")])[:, 1]
    lof_adv_score = lof.decision_function(data_nov_lof[list(c for c in data_nov_lof.columns if c != "_y" and c != "Unnamed: 0")])

    fpr_mlp, tpr_mlp, _ = roc_curve(y_d, mlp_score)
    fpr_rnd, tpr_rnd, _ = roc_curve(y_d, rnd_score)
    fpr_lof, tpr_lof, _ = roc_curve(y_d, lof_score)

    fpr_adv_mlp, tpr_adv_mlp, _ = roc_curve(y_d, mlp_adv_score)
    fpr_adv_rnd, tpr_adv_rnd, _ = roc_curve(y_d, rnd_adv_score)
    fpr_adv_lof, tpr_adv_lof, _ = roc_curve(y_d, lof_adv_score)

    roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
    roc_auc_rnd = auc(fpr_rnd, tpr_rnd)
    roc_auc_lof = auc(fpr_lof, tpr_lof)

    roc_auc_adv_mlp = auc(fpr_adv_mlp, tpr_adv_mlp)
    roc_auc_adv_rnd = auc(fpr_adv_rnd, tpr_adv_rnd)
    roc_auc_adv_lof = auc(fpr_adv_lof, tpr_adv_lof)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_mlp, tpr_mlp, color='blue', lw=2, label=f"MLPClassifier w/o generated attacks (AUC = {roc_auc_mlp:.2f})")
    plt.plot(fpr_rnd, tpr_rnd, color='green', lw=2, label=f"RandomForestClassifier w/o generated attacks (AUC = {roc_auc_rnd:.2f})")
    plt.plot(fpr_lof, tpr_lof, color='red', lw=2, label=f"LocalOutlierFactor w/o generated attacks (AUC = {roc_auc_lof:.2f})")

    plt.plot(fpr_adv_mlp, tpr_adv_mlp, color='blue', linestyle=':', lw=2, label=f"MLPClassifier w/ synthetic attacks using {strategy} sampling strategy (AUC = {roc_auc_adv_mlp:.2f})")
    plt.plot(fpr_adv_rnd, tpr_adv_rnd, color='green', linestyle=':', lw=2, label=f"RandomForestClassifier w/ synthetic attack using {strategy} sampling strategy (AUC = {roc_auc_adv_rnd:.2f})")
    plt.plot(fpr_adv_lof, tpr_adv_lof, color='red', linestyle=':', lw=2, label=f"LocalOutlierFactor w/ synthetic attack using {strategy} sampling strategy (AUC = {roc_auc_adv_lof:.2f})")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    if args.save:
        plt.savefig(f"results//{dataset}_ROC_using_{strategy}_sampling_strategy.pdf", format='pdf', dpi=1200)

    if args.display:
        plt.show()
    spinner.succeed(Style.BRIGHT + Fore.RED + "DRAWN SUCCESSFULLY!")