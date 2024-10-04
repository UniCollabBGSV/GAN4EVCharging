import argparse
import yaml
from yaml import Loader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    parser = argparse.ArgumentParser(description="T-SNE distribution of attack and adversarial attack/normal charging sessions")
    parser.add_argument('dataset',
                        help="Data type",
                        type=str)
    parser.add_argument('strategy',
                        help="Sampling strategy",
                        type=str)
    parser.add_argument("-b", "--base", action='store_true',
                        help="Display attack/normal charging sessions")
    parser.add_argument("-s", "--save", action='store_true',
                        help="Save figure")
    parser.add_argument("-d", "--display", action='store_true',
                        help="Display ROC Curve figure")
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
    strategy_ = "attack" if strategy == "normal" else "normal"
    print(Style.BRIGHT + Fore.BLUE + f"T-SNE DISTRIBUTION IN {dataset.upper()} DATASETS USING {strategy.upper()} SAMPLING STRATEGY".center(120, "-"))

    data_path = dictionary['LSTMGAN']['data'][dataset]
    df1 = pd.read_csv(data_path[0])
    df2 = pd.read_csv(data_path[1])
    df = pd.concat([df1, df2], ignore_index=True)
    print()
    print(Style.BRIGHT + Fore.BLUE + "PREPROCESS DATASETS".center(120, "-"))
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
    print(Style.BRIGHT + Fore.BLUE + "LOADING GENERATOR".center(120, "-"))

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

    X_d_lof = data_processing_after(gen_adv_lof, LEN_SESSIONS, sids, scaler, df)
    X_d_mlp = data_processing_after(gen_adv_mlp, LEN_SESSIONS, sids, scaler, df)
    X_d_rnd = data_processing_after(gen_adv_rnd, LEN_SESSIONS, sids, scaler, df)
    X_base = data_processing_after(X_ff, LEN_SESSIONS, sids, scaler, df)

    X_d_lof_1 = create_prediction_file_1(X_d_lof, dataset)
    X_d_lof_5 = create_prediction_file_5(X_d_lof, dataset)
    X_d_mlp_1 = create_prediction_file_1(X_d_mlp, dataset)
    X_d_mlp_5 = create_prediction_file_5(X_d_mlp, dataset)
    X_d_rnd_1 = create_prediction_file_1(X_d_rnd, dataset)
    X_d_rnd_5 = create_prediction_file_5(X_d_rnd, dataset)
    X_base_1 = create_prediction_file_1(X_base, dataset)
    X_base_5 = create_prediction_file_5(X_base, dataset)

    data_nov_lof = extract_features_from_raw_data(df_base_file=X_d_lof, df_pred_single=X_d_lof_1, df_pred_part_5=X_d_lof_5, CONFIG=dataset, do_clf='LocalOutlierFactor')
    data_clf_mlp = extract_features_from_raw_data(df_base_file=X_d_mlp, df_pred_single=X_d_mlp_1, df_pred_part_5=X_d_mlp_5, CONFIG=dataset, do_clf='MLPClassifier')
    data_clf_rnd = extract_features_from_raw_data(df_base_file=X_d_rnd, df_pred_single=X_d_rnd_1, df_pred_part_5=X_d_rnd_5, CONFIG=dataset, do_clf='MLPClassifier')
    data_clf_base = extract_features_from_raw_data(df_base_file=X_base, df_pred_single=X_base_1, df_pred_part_5=X_base_5, CONFIG=dataset, do_clf='MLPClassifier')

    tsne = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, random_state=42))])
    base = tsne.fit_transform(data_clf_base)
    mlp = tsne.fit_transform(data_clf_mlp)
    rnd = tsne.fit_transform(data_clf_rnd)
    lof = tsne.fit_transform(data_nov_lof)

    def plot_distribution(data, color, model):
        plt.figure(figsize=(10, 6))
        label = f"Adversarial {strategy} {model} charging sessions"
        plt.scatter(base[:, 0], base[:, 1], label=f"{strategy.capitalize()}" + " charging sessions", c="crimson" if strategy == "attack" else "blue", edgecolor='face', s=50)
        plt.scatter(data[:, 0], data[:, 1], label=label, c=color, edgecolor='face', s=50)
        plt.xlabel('T-SNE feature 1')
        plt.ylabel('T-SNE feature 2')
        plt.legend(loc='lower right')
        plt.title(f"T-SNE distribution of {strategy} and adversarial {strategy} charging sessions")
        name = f"T_SNE_{strategy}_and_adversarial_{strategy}_{model.replace(" ", "_")}_charging_sessions_in_{dataset}"

        if args.save:
            plt.savefig(f'results//{name}.pdf', format='pdf', dpi=1200)

    spinner = Halo(text=Style.BRIGHT + Fore.RED + "PROCESSING ROC CURVE", spinner='dots2')
    spinner.start()

    plot_distribution(lof, color="hotpink", model="LocalOutlierFactor")
    plot_distribution(mlp, color="orange", model="MLPClassifier and Ensemble method")
    plot_distribution(rnd, color="green", model="RandomForestClassifier")

    if args.base:
        X_ff_ = X_d_attack if strategy == 'normal' else X_d_normal
        X_base_ = data_processing_after(X_ff_, LEN_SESSIONS, sids, scaler, df)
        X_base_1_ = create_prediction_file_1(X_base_, dataset)
        X_base_5_ = create_prediction_file_5(X_base_, dataset)
        data_clf_base_ = extract_features_from_raw_data(df_base_file=X_base_, df_pred_single=X_base_1_, df_pred_part_5=X_base_5_, CONFIG=dataset, do_clf='MLPClassifier')
        base_ = tsne.fit_transform(data_clf_base_)

        plt.figure(figsize=(10, 6))
        plt.scatter(base[:, 0], base[:, 1], label=f"{strategy.capitalize()}" + " charging sessions", c="crimson" if strategy=="attack" else "blue", edgecolor='face', s=50)
        plt.scatter(base_[:, 0], base_[:, 1], label=f"{strategy_.capitalize()}" + " charging sessions", c="crimson" if strategy_=="attack" else "blue", edgecolor='face', s=50)
        plt.xlabel('T-SNE feature 1')
        plt.ylabel('T-SNE feature 2')
        plt.legend(loc='lower right')
        plt.title("T-SNE distribution of attack and normal charging sessions")
        name = f"T_SNE_attack_and_normal_charging_session_in_{dataset}"

        if args.save:
            plt.savefig(f'results//{name}.pdf', format='pdf', dpi=1200)

    if args.display:
        plt.show()

    spinner.succeed(Style.BRIGHT + Fore.RED + "DRAWN SUCCESSFULLY!")
