import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pathlib
import torch
root_path = pathlib.Path("")
parser = argparse.ArgumentParser()
parser.add_argument("--load_names", 
                    required=True, type=str, nargs="*")
parser.add_argument("--model_names",
                    required=True, type=str, nargs="*")
parser.add_argument("--save_img_name", type=str,
                    default="losses_plot.png")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plt_losses(train_losses_list, valid_losses_list, model_names,
               save_img_path):
    color_list = ["r", "g", "b", "c", "m", "y"]
    fig, ax = plt.subplots()
    ax.grid()
    for i, model_name in enumerate(model_names):
        train_losses = train_losses_list[i]
        valid_losses = valid_losses_list[i]
        ax.plot(train_losses, marker=".", markersize=3, 
                color=color_list[2*i], label=f"train loss ({model_name})")
        ax.plot(valid_losses, marker=".", markersize=3, 
                color=color_list[2*i+1], label=f"valid loss ({model_name})")
    ax.legend(loc="upper right")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(str(save_img_path))
    plt.close(fig) 
    return

if __name__ == "__main__":
    checkpoints_path = root_path / "checkpoints"
    load_names = args.load_names
    load_paths = [checkpoints_path / load_name for load_name in load_names]
    checkpoints = [torch.load(load_path, map_location=torch.device(device)) \
                   for load_path in load_paths]
    train_losses_list = [checkpoint["train_losses"] for checkpoint in checkpoints]
    valid_losses_list = [checkpoint["valid_losses"] for checkpoint in checkpoints]
    model_names = args.model_names
    save_img_name = args.save_img_name
    save_img_path = root_path / "images" / save_img_name
    plt_losses(train_losses_list, valid_losses_list, model_names,
               save_img_path)